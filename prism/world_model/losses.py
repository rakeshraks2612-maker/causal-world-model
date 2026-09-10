"""World Model Loss Functions and Training Objective Contracts."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.world_model.config import LossWeightsConfig


@dataclass
class LossOutput:
    """Container holding total loss and individual loss components for logging."""

    total_loss: Tensor
    obs_loss: Tensor
    trans_loss: Tensor
    kl_loss: Tensor
    metrics: Dict[str, float]


class WorldModelLossCalculator:
    """Computes the complete variational world model loss objective.
    
    L = lambda_obs * L_obs + lambda_trans * L_trans + beta_kl * L_kl
    """

    def __init__(self, weights: Optional[LossWeightsConfig] = None) -> None:
        self.weights = weights or LossWeightsConfig()

    def compute_loss(
        self,
        posterior_latents: LatentDistribution,       # q_phi(Z_t | O, A) [B, T, d_z]
        prior_transitions: LatentDistribution,       # p_theta(Z_{t+1} | Z_t, A_t) [B, T-1, d_z]
        reconstructed_obs: ObservationDistribution,  # p_theta(O_t | Z_t) [B, T, 8]
        target_obs: Tensor,                          # Ground truth observations [B, T, 8]
        observation_mask: Tensor,                    # Binary mask [B, T, 8]
    ) -> LossOutput:
        """Compute the full variational ELBO loss across a sequence batch.
        
        Args:
            posterior_latents: Variational posterior q(Z_t) over full sequence [B, T, d_z]
            prior_transitions: Transition model predictions p(Z_{t+1} | Z_t, A_t) [B, T-1, d_z]
            reconstructed_obs: Decoder predictions p(O_t | Z_t) [B, T, 8]
            target_obs: True observation targets [B, T, 8]
            observation_mask: Sensor availability mask [B, T, 8]
            
        Returns:
            LossOutput containing scalar total_loss and decomposed terms
        """
        # 1. Observation Reconstruction Loss: Masked Negative Log-Likelihood
        # masked_log_prob returns sum over observed channels: [B, T]
        log_p_obs = reconstructed_obs.masked_log_prob(target_obs, observation_mask)
        # Average over batch and valid time steps
        obs_nll = -torch.mean(log_p_obs)

        # 2. Transition Consistency Loss: KL( q(Z_{t+1}) || p(Z_{t+1} | Z_t, A_t) )
        # Posterior slice for t=1..T-1: [B, T-1, d_z]
        post_next = LatentDistribution(
            mean=posterior_latents.mean[:, 1:],
            logvar=posterior_latents.logvar[:, 1:],
            min_std=posterior_latents.min_std,
            max_std=posterior_latents.max_std,
        )
        # Analytical Gaussian KL divergence between posterior and transition prior: [B, T-1]
        trans_kl = post_next.kl_divergence(prior_transitions)
        trans_loss = torch.mean(trans_kl)

        # 3. Latent Regularization: KL( q(Z_t) || N(0, I) )
        prior_kl = posterior_latents.kl_divergence(prior=None)  # [B, T]
        kl_loss = torch.mean(prior_kl)

        # Total Weighted Loss
        total_loss = (
            self.weights.lambda_obs * obs_nll
            + self.weights.lambda_trans * trans_loss
            + self.weights.beta_kl * kl_loss
        )

        metrics = {
            "loss/total": float(total_loss.detach().item()),
            "loss/obs_nll": float(obs_nll.detach().item()),
            "loss/trans_kl": float(trans_loss.detach().item()),
            "loss/prior_kl": float(kl_loss.detach().item()),
        }

        return LossOutput(
            total_loss=total_loss,
            obs_loss=obs_nll,
            trans_loss=trans_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )
