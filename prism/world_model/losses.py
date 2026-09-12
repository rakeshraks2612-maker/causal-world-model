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
    rollout_loss: Optional[Tensor] = None


class WorldModelLossCalculator:
    """Computes the complete variational world model loss objective.
    
    L = lambda_obs * L_obs + lambda_trans * L_trans + beta_kl * L_kl + lambda_rollout * L_rollout
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
        rollout_obs: Optional[List[ObservationDistribution]] = None,
        rollout_targets: Optional[List[Tuple[Tensor, Tensor]]] = None,
        timestep_weights: Optional[Tensor] = None,   # [B, T] or [B, T, 1] weighting
    ) -> LossOutput:
        """Compute the full variational ELBO loss across a sequence batch.
        
        Args:
            posterior_latents: Variational posterior q(Z_t) over full sequence [B, T, d_z]
            prior_transitions: Transition model predictions p(Z_{t+1} | Z_t, A_t) [B, T-1, d_z]
            reconstructed_obs: Decoder predictions p(O_t | Z_t) [B, T, 8]
            target_obs: True observation targets [B, T, 8]
            observation_mask: Sensor availability mask [B, T, 8]
            rollout_obs: Optional list of predicted ObservationDistributions at rollout horizons [1..K]
            rollout_targets: Optional list of (target_obs, observation_mask) tuples at rollout horizons [1..K]
            timestep_weights: Optional sample/timestep weights for safety-critical weighting
            
        Returns:
            LossOutput containing scalar total_loss and decomposed terms
        """
        # 1. Observation Reconstruction Loss: Masked Negative Log-Likelihood with safety weights
        log_p_obs = reconstructed_obs.masked_log_prob(target_obs, observation_mask, weights=timestep_weights)
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

        # 4. Multi-step Autoregressive Rollout Loss
        rollout_nll = torch.tensor(0.0, device=obs_nll.device)
        if rollout_obs is not None and rollout_targets is not None and len(rollout_obs) > 0:
            step_nlls = []
            for r_obs, (r_target, r_mask) in zip(rollout_obs, rollout_targets):
                r_log_p = r_obs.masked_log_prob(r_target, r_mask)
                step_nlls.append(-torch.mean(r_log_p))
            if step_nlls:
                rollout_nll = torch.stack(step_nlls).mean()

        # Total Weighted Loss
        total_loss = (
            self.weights.lambda_obs * obs_nll
            + self.weights.lambda_trans * trans_loss
            + self.weights.beta_kl * kl_loss
            + self.weights.lambda_rollout * rollout_nll
        )

        metrics = {
            "loss/total": float(total_loss.detach().item()),
            "loss/obs_nll": float(obs_nll.detach().item()),
            "loss/trans_kl": float(trans_loss.detach().item()),
            "loss/prior_kl": float(kl_loss.detach().item()),
            "loss/rollout_nll": float(rollout_nll.detach().item()),
        }

        return LossOutput(
            total_loss=total_loss,
            obs_loss=obs_nll,
            trans_loss=trans_loss,
            kl_loss=kl_loss,
            metrics=metrics,
            rollout_loss=rollout_nll,
        )
