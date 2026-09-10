"""Master Causal World Model Architecture Binding Encoder, Transition, Decoder, and Rollout."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import LatentDistribution, ObservationDistribution, LatentState
from prism.world_model.encoder import BaseEncoder, GRUEncoder
from prism.world_model.transition import BaseTransition, MLPTransition
from prism.world_model.decoder import BaseDecoder, MLPDecoder
from prism.world_model.rollout import RolloutManager, RolloutTrajectory
from prism.world_model.losses import WorldModelLossCalculator, LossOutput


class CausalWorldModel(nn.Module):
    """PRISM Uncertainty-Aware Causal World Model.
    
    Integrates:
    - Sequence Encoder: (O_{<=t}, M_{<=t}, A_{<=t}) -> q_phi(Z_t | O, A)
    - Latent Transition Model: (Z_t, A_t) -> p_theta(Z_{t+1} | Z_t, A_t)
    - Observation Decoder: Z_t -> p_theta(O_t | Z_t)
    - Multi-Step Rollout & Uncertainty Engine: Autoregressive & Monte Carlo particle propagation
    """

    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()

        # 1. Sequence Encoder
        if self.config.encoder.type == "gru":
            self.encoder: BaseEncoder = GRUEncoder(self.config.model, self.config.encoder)
        else:
            raise ValueError(f"Unknown encoder type: {self.config.encoder.type}")

        # 2. Latent Transition Model
        self.transition: BaseTransition = MLPTransition(self.config.model, self.config.transition)

        # 3. Observation Decoder
        self.decoder: BaseDecoder = MLPDecoder(self.config.model, self.config.decoder)

        # 4. Rollout and Loss Utilities
        self.rollout_manager = RolloutManager(self.transition, self.decoder, self.config.rollout)
        self.loss_calculator = WorldModelLossCalculator(self.config.loss_weights)

    def encode(
        self,
        inputs: ModelInputs,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[LatentDistribution, Optional[Tensor]]:
        """Encode observable history into variational posterior distributions q_phi(Z_t | O, A)."""
        return self.encoder(inputs, hidden_state)

    def decode(self, z: Tensor) -> ObservationDistribution:
        """Decode latent state vector Z into observation distributions p_theta(O_t | Z_t)."""
        return self.decoder(z)

    def transition_step(self, z: Tensor, action: Tensor) -> LatentDistribution:
        """Predict prior transition distribution p_theta(Z_{t+1} | Z_t, A_t)."""
        return self.transition(z, action)

    def forward(
        self,
        inputs: ModelInputs,
    ) -> Tuple[LatentDistribution, LatentDistribution, ObservationDistribution]:
        """Full forward pass over sequence inputs for training.
        
        Returns:
            (posterior_latents [B, T], prior_transitions [B, T-1], reconstructed_obs [B, T])
        """
        # 1. Infer posterior latent distributions for all t in [0, T-1]: [B, T, d_z]
        posterior_latents, _ = self.encoder(inputs)

        # 2. Sample latent states via reparameterization trick
        z_samples = posterior_latents.sample(deterministic=not self.training)

        # 3. Compute transition predictions p(Z_{t+1} | Z_t, A_t) for t in [0, T-2]: [B, T-1, d_z]
        z_curr = z_samples[:, :-1]
        act_curr = inputs.actions[:, :-1]
        prior_transitions = self.transition(z_curr, act_curr)

        # 4. Reconstruct observations p(O_t | Z_t) for all t: [B, T, 8]
        reconstructed_obs = self.decoder(z_samples)

        return posterior_latents, prior_transitions, reconstructed_obs

    def compute_loss(self, inputs: ModelInputs) -> LossOutput:
        """Execute forward pass and compute complete variational loss."""
        post_latents, prior_trans, recon_obs = self.forward(inputs)
        return self.loss_calculator.compute_loss(
            posterior_latents=post_latents,
            prior_transitions=prior_trans,
            reconstructed_obs=recon_obs,
            target_obs=inputs.observations,
            observation_mask=inputs.observation_mask,
        )

    def forecast(
        self,
        history: ModelInputs,
        future_actions: Tensor,
        num_particles: Optional[int] = None,
        deterministic: bool = False,
    ) -> RolloutTrajectory:
        """Forecast future trajectory given historical context and planned future actions.
        
        Args:
            history: Historical observations, masks, and actions up to time t*
            future_actions: Action plan for timesteps [t*+1, t*+H] of shape [B, H, 4]
            num_particles: Number of Monte Carlo particles (if not deterministic)
            deterministic: If True, propagate mean states deterministically
            
        Returns:
            RolloutTrajectory with predicted observation distributions and intervals
        """
        # Encode history to obtain latent state at t*
        posterior_latents, _ = self.encoder(history)
        
        # Last timestep latent distribution: [B, d_z]
        z_t_star_dist = LatentDistribution(
            mean=posterior_latents.mean[:, -1],
            logvar=posterior_latents.logvar[:, -1],
            min_std=self.config.model.min_std,
            max_std=self.config.model.max_std,
        )

        if deterministic:
            return self.rollout_manager.rollout_deterministic(
                initial_z=z_t_star_dist.mean,
                actions=future_actions,
            )
        else:
            return self.rollout_manager.rollout_monte_carlo(
                initial_z_dist=z_t_star_dist,
                actions=future_actions,
                num_samples=num_particles,
            )
