"""Autoregressive Multi-Step Rollout and Forecasting Contracts."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.world_model.transition import BaseTransition
from prism.world_model.decoder import BaseDecoder
from prism.world_model.config import RolloutConfig


@dataclass
class RolloutTrajectory:
    """Container for multi-step autoregressive trajectory predictions."""

    latent_mean: Tensor                 # Shape: [B, H, d_z]
    latent_variance: Tensor             # Shape: [B, H, d_z]
    observations: ObservationDistribution # Mean & Variance across horizon: [B, H, 8]
    lower_bounds_90: Tensor             # Shape: [B, H, 8]
    upper_bounds_90: Tensor             # Shape: [B, H, 8]
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])


class RolloutManager:
    """Manages autoregressive rollout rollouts over transition and decoder models."""

    def __init__(
        self,
        transition: BaseTransition,
        decoder: BaseDecoder,
        config: Optional[RolloutConfig] = None,
    ) -> None:
        self.transition = transition
        self.decoder = decoder
        self.config = config or RolloutConfig()

    def rollout_deterministic(
        self,
        initial_z: Tensor,
        actions: Tensor,
    ) -> RolloutTrajectory:
        """Autoregressive rollout using deterministic mean transitions.
        
        Args:
            initial_z: Latent state at start of rollout Z_t [B, d_z]
            actions: Action sequence to execute [B, H, 4]
            
        Returns:
            RolloutTrajectory spanning H timesteps
        """
        batch_size, horizon, act_dim = actions.shape
        curr_z = initial_z

        z_means: List[Tensor] = []
        z_vars: List[Tensor] = []

        for step in range(horizon):
            act_step = actions[:, step]
            next_z_dist = self.transition(curr_z, act_step)
            curr_z = next_z_dist.mean  # Deterministic mode: propagate mean

            z_means.append(next_z_dist.mean)
            z_vars.append(next_z_dist.variance)

        all_z_mean = torch.stack(z_means, dim=1)  # [B, H, d_z]
        all_z_var = torch.stack(z_vars, dim=1)    # [B, H, d_z]

        obs_dist = self.decoder(all_z_mean)
        low90, high90 = obs_dist.confidence_interval(confidence=self.config.confidence_level)

        return RolloutTrajectory(
            latent_mean=all_z_mean,
            latent_variance=all_z_var,
            observations=obs_dist,
            lower_bounds_90=low90,
            upper_bounds_90=high90,
            horizons=self.config.horizons,
        )

    def rollout_monte_carlo(
        self,
        initial_z_dist: LatentDistribution,
        actions: Tensor,
        num_samples: Optional[int] = None,
    ) -> RolloutTrajectory:
        """Monte Carlo particle ensemble rollout propagating both latent and observation uncertainty.
        
        Args:
            initial_z_dist: Latent distribution at start of rollout q(Z_t) [B, d_z]
            actions: Action sequence to execute [B, H, 4]
            num_samples: Number of parallel stochastic particles N
            
        Returns:
            RolloutTrajectory with aggregated predictive mean and interval coverage
        """
        n_samples = num_samples or self.config.num_samples
        batch_size, horizon, act_dim = actions.shape

        # Initial particle ensemble: [N, B, d_z]
        curr_particles = initial_z_dist.rsample_n(n_samples)

        sampled_obs: List[Tensor] = []
        sampled_z: List[Tensor] = []

        # Expand actions for particle broadcast: [N, B, 4]
        for step in range(horizon):
            act_step = actions[:, step].unsqueeze(0).expand(n_samples, -1, -1)  # [N, B, 4]
            
            # Transition each particle: [N*B, d_z]
            flat_z = curr_particles.reshape(-1, curr_particles.shape[-1])
            flat_act = act_step.reshape(-1, act_dim)

            next_z_dist = self.transition(flat_z, flat_act)
            next_z_sample = next_z_dist.sample()

            # Reshape back to [N, B, d_z]
            curr_particles = next_z_sample.reshape(n_samples, batch_size, -1)
            sampled_z.append(curr_particles)

            # Decode observations for this step: [N*B, 8]
            step_obs_dist = self.decoder(next_z_sample)
            step_obs_sample = step_obs_dist.sample()
            sampled_obs.append(step_obs_sample.reshape(n_samples, batch_size, -1))

        # Tensor shapes: [N, B, H, d_z] and [N, B, H, 8]
        all_z_particles = torch.stack(sampled_z, dim=2)
        all_obs_particles = torch.stack(sampled_obs, dim=2)

        # Aggregate empirical statistics across particles
        z_mean = torch.mean(all_z_particles, dim=0)       # [B, H, d_z]
        z_var = torch.var(all_z_particles, dim=0)         # [B, H, d_z]

        obs_mean = torch.mean(all_obs_particles, dim=0)   # [B, H, 8]
        obs_var = torch.var(all_obs_particles, dim=0)     # [B, H, 8]
        obs_logvar = torch.log(obs_var + 1e-8)

        aggregated_obs_dist = ObservationDistribution(
            mean=obs_mean,
            logvar=obs_logvar,
        )

        # Empirical quantiles for predictive interval
        alpha = (1.0 - self.config.confidence_level) / 2.0
        low_q = float(alpha)
        high_q = float(1.0 - alpha)

        low90 = torch.quantile(all_obs_particles, low_q, dim=0)   # [B, H, 8]
        high90 = torch.quantile(all_obs_particles, high_q, dim=0) # [B, H, 8]

        return RolloutTrajectory(
            latent_mean=z_mean,
            latent_variance=z_var,
            observations=aggregated_obs_dist,
            lower_bounds_90=low90,
            upper_bounds_90=high90,
            horizons=self.config.horizons,
        )
