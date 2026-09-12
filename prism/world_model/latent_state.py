"""Probabilistic Latent State and Observation Distribution Contracts."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
from torch import Tensor


@dataclass
class LatentDistribution:
    """Represents a diagonal Gaussian posterior or prior over the latent dynamical space: N(mu, diag(sigma^2))."""

    mean: Tensor        # Shape: [B, T, d_z] or [B, d_z]
    logvar: Tensor      # Shape: [B, T, d_z] or [B, d_z]
    min_std: float = 1e-3
    max_std: float = 10.0

    @property
    def std(self) -> Tensor:
        """Standard deviation with numerical stability clamps."""
        raw_std = torch.exp(0.5 * self.logvar)
        return torch.clamp(raw_std, min=self.min_std, max=self.max_std)

    @property
    def variance(self) -> Tensor:
        return self.std ** 2

    def sample(self, deterministic: bool = False) -> Tensor:
        """Sample latent vector using the reparameterization trick: Z = mu + sigma * eps."""
        if deterministic:
            return self.mean
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def rsample_n(self, num_samples: int) -> Tensor:
        """Sample multiple independent Monte Carlo draws: Shape [num_samples, B, (T), d_z]."""
        eps = torch.randn((num_samples,) + self.mean.shape, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean.unsqueeze(0) + self.std.unsqueeze(0) * eps

    def kl_divergence(self, prior: Optional[LatentDistribution] = None) -> Tensor:
        """Analytical KL divergence D_KL(q(Z) || p(Z)). Default prior is standard Normal N(0, I)."""
        if prior is None:
            # KL( N(mu, sigma^2) || N(0, I) ) = 0.5 * sum( mu^2 + sigma^2 - 1 - log(sigma^2) )
            kl = 0.5 * (self.mean ** 2 + self.variance - 1.0 - self.logvar)
        else:
            # KL( N(mu_1, sigma_1^2) || N(mu_2, sigma_2^2) )
            var_ratio = self.variance / (prior.variance + 1e-8)
            diff_sq = (self.mean - prior.mean) ** 2 / (prior.variance + 1e-8)
            kl = 0.5 * (var_ratio + diff_sq - 1.0 + prior.logvar - self.logvar)
        return torch.sum(kl, dim=-1)  # Sum over latent dimension d_z

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute Gaussian log-likelihood log p(value | mu, sigma^2)."""
        diff_sq = (value - self.mean) ** 2
        log_p = -0.5 * (math.log(2.0 * math.pi) + self.logvar + diff_sq / (self.variance + 1e-8))
        return torch.sum(log_p, dim=-1)


@dataclass
class ObservationDistribution:
    """Represents a diagonal Gaussian predictive distribution over the 8 observable sensor channels."""

    mean: Tensor        # Shape: [B, T, 8] or [B, 8]
    logvar: Tensor      # Shape: [B, T, 8] or [B, 8]
    min_std: float = 1e-3
    max_std: float = 10.0

    @property
    def std(self) -> Tensor:
        raw_std = torch.exp(0.5 * self.logvar)
        return torch.clamp(raw_std, min=self.min_std, max=self.max_std)

    @property
    def variance(self) -> Tensor:
        return self.std ** 2

    def sample(self, deterministic: bool = False) -> Tensor:
        if deterministic:
            return self.mean
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def masked_log_prob(
        self,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Gaussian log-likelihood over valid observation channels, with optional weighting."""
        diff_sq = (target - self.mean) ** 2
        element_log_p = -0.5 * (math.log(2.0 * math.pi) + self.logvar + diff_sq / (self.variance + 1e-8))

        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            element_log_p = element_log_p * mask_bool.float()

        if weights is not None:
            w = weights.unsqueeze(-1) if weights.ndim == element_log_p.ndim - 1 else weights
            element_log_p = element_log_p * w.float()

        return torch.sum(element_log_p, dim=-1)

    def confidence_interval(self, confidence: float = 0.90) -> Tuple[Tensor, Tensor]:
        """Compute symmetric confidence intervals [lower, upper] for given confidence level."""
        # Standard normal critical values
        z_scores = {
            0.80: 1.28155,
            0.90: 1.64485,
            0.95: 1.95996,
            0.99: 2.57583,
        }
        z = z_scores.get(confidence, 1.64485)
        margin = z * self.std
        return self.mean - margin, self.mean + margin


@dataclass
class LatentState:
    """Container holding a realized latent state Z, its distribution, and recurrent state."""

    z: Tensor                               # Realized latent vector [B, T, d_z] or [B, d_z]
    distribution: LatentDistribution        # q(Z | O, A)
    recurrent_state: Optional[Tensor] = None # Hidden state h from recurrent cell
