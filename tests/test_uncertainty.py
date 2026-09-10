"""Tests for Monte Carlo Particle Ensemble, Predictive Distribution, and Uncertainty Propagation."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.latent_state import LatentDistribution
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.uncertainty_metrics import evaluate_monte_carlo_uncertainty


def test_monte_carlo_particle_rollout():
    torch.manual_seed(42)
    B, H, act_dim, num_particles = 1, 10, 4, 25

    config = WorldModelConfig()
    d_z = config.model.latent_dim
    model = CausalWorldModel(config)
    model.eval()

    z_mean = torch.randn(B, d_z)
    z_logvar = torch.zeros(B, d_z)
    z_dist = LatentDistribution(mean=z_mean, logvar=z_logvar)

    actions = torch.randn(B, H, act_dim)

    traj = model.rollout_manager.rollout_monte_carlo(
        initial_z_dist=z_dist,
        actions=actions,
        num_samples=num_particles,
    )

    assert traj.latent_mean.shape == (B, H, d_z)
    assert traj.latent_variance.shape == (B, H, d_z)
    assert traj.observations.mean.shape == (B, H, 8)
    assert traj.observations.variance.shape == (B, H, 8)
    assert traj.lower_bounds_90.shape == (B, H, 8)
    assert traj.upper_bounds_90.shape == (B, H, 8)

    # Upper bounds must be strictly >= lower bounds
    assert torch.all(traj.upper_bounds_90 >= traj.lower_bounds_90)


def test_uncertainty_growth_with_horizon():
    """Verify that stochastic rollout variance generally accumulates over time."""
    torch.manual_seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)
    obs = np.random.uniform(20.0, 80.0, size=(T, 8))
    mask = np.ones((T, 8), dtype=bool)
    act = np.random.uniform(10.0, 90.0, size=(T, 4))
    ep = LearnerEpisode("ep_test", SplitType.TEST, timestamps, obs, mask, act)

    normalizer = ObservationNormalizer.fit([ep])
    model = CausalWorldModel(WorldModelConfig())

    report = evaluate_monte_carlo_uncertainty(
        model=model,
        episodes=[ep],
        normalizer=normalizer,
        horizons=[1, 5, 10, 20, 40],
        num_particles=20,
        context_length=40,
        stride=50,
    )

    h1_std = report.by_horizon[1].mean_predictive_std_norm
    h40_std = report.by_horizon[40].mean_predictive_std_norm

    # Horizon 40 predictive uncertainty is >= Horizon 1
    assert h40_std >= h1_std * 0.95
