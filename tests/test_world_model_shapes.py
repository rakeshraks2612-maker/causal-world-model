"""Tests for World Model Tensor Shapes, Dimensions, and Rollout Invariants."""

import pytest
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [10, 50, 100])
def test_encoder_shapes(batch_size: int, seq_len: int) -> None:
    """Test encoder posterior distribution shapes [B, T, d_z]."""
    model = CausalWorldModel()
    obs = torch.randn(batch_size, seq_len, 8)
    mask = torch.ones(batch_size, seq_len, 8)
    act = torch.randn(batch_size, seq_len, 4)
    inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

    dist, h_n = model.encode(inputs)
    assert dist.mean.shape == (batch_size, seq_len, 32)
    assert dist.logvar.shape == (batch_size, seq_len, 32)
    assert dist.std.shape == (batch_size, seq_len, 32)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_transition_step_shapes(batch_size: int) -> None:
    """Test single-step transition prior shapes [B, d_z]."""
    model = CausalWorldModel()
    z = torch.randn(batch_size, 32)
    act = torch.randn(batch_size, 4)

    next_dist = model.transition_step(z, act)
    assert next_dist.mean.shape == (batch_size, 32)
    assert next_dist.logvar.shape == (batch_size, 32)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [5, 20])
def test_decoder_shapes(batch_size: int, seq_len: int) -> None:
    """Test observation decoder reconstruction shapes [B, T, 8]."""
    model = CausalWorldModel()
    z = torch.randn(batch_size, seq_len, 32)

    obs_dist = model.decode(z)
    assert obs_dist.mean.shape == (batch_size, seq_len, 8)
    assert obs_dist.logvar.shape == (batch_size, seq_len, 8)
    assert obs_dist.std.shape == (batch_size, seq_len, 8)


def test_forward_pass_composite_shapes() -> None:
    """Test full training forward pass output shapes."""
    model = CausalWorldModel()
    batch_size = 4
    seq_len = 25

    inputs = ModelInputs(
        observations=torch.randn(batch_size, seq_len, 8),
        observation_mask=torch.ones(batch_size, seq_len, 8),
        actions=torch.randn(batch_size, seq_len, 4),
    )

    post_latents, prior_trans, recon_obs = model(inputs)

    assert post_latents.mean.shape == (batch_size, seq_len, 32)
    assert prior_trans.mean.shape == (batch_size, seq_len - 1, 32)
    assert recon_obs.mean.shape == (batch_size, seq_len, 8)


@pytest.mark.parametrize("horizon", [1, 5, 10, 20, 40])
def test_deterministic_forecast_shapes(horizon: int) -> None:
    """Test deterministic rollout forecast shapes across multi-horizon evaluation targets."""
    model = CausalWorldModel()
    history = ModelInputs(
        observations=torch.randn(2, 20, 8),
        observation_mask=torch.ones(2, 20, 8),
        actions=torch.randn(2, 20, 4),
    )
    future_acts = torch.randn(2, horizon, 4)

    rollout = model.forecast(history, future_acts, deterministic=True)
    assert rollout.latent_mean.shape == (2, horizon, 32)
    assert rollout.latent_variance.shape == (2, horizon, 32)
    assert rollout.observations.mean.shape == (2, horizon, 8)
    assert rollout.lower_bounds_90.shape == (2, horizon, 8)
    assert rollout.upper_bounds_90.shape == (2, horizon, 8)


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_monte_carlo_forecast_shapes(horizon: int) -> None:
    """Test Monte Carlo ensemble rollout shapes and particle aggregation."""
    model = CausalWorldModel()
    history = ModelInputs(
        observations=torch.randn(3, 15, 8),
        observation_mask=torch.ones(3, 15, 8),
        actions=torch.randn(3, 15, 4),
    )
    future_acts = torch.randn(3, horizon, 4)

    rollout = model.forecast(history, future_acts, num_particles=20, deterministic=False)
    assert rollout.latent_mean.shape == (3, horizon, 32)
    assert rollout.observations.mean.shape == (3, horizon, 8)
    assert rollout.lower_bounds_90.shape == (3, horizon, 8)
    assert rollout.upper_bounds_90.shape == (3, horizon, 8)
    # Upper bound should strictly exceed or equal lower bound
    assert torch.all(rollout.upper_bounds_90 >= rollout.lower_bounds_90)
