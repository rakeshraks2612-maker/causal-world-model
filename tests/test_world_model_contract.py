"""Tests for World Model Architectural Contracts, Latent Distributions, Losses, and Zero-Leakage."""

import pytest
import numpy as np
import torch
from pathlib import Path

from prism.dataset.schema import SplitType, LearnerEpisode
from prism.dataset.generator import generate_single_episode
from prism.dataset.validators import DatasetValidationError
from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.world_model.losses import WorldModelLossCalculator
from prism.world_model.uncertainty import evaluate_calibration, compute_prediction_intervals
from prism.world_model.model import CausalWorldModel


def test_config_from_yaml() -> None:
    """Test loading Master WorldModelConfig from YAML."""
    cfg_path = Path("configs/world_model.yaml")
    assert cfg_path.exists()
    
    cfg = WorldModelConfig.from_yaml(cfg_path)
    assert cfg.model.obs_dim == 8
    assert cfg.model.action_dim == 4
    assert cfg.model.latent_dim == 32
    assert cfg.encoder.type == "gru"
    assert cfg.transition.residual is True
    assert cfg.decoder.learn_variance is True
    assert cfg.loss_weights.lambda_obs == 1.0


def test_model_inputs_validation() -> None:
    """Test shape validation and forbidden key leakage protection in ModelInputs."""
    # Valid tensors
    obs = torch.randn(2, 20, 8)
    mask = torch.ones(2, 20, 8)
    act = torch.randn(2, 20, 4)

    inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)
    assert inputs.batch_size == 2
    assert inputs.seq_len == 20

    # Invalid obs channel count
    with pytest.raises(ValueError, match="observations last dimension must be 8"):
        ModelInputs(observations=torch.randn(2, 20, 12), observation_mask=torch.ones(2, 20, 12), actions=act)

    # Invalid action channel count
    with pytest.raises(ValueError, match="actions last dimension must be 4"):
        ModelInputs(observations=obs, observation_mask=mask, actions=torch.randn(2, 20, 5))

    # Forbidden latent metadata key
    with pytest.raises(DatasetValidationError, match="Forbidden latent key"):
        ModelInputs(observations=obs, observation_mask=mask, actions=act, metadata={"T_amb": 25.0})


def test_model_inputs_from_learner_episode() -> None:
    """Test converting LearnerEpisode into ModelInputs tensor."""
    ep_oracle = generate_single_episode(SplitType.TRAIN, index=0, regime="nominal", length=30)
    ep_learner = ep_oracle.to_learner_episode()

    inputs = ModelInputs.from_learner_episode(ep_learner)
    assert inputs.observations.shape == (1, 31, 8)
    assert inputs.observation_mask.shape == (1, 31, 8)
    assert inputs.actions.shape == (1, 31, 4)
    assert inputs.timestamps.shape == (1, 31)


def test_latent_distribution_math_and_reparameterization() -> None:
    """Test Gaussian latent distribution properties, KL divergence, and reparameterization."""
    mean = torch.zeros(4, 10, 32, requires_grad=True)
    logvar = torch.zeros(4, 10, 32, requires_grad=True)
    dist = LatentDistribution(mean=mean, logvar=logvar)

    # Standard Normal N(0, I) has exact KL = 0.0 against itself
    kl_prior = dist.kl_divergence(prior=None)
    assert kl_prior.shape == (4, 10)
    np.testing.assert_allclose(kl_prior.detach().numpy(), 0.0, atol=1e-5)

    # Reparameterization sample supports gradient flow
    sample = dist.sample(deterministic=False)
    loss = torch.sum(sample ** 2)
    loss.backward()
    assert mean.grad is not None
    assert logvar.grad is not None

    # Deterministic mode returns exact mean
    det_sample = dist.sample(deterministic=True)
    np.testing.assert_array_equal(det_sample.detach().numpy(), mean.detach().numpy())


def test_observation_distribution_masked_log_prob() -> None:
    """Test observation log-prob with masking."""
    mean = torch.zeros(2, 5, 8)
    logvar = torch.zeros(2, 5, 8)
    dist = ObservationDistribution(mean=mean, logvar=logvar)

    targets = torch.zeros(2, 5, 8)
    mask_full = torch.ones(2, 5, 8)
    mask_half = torch.zeros(2, 5, 8)
    mask_half[:, :, :4] = 1.0  # Only first 4 channels observed

    log_p_full = dist.masked_log_prob(targets, mask_full)
    log_p_half = dist.masked_log_prob(targets, mask_half)

    assert log_p_full.shape == (2, 5)
    assert log_p_half.shape == (2, 5)
    # Masking half the channels halves the summed log-likelihood
    np.testing.assert_allclose(log_p_half.numpy(), 0.5 * log_p_full.numpy(), rtol=1e-4)


def test_uncertainty_calibration_metrics() -> None:
    """Test empirical calibration and confidence interval contracts."""
    # Synthetic calibrated predictions
    mean = torch.zeros(100, 8)
    logvar = torch.zeros(100, 8)  # std = 1.0
    dist = ObservationDistribution(mean=mean, logvar=logvar)

    # Standard Normal targets
    targets = torch.randn(100, 8)
    cal_results = evaluate_calibration(dist, targets, confidence_levels=(0.90,))
    
    assert 0.90 in cal_results
    metrics_list = cal_results[0.90]
    assert len(metrics_list) == 8
    for m in metrics_list:
        assert 0.0 <= m.empirical_coverage <= 1.0
        assert m.expected_confidence == 0.90
        assert m.mean_interval_width > 0.0


def test_world_model_loss_backward_pass() -> None:
    """Test end-to-end forward and loss backward pass through CausalWorldModel."""
    model = CausalWorldModel()
    model.train()

    obs = torch.randn(2, 15, 8, requires_grad=False)
    mask = torch.ones(2, 15, 8, requires_grad=False)
    act = torch.randn(2, 15, 4, requires_grad=False)
    inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

    loss_output = model.compute_loss(inputs)
    assert torch.isfinite(loss_output.total_loss)
    assert "loss/total" in loss_output.metrics
    assert "loss/obs_nll" in loss_output.metrics
    assert "loss/trans_kl" in loss_output.metrics

    loss_output.total_loss.backward()

    # Check encoder, transition, and decoder gradients exist
    for p in model.encoder.parameters():
        assert p.grad is not None
    for p in model.transition.parameters():
        assert p.grad is not None
    for p in model.decoder.parameters():
        assert p.grad is not None
