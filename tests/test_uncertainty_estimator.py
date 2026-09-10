"""Tests for Epistemic Uncertainty and Latent Novelty Scoring (Task 3.7)."""

from __future__ import annotations
import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.world_model.latent_state import LatentDistribution
from prism.uncertainty.estimator import (
    LatentManifoldDensityEstimator,
    EpistemicUncertaintyEstimator,
    UncertaintyEstimate,
)
from prism.uncertainty.calibrator import (
    compute_prediction_interval_coverage,
    compute_expected_calibration_error,
)


def test_latent_manifold_density_estimator():
    """Test fitting and Mahalanobis distance calculation."""
    d_z = 64
    rng = np.random.default_rng(42)
    train_latents = rng.normal(loc=0.0, scale=1.0, size=(200, d_z))
    
    estimator = LatentManifoldDensityEstimator(d_z=d_z, novelty_threshold=10.0)
    estimator.fit(train_latents)
    
    assert estimator.is_fitted
    assert estimator.mean.shape == (d_z,)
    assert estimator.inv_cov.shape == (d_z, d_z)
    
    # In-distribution point should have small distance
    in_dist_z = rng.normal(loc=0.0, scale=1.0, size=(1, d_z))
    d_in = estimator.compute_mahalanobis_distance(in_dist_z)[0]
    
    # Out-of-distribution point should have large distance
    ood_z = rng.normal(loc=10.0, scale=1.0, size=(1, d_z))
    d_ood = estimator.compute_mahalanobis_distance(ood_z)[0]
    
    assert d_ood > d_in
    assert d_ood > 10.0


def test_epistemic_uncertainty_and_abstention():
    """Test full rollout uncertainty estimation and abstention gating."""
    b3_dir = Path("artifacts/baseline_003")
    norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    
    estimator = EpistemicUncertaintyEstimator(
        world_model=model,
        normalizer=norm,
        novelty_threshold=10.0,
        variance_threshold=50.0,
    )
    
    # In-distribution initial latent
    z_mean = torch.zeros(1, 64)
    z_logvar = torch.zeros(1, 64)
    z_dist = LatentDistribution(mean=z_mean, logvar=z_logvar, min_std=0.01, max_std=10.0)
    actions = torch.ones(1, 40, 4) * 50.0
    
    unc = estimator.evaluate_rollout_uncertainty(initial_z_dist=z_dist, actions=actions, num_particles=10)
    
    assert isinstance(unc, UncertaintyEstimate)
    assert unc.aleatoric_variance.shape == (40, 8)
    assert unc.total_variance.shape == (40, 8)
    assert unc.latent_novelty_scores.shape == (40,)


def test_calibration_and_ece_metrics():
    """Test PICP, MPIW, and ECE computation."""
    y_true = np.array([[10.0, 20.0], [15.0, 25.0]])
    y_low = np.array([[8.0, 18.0], [12.0, 22.0]])
    y_high = np.array([[12.0, 22.0], [18.0, 28.0]])
    
    cal = compute_prediction_interval_coverage(y_true, y_low, y_high, variable_names=["T_core", "T_cool"])
    assert cal.picp_90 == 1.0
    assert cal.mpiw_90 == 5.0
    
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    ece = compute_expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 0.3
