"""Tests for Uncertainty Diagnostics and Latent Distribution Shifts."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.uncertainty_diagnostics import compute_uncertainty_diagnostics


def test_uncertainty_diagnostics_pipeline():
    torch.manual_seed(42)
    np.random.seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)

    # Synthetic train, test, and OOD episodes
    train_eps = [LearnerEpisode(f"tr_{i}", SplitType.TRAIN, timestamps, np.random.uniform(20.0, 80.0, (T, 8)), np.ones((T, 8), bool), np.random.uniform(10.0, 90.0, (T, 4))) for i in range(2)]
    test_eps = [LearnerEpisode(f"te_{i}", SplitType.TEST, timestamps, np.random.uniform(20.0, 80.0, (T, 8)), np.ones((T, 8), bool), np.random.uniform(10.0, 90.0, (T, 4))) for i in range(1)]
    ood_eps = {"ood_1_extreme_ambient": [LearnerEpisode("ood_0", SplitType.OOD, timestamps, np.random.uniform(50.0, 95.0, (T, 8)), np.ones((T, 8), bool), np.random.uniform(10.0, 90.0, (T, 4)))]}

    normalizer = ObservationNormalizer.fit(train_eps)
    model = CausalWorldModel(WorldModelConfig())

    report = compute_uncertainty_diagnostics(
        model=model,
        train_episodes=train_eps,
        test_episodes=test_eps,
        ood_regimes=ood_eps,
        normalizer=normalizer,
        num_particles=10,
        context_length=40,
        stride=40,
        horizon=40,
    )

    assert "Test_InDistribution" in report.regime_diagnostics
    assert "ood_1_extreme_ambient" in report.regime_diagnostics
    
    id_diag = report.regime_diagnostics["Test_InDistribution"]
    assert id_diag.latent_mahalanobis_distance >= 0.0
    assert id_diag.particle_latent_variance_h40 >= 0.0
    assert report.diagnosis_verdict != ""
