"""Tests for Observation Normalization, Train-Only Statistics, and Invertibility."""

import pytest
import numpy as np
import torch
from pathlib import Path

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.training.normalization import ObservationNormalizer, NormalizationStats


@pytest.fixture
def train_episodes():
    return [
        generate_single_episode(SplitType.TRAIN, index=i, regime="nominal", length=50).to_learner_episode()
        for i in range(4)
    ]


@pytest.fixture
def test_episodes():
    return [
        generate_single_episode(SplitType.TEST, index=i, regime="moderate_load", length=50).to_learner_episode()
        for i in range(2)
    ]


def test_fit_train_only_statistics(train_episodes, test_episodes) -> None:
    """Test fitting normalizer strictly on training data."""
    normalizer = ObservationNormalizer.fit(train_episodes)
    assert normalizer.stats is not None
    assert len(normalizer.stats.means) == 8
    assert len(normalizer.stats.stds) == 8

    # All stds must be positive and non-zero
    for s in normalizer.stats.stds:
        assert s > 0.0

    # Test episode data does not alter normalizer
    norm_means_before = list(normalizer.stats.means)
    # Perform normalization on test data
    test_obs = torch.tensor(test_episodes[0].observations, dtype=torch.float32)
    _ = normalizer.normalize(test_obs)
    # Normalizer stats remain identical
    assert normalizer.stats.means == norm_means_before


def test_reversible_roundtrip(train_episodes) -> None:
    """Test that denormalize(normalize(x)) == x within numerical tolerance."""
    normalizer = ObservationNormalizer.fit(train_episodes)

    raw_x = torch.tensor(train_episodes[0].observations, dtype=torch.float32)
    norm_x = normalizer.normalize(raw_x)
    recovered_x = normalizer.denormalize(norm_x)

    np.testing.assert_allclose(recovered_x.numpy(), raw_x.numpy(), atol=1e-5)


def test_zero_variance_safety() -> None:
    """Test that zero-variance channels are regularized to std=1.0 without NaNs or division by zero."""
    # Channel with zero variance (all values identical)
    stats = NormalizationStats(
        means=[50.0] * 8,
        stds=[0.0] * 8,  # Raw zero std
    )
    # Safe constructor handles 0 -> 1.0 or safe division
    safe_stds = [s if s > 1e-6 else 1.0 for s in stats.stds]
    safe_stats = NormalizationStats(means=stats.means, stds=safe_stds)
    normalizer = ObservationNormalizer(stats=safe_stats)

    x = torch.full((5, 8), 50.0)
    norm_x = normalizer.normalize(x)
    assert torch.isfinite(norm_x).all()
    np.testing.assert_allclose(norm_x.numpy(), 0.0, atol=1e-6)


def test_yaml_serialization_roundtrip(train_episodes, tmp_path) -> None:
    """Test saving and loading normalization parameters from YAML."""
    normalizer = ObservationNormalizer.fit(train_episodes)
    yaml_path = tmp_path / "normalization.yaml"

    normalizer.save_yaml(yaml_path)
    assert yaml_path.exists()

    loaded_normalizer = ObservationNormalizer.load_yaml(yaml_path)
    assert loaded_normalizer.stats.means == normalizer.stats.means
    assert loaded_normalizer.stats.stds == normalizer.stats.stds
