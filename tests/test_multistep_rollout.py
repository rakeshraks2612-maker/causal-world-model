"""Tests for Multi-Step Rollout Horizons and Shapes across Models."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.training.metrics import MLPDynamicsBaseline
from prism.evaluation.forecasting import (
    evaluate_prism_multistep,
    evaluate_mlp_multistep,
    evaluate_persistence_multistep,
)


@pytest.fixture
def synthetic_episodes():
    torch.manual_seed(42)
    np.random.seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)

    eps = []
    for i in range(2):
        obs = np.random.uniform(20.0, 80.0, size=(T, 8))
        mask = np.ones((T, 8), dtype=bool)
        act = np.random.uniform(10.0, 90.0, size=(T, 4))

        ep = LearnerEpisode(
            episode_id=f"ep_test_{i}",
            split=SplitType.TEST,
            timestamps=timestamps,
            observations=obs,
            observation_mask=mask,
            actions=act,
        )
        eps.append(ep)
    return eps


def test_prism_multistep_horizons_and_shapes(synthetic_episodes):
    normalizer = ObservationNormalizer.fit(synthetic_episodes)
    model = CausalWorldModel(WorldModelConfig())
    horizons = [1, 5, 10, 20, 40]

    report = evaluate_prism_multistep(
        model=model,
        episodes=synthetic_episodes,
        normalizer=normalizer,
        horizons=horizons,
        context_length=40,
        stride=10,
        mode="open_loop",
        deterministic=True,
    )

    assert report.horizons == horizons
    for h in horizons:
        assert h in report.by_horizon
        h_sum = report.by_horizon[h]
        assert h_sum.horizon == h
        assert len(h_sum.per_channel) == 8
        assert "T_core" in h_sum.per_channel
        assert h_sum.raw_overall_mae >= 0.0


def test_mlp_and_persistence_multistep(synthetic_episodes):
    normalizer = ObservationNormalizer.fit(synthetic_episodes)
    mlp = MLPDynamicsBaseline()
    horizons = [1, 5, 10, 20, 40]

    mlp_report = evaluate_mlp_multistep(
        mlp_model=mlp,
        episodes=synthetic_episodes,
        normalizer=normalizer,
        horizons=horizons,
        context_length=40,
        stride=10,
    )
    pers_report = evaluate_persistence_multistep(
        episodes=synthetic_episodes,
        normalizer=normalizer,
        horizons=horizons,
        context_length=40,
        stride=10,
    )

    assert mlp_report.horizons == horizons
    assert pers_report.horizons == horizons
    for h in horizons:
        assert h in mlp_report.by_horizon
        assert h in pers_report.by_horizon
