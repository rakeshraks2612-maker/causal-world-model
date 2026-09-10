"""Tests for Dataset Contracts, Learner vs Oracle Boundary, and Pilot Generator."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.splits import get_partition_seed, generate_episode_id
from prism.dataset.validators import (
    validate_learner_isolation,
    validate_split_disjointness,
    validate_temporal_monotonicity,
    validate_mask_consistency,
    validate_oracle_episode_integrity,
    DatasetValidationError,
)
from prism.dataset.generator import generate_single_episode, generate_split_dataset
from prism.dataset.manifest import DatasetManifest


def test_learner_vs_oracle_isolation() -> None:
    """Verify strict learner isolation and prevention of latent leakage."""
    oracle_ep = generate_single_episode(SplitType.TRAIN, index=0, regime="nominal", length=30)
    learner_ep = oracle_ep.to_learner_episode()

    # Valid learner episode passes isolation check
    assert validate_learner_isolation(learner_ep) is True

    # 1. Learner does NOT have ground truth state or noise attributes
    assert not hasattr(learner_ep, "ground_truth_states")
    assert not hasattr(learner_ep, "exogenous_noise")
    assert not hasattr(learner_ep, "latent_states")

    # 2. Observations shape is strictly (T, 8) where T = length + 1 (initial state + 30 steps)
    assert learner_ep.observations.shape == (oracle_ep.length, 8)
    assert learner_ep.actions.shape == (oracle_ep.length, 4)
    assert learner_ep.length == 31

    # 3. Injecting a forbidden latent key in metadata raises ValueError/DatasetValidationError
    with pytest.raises((ValueError, DatasetValidationError)):
        LearnerEpisode(
            episode_id=learner_ep.episode_id,
            split=learner_ep.split,
            timestamps=learner_ep.timestamps,
            observations=learner_ep.observations,
            observation_mask=learner_ep.observation_mask,
            actions=learner_ep.actions,
            metadata={"policy": "Nominal", "T_amb": 35.0},  # FORBIDDEN!
        )


def test_split_disjointness_validator() -> None:
    """Verify split disjointness validator detects any episode overlap."""
    train_eps = [generate_single_episode(SplitType.TRAIN, index=i, regime="nominal", length=10) for i in range(3)]
    val_eps = [generate_single_episode(SplitType.VAL, index=i, regime="nominal", length=10) for i in range(2)]
    test_eps = [generate_single_episode(SplitType.TEST, index=i, regime="nominal", length=10) for i in range(2)]

    # Disjoint splits pass
    assert validate_split_disjointness(train_eps, val_eps, test_eps) is True

    # Overlapping episode raises DatasetValidationError
    corrupted_val = val_eps + [train_eps[0]]
    with pytest.raises(DatasetValidationError):
        validate_split_disjointness(train_eps, corrupted_val, test_eps)


def test_pilot_dataset_generation_pipeline() -> None:
    """Verify pilot generation of 20 Train, 5 Val, and 5 Test episodes with manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "pilot_data"
        regimes = {
            "nominal": 0.70,
            "moderate_load": 0.15,
            "moderate_ambient": 0.10,
            "moderate_wear": 0.05,
        }

        train_oracle, train_learner = generate_split_dataset(SplitType.TRAIN, 20, regimes, out_path, length=30)
        val_oracle, val_learner = generate_split_dataset(SplitType.VAL, 5, regimes, out_path, length=30)
        test_oracle, test_learner = generate_split_dataset(SplitType.TEST, 5, regimes, out_path, length=30)

        assert len(train_learner) == 20
        assert len(val_learner) == 5
        assert len(test_learner) == 5

        # Validate disjointness
        validate_split_disjointness(train_learner, val_learner, test_learner)

        # Validate file existence on disk
        for ep in train_learner:
            assert (out_path / "learner" / "train" / f"{ep.episode_id}.npz").exists()
        for ep in train_oracle:
            assert (out_path / "oracle" / "train" / f"{ep.episode_id}.npz").exists()

        # Generate and save manifest
        manifest = DatasetManifest.create(
            split_counts={"train": 20, "val": 5, "test": 5},
            regime_counts={"nominal": 21, "moderate_load": 4, "moderate_ambient": 3, "moderate_wear": 2},
        )
        manifest.save_json(out_path / "manifest.json")
        assert (out_path / "manifest.json").exists()

        loaded_manifest = DatasetManifest.load_json(out_path / "manifest.json")
        assert loaded_manifest.total_episodes == 30
