"""Tests for OOD Evaluation and Firewall Isolation."""

from __future__ import annotations
import pytest
import torch
import numpy as np
from pathlib import Path

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.ood import evaluate_ood_regimes


def test_ood_evaluation_firewall(tmp_path):
    """Verify that OOD evaluation loads regimes and computes metrics without altering model weights."""
    torch.manual_seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)

    test_ep = LearnerEpisode("ep_test", SplitType.TEST, timestamps, np.random.uniform(20.0, 80.0, (T, 8)), np.ones((T, 8), bool), np.random.uniform(10.0, 90.0, (T, 4)))

    # Create dummy OOD folder structure
    ood_dir = tmp_path / "ood"
    reg1_dir = ood_dir / "ood_1_extreme_ambient"
    reg1_dir.mkdir(parents=True)
    ood_ep = LearnerEpisode("ep_ood", SplitType.OOD, timestamps, np.random.uniform(40.0, 90.0, (T, 8)), np.ones((T, 8), bool), np.random.uniform(10.0, 90.0, (T, 4)))
    ood_ep.save_npz(reg1_dir / "ep_ood_001.npz")

    normalizer = ObservationNormalizer.fit([test_ep])
    model = CausalWorldModel(WorldModelConfig())
    model.eval()

    # Capture initial weights
    w_init = {k: v.clone() for k, v in model.state_dict().items()}

    report = evaluate_ood_regimes(
        model=model,
        test_episodes=[test_ep],
        ood_root_dir=ood_dir,
        normalizer=normalizer,
        horizons=[1, 5],
        num_particles=5,
    )

    # Assert weights untouched
    for k, v in model.state_dict().items():
        assert torch.equal(v, w_init[k]), f"Weight {k} modified during OOD evaluation!"

    assert "Test_InDistribution" in report.regimes
    assert "ood_1_extreme_ambient" in report.regimes
