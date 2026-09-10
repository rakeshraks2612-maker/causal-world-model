"""Tests for Action Branching and Causal Divergence Invariants."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.action_branching import evaluate_action_branching


def test_action_branching_invariants():
    torch.manual_seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)
    obs = np.random.uniform(20.0, 80.0, size=(T, 8))
    mask = np.ones((T, 8), dtype=bool)
    act = np.random.uniform(10.0, 90.0, size=(T, 4))
    ep = LearnerEpisode("ep_ref", SplitType.TEST, timestamps, obs, mask, act)

    normalizer = ObservationNormalizer.fit([ep])
    model = CausalWorldModel(WorldModelConfig())

    report = evaluate_action_branching(
        model=model,
        reference_episode=ep,
        normalizer=normalizer,
        horizon=40,
        context_length=40,
    )

    # 1. Invariant: Pre-branch state is identical
    assert report.valve_experiment.pre_branch_identical
    assert report.throttle_experiment.pre_branch_identical

    # 2. Invariant: Trajectories diverge post-branch
    assert report.valve_experiment.post_branch_divergent
    assert report.throttle_experiment.post_branch_divergent

    # 3. Branch counts
    assert len(report.valve_experiment.branches) == 3
    assert len(report.throttle_experiment.branches) == 3
