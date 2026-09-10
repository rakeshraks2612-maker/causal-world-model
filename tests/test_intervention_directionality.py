"""Integration Tests for Physical Directionality of Learned Interventions."""

from pathlib import Path
import numpy as np
import torch
import pytest

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.intervention.spec import state_clamp
from prism.intervention.validation import (
    check_valve_directionality,
    check_throttle_directionality,
)
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode


@pytest.fixture(scope="module")
def simulator():
    art_dir = Path("artifacts/baseline_002")
    if not (art_dir / "best.pt").exists():
        pytest.skip("baseline_002 checkpoint not found")

    normalizer = ObservationNormalizer.load_yaml(art_dir / "normalization.yaml")
    config = WorldModelConfig.from_yaml(art_dir / "config.yaml")
    model = CausalWorldModel(config)
    checkpoint = torch.load(art_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LearnedInterventionSimulator(model, normalizer)


def test_valve_opening_directionality(simulator):
    """Test that increasing valve opening increases flow and decreases/cools core temperature."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 40
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    passed, metrics = check_valve_directionality(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        val_low=50.0,
        val_high=100.0,
        horizon_step=10,
    )

    assert passed, f"Valve directionality check failed: {metrics}"
    assert metrics["delta_flow"] >= 0.0, f"Expected flow increase, got {metrics['delta_flow']}"
    assert metrics["delta_t_core"] <= 0.0, f"Expected cooling, got {metrics['delta_t_core']}"


def test_cpu_workload_directionality(simulator):
    """Test that higher CPU utilization increases core temperature."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 40
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    passed, metrics = check_throttle_directionality(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        load_low=20.0,
        load_high=80.0,
        horizon_step=10,
    )

    assert passed, f"Throttle directionality check failed: {metrics}"
    assert metrics["delta_l_cpu"] > 0.0, f"Expected positive delta_l_cpu, got {metrics['delta_l_cpu']}"
