"""Integration Tests for Causal Invariant Equivalence and Distinction."""

from pathlib import Path
import numpy as np
import torch
import pytest

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.intervention.spec import state_clamp, action_control
from prism.intervention.validation import (
    check_no_intervention_equivalence,
    check_non_descendant_invariance,
    check_state_clamp_vs_action_control,
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


def test_no_intervention_equivalence(simulator):
    """Property 1: Intervening with None or no-op yields identical rollout to baseline."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 30
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    passed, max_diff = check_no_intervention_equivalence(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        tolerance=1e-5,
    )
    assert passed, f"No-intervention rollout diverged with max diff {max_diff}"


def test_same_inferred_state_different_interventions(simulator):
    """Property 2: Same initial state + different interventions -> diverging counterfactual futures."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 30
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    res_50 = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("V_pos", 50.0, intervention_time=t_star))
    res_100 = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("V_pos", 100.0, intervention_time=t_star))

    # Identical inferred latent state
    assert np.allclose(res_50.inferred_z_t_star, res_100.inferred_z_t_star)

    # Divergent trajectories
    traj_diff = np.max(np.abs(res_100.intervened_observations - res_50.intervened_observations))
    assert traj_diff > 1.0, f"Trajectories failed to diverge under different interventions: diff={traj_diff}"


def test_state_clamp_vs_action_control_distinction(simulator):
    """Property 3: do(V_pos=85) != A_valve=85 due to actuator response dynamics."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 30
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    passed, details = check_state_clamp_vs_action_control(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        value=85.0,
    )
    assert passed, f"do(V_pos=85) and A_valve=85 distinction failed: {details}"
    assert details["clamp_is_instant"], "do(V_pos=85) must clamp V_pos to 85.0 instantly at t*+1"
    assert details["action_has_lag"], "A_valve=85 must exhibit actuator lag at t*+1"


def test_non_descendant_invariance(simulator):
    """Property 9: Intervening on Vib_pump does not alter thermal/hydraulic state."""
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 30
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    passed, deltas = check_non_descendant_invariance(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        vib_value=5.0,
        tolerance=1e-5,
    )
    assert passed, f"Non-descendant invariance failed: {deltas}"
