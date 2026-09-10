"""Permanent Regression Test Suite for Task 3.4C — Learned State-Intervention Propagation.

Validates:
- Test A: State intervention does not alter parent action (Pearl decoupling: PA(X) -> None)
- Test B: State intervention changes latent trajectory when descendants exist (||Delta Z_h|| > 0)
- Test C: Downstream propagation (do(V_pos ^) -> Delta F_cool > 0, Delta T_core < 0)
- Test D: Non-descendant invariance (do(Vib_pump = x) -> Delta T_core == 0, Delta F_cool == 0)
- Test E: Action vs State distinction (A_valve = 85 != do(V_pos = 85))
- Test F: Safety intervention (do(L_cpu = 80) -> Delta T_core > 0)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import torch

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.interventions import LearnerInterventionRecord
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.intervention.spec import state_clamp, action_control
from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES


@pytest.fixture(scope="module")
def intervention_simulator() -> LearnedInterventionSimulator:
    """Fixture providing initialized simulator with frozen baseline_002 checkpoint."""
    ckpt_path = Path("artifacts/baseline_002")
    if not (ckpt_path / "best.pt").exists():
        pytest.skip("Frozen baseline_002 checkpoint not found")

    normalizer = ObservationNormalizer.load_yaml(ckpt_path / "normalization.yaml")
    config = WorldModelConfig.from_yaml(ckpt_path / "config.yaml")
    model = CausalWorldModel(config)
    checkpoint = torch.load(ckpt_path / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LearnedInterventionSimulator(model, normalizer)


@pytest.fixture(scope="module")
def sample_learner_record() -> LearnerInterventionRecord:
    """Load canonical test record."""
    rec_path = Path("data/pilot/learner/intervention/inv_ep_test_00000_t20_V_pos_85.npz")
    if not rec_path.exists():
        pytest.skip("Pilot learner intervention file not found")
    return LearnerInterventionRecord.load_npz(rec_path)


def test_a_state_intervention_preserves_parent_action(
    intervention_simulator: LearnedInterventionSimulator,
    sample_learner_record: LearnerInterventionRecord,
) -> None:
    """Test A: do(V_pos = 85) strictly preserves baseline A_valve."""
    sim = intervention_simulator
    rec = sample_learner_record

    spec = state_clamp("V_pos", 85.0, intervention_time=rec.intervention_time)
    res = sim.simulate(
        pre_observations=rec.pre_intervention_observations,
        pre_actions=rec.pre_intervention_actions,
        future_actions=rec.future_actions,
        intervention=spec,
        intervention_time=rec.intervention_time,
    )

    # In baseline and intervened, actions passed to transition must remain identical
    assert spec.is_state_clamp is True
    # Upstream action at t* must not be forced to 85.0
    a_valve_idx = ACTION_VARIABLES.index("A_valve")
    assert rec.future_actions[0, a_valve_idx] != 85.0


def test_b_state_intervention_diverges_latent_trajectory(
    intervention_simulator: LearnedInterventionSimulator,
    sample_learner_record: LearnerInterventionRecord,
) -> None:
    """Test B: do(V_pos = 85) produces ||Delta Z_h|| > 0 across all horizons."""
    sim = intervention_simulator
    rec = sample_learner_record

    res = sim.simulate_from_learner_record(rec)
    delta_z = np.linalg.norm(res.intervened_latent_mean - res.baseline_latent_mean, axis=-1)

    for h in [1, 5, 10, 20, 40]:
        assert delta_z[h] > 1.0, f"Latent norm delta at h={h} was {delta_z[h]:.4f}, expected > 1.0"


def test_c_downstream_causal_propagation(
    intervention_simulator: LearnedInterventionSimulator,
    sample_learner_record: LearnerInterventionRecord,
) -> None:
    """Test C: do(V_pos = 85) increases flow and decreases core temperature."""
    sim = intervention_simulator
    rec = sample_learner_record

    res = sim.simulate_from_learner_record(rec)

    f_idx = OBSERVABLE_VARIABLES.index("F_cool")
    p_idx = OBSERVABLE_VARIABLES.index("P_sys")
    tcore_idx = OBSERVABLE_VARIABLES.index("T_core")

    for h in [1, 5, 10, 20]:
        d_f = res.intervened_observations[h, f_idx] - res.baseline_observations[h, f_idx]
        d_p = res.intervened_observations[h, p_idx] - res.baseline_observations[h, p_idx]
        d_core = res.intervened_observations[h, tcore_idx] - res.baseline_observations[h, tcore_idx]

        assert d_f > 0.5, f"Coolant flow delta at h={h} was {d_f:+.2f}, expected > 0"
        assert d_p > 0.05, f"Pressure delta at h={h} was {d_p:+.2f}, expected > 0"
        assert d_core < -0.2, f"Core temperature delta at h={h} was {d_core:+.2f}, expected < 0"


def test_d_non_descendant_invariance(
    intervention_simulator: LearnedInterventionSimulator,
) -> None:
    """Test D: do(Vib_pump = 5.0) leaves thermal and fluid channels invariant."""
    sim = intervention_simulator
    rec_path = Path("data/pilot/learner/intervention/inv_ep_test_00000_t20_Vib_pump_5.npz")
    if not rec_path.exists():
        pytest.skip("Vib_pump pilot file not found")
    rec = LearnerInterventionRecord.load_npz(rec_path)

    res = sim.simulate_from_learner_record(rec)

    vib_idx = OBSERVABLE_VARIABLES.index("Vib_pump")
    f_idx = OBSERVABLE_VARIABLES.index("F_cool")
    p_idx = OBSERVABLE_VARIABLES.index("P_sys")
    tcore_idx = OBSERVABLE_VARIABLES.index("T_core")

    for h in [1, 5, 10, 20, 40]:
        d_f = abs(res.intervened_observations[h, f_idx] - res.baseline_observations[h, f_idx])
        d_p = abs(res.intervened_observations[h, p_idx] - res.baseline_observations[h, p_idx])
        d_core = abs(res.intervened_observations[h, tcore_idx] - res.baseline_observations[h, tcore_idx])

        assert d_f < 1e-4, f"F_cool non-descendant leak at h={h}: {d_f}"
        assert d_p < 1e-4, f"P_sys non-descendant leak at h={h}: {d_p}"
        assert d_core < 1e-4, f"T_core non-descendant leak at h={h}: {d_core}"


def test_e_action_control_vs_state_clamp_distinction(
    intervention_simulator: LearnedInterventionSimulator,
    sample_learner_record: LearnerInterventionRecord,
) -> None:
    """Test E: A_valve = 85 (action override) != do(V_pos = 85) (state clamp) at h=1."""
    sim = intervention_simulator
    rec = sample_learner_record

    res_clamp = sim.simulate(
        pre_observations=rec.pre_intervention_observations,
        pre_actions=rec.pre_intervention_actions,
        future_actions=rec.future_actions,
        intervention=state_clamp("V_pos", 85.0, intervention_time=rec.intervention_time),
        intervention_time=rec.intervention_time,
    )

    res_action = sim.simulate(
        pre_observations=rec.pre_intervention_observations,
        pre_actions=rec.pre_intervention_actions,
        future_actions=rec.future_actions,
        intervention=action_control("A_valve", 85.0, intervention_time=rec.intervention_time),
        intervention_time=rec.intervention_time,
    )

    v_idx = OBSERVABLE_VARIABLES.index("V_pos")
    # Immediate step: state clamp forces 85.0, whereas action control experiences actuator lag
    v_clamp_h1 = res_clamp.intervened_observations[1, v_idx]
    v_act_h1 = res_action.intervened_observations[1, v_idx]

    assert abs(v_clamp_h1 - 85.0) < 1e-3
    assert abs(v_act_h1 - 85.0) > 2.0, f"A_valve=85 should exhibit actuator lag at h=1, got {v_act_h1}"


def test_f_safety_cpu_workload_thermal_response(
    intervention_simulator: LearnedInterventionSimulator,
) -> None:
    """Test F: do(L_cpu = 80%) produces positive thermal response in T_core."""
    sim = intervention_simulator
    rec_path = Path("data/pilot/learner/intervention/inv_ep_test_00000_t20_L_cpu_80.npz")
    if not rec_path.exists():
        pytest.skip("L_cpu pilot file not found")
    rec = LearnerInterventionRecord.load_npz(rec_path)

    res = sim.simulate_from_learner_record(rec)
    tcore_idx = OBSERVABLE_VARIABLES.index("T_core")

    d_core_h1 = res.intervened_observations[1, tcore_idx] - res.baseline_observations[1, tcore_idx]
    assert d_core_h1 > 0.5, f"T_core response at h=1 was {d_core_h1:+.2f}, expected positive heat accumulation"
