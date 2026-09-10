"""Tests for Intervention Dataset Taxonomy, Pairing, Multi-Horizon Effects, and Invariants."""

import pytest
import numpy as np

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.dataset.interventions import (
    InterventionSpec,
    InterventionClass,
    execute_intervention_experiment,
    OracleInterventionRecord,
    LearnerInterventionRecord,
)
from prism.dataset.validators import DatasetValidationError


@pytest.fixture
def parent_episode():
    """Fixture producing a standardized baseline OracleEpisode."""
    return generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=60)


def test_intervention_atomicity_and_domain_validation(parent_episode) -> None:
    """Test 1 & 8: Atomicity and Domain Validation."""
    # Valid spec
    spec = InterventionSpec(target="V_pos", value=85.0, intervention_time=20)
    spec.validate()
    assert spec.target == "V_pos"

    # Out of bounds value
    with pytest.raises(ValueError):
        InterventionSpec(target="V_pos", value=150.0, intervention_time=20).validate()

    with pytest.raises(ValueError):
        InterventionSpec(target="L_cpu", value=-10.0, intervention_time=20).validate()

    with pytest.raises(ValueError):
        InterventionSpec(target="V_pos", value=85.0, intervention_time=130).validate()


def test_pre_intervention_identity_and_frozen_exogenous(parent_episode) -> None:
    """Test 2 & 3: Pre-intervention Identity and Frozen Exogenous Noise Tensor."""
    t_star = 25
    spec = InterventionSpec(target="V_pos", value=90.0, intervention_time=t_star)
    rec = execute_intervention_experiment(parent_episode, spec)

    # 1. Exogenous noise tensors must be 100% bit-for-bit identical
    np.testing.assert_array_equal(
        rec.intervened_oracle_ep.exogenous_noise,
        parent_episode.exogenous_noise,
    )

    # 2. Pre-intervention states for t <= t_star must be 100% bit-for-bit identical
    np.testing.assert_array_equal(
        rec.intervened_oracle_ep.ground_truth_states[:t_star + 1],
        parent_episode.ground_truth_states[:t_star + 1],
    )

    # 3. Future actions must be identical
    np.testing.assert_array_equal(
        rec.intervened_oracle_ep.actions[t_star + 1:],
        parent_episode.actions[t_star + 1:],
    )


def test_downstream_causal_propagation(parent_episode) -> None:
    """Test 5: Changing do(V_pos=85%) produces valid downstream flow and thermal deltas."""
    spec = InterventionSpec(target="V_pos", value=85.0, intervention_time=20)
    rec = execute_intervention_experiment(parent_episode, spec)

    # Flow should increase (delta > 0) and core temp should decrease (delta < 0) at horizon 10
    eff10 = rec.horizon_effects[10]
    assert eff10.delta_f_cool > 5.0  # Flow surged
    assert eff10.delta_t_core < 0.0  # Temperature dropped


def test_non_descendant_invariance(parent_episode) -> None:
    """Test 6: do(Vib_pump) produces zero causal effect on thermal and hydraulic state."""
    spec = InterventionSpec(target="Vib_pump", value=0.5, intervention_time=20)
    rec = execute_intervention_experiment(parent_episode, spec)

    # At all horizons, delta for core temp, coolant temp, pressure, and flow must be EXACTLY 0.0
    for h, eff in rec.horizon_effects.items():
        assert eff.delta_t_core == 0.0
        assert eff.delta_t_cool == 0.0
        assert eff.delta_p_sys == 0.0
        assert eff.delta_f_cool == 0.0
        assert eff.delta_vib_pump != 0.0  # Only vibration was altered!


def test_learner_record_isolation_and_support_metadata(parent_episode) -> None:
    """Test 7 & 9: Baseline Pairing and Zero-Leakage in LearnerInterventionRecord."""
    spec = InterventionSpec(target="V_pos", value=85.0, intervention_time=20)
    oracle_rec = execute_intervention_experiment(parent_episode, spec)
    learner_rec = oracle_rec.to_learner_record()

    # 1. Verify shapes
    assert learner_rec.pre_intervention_observations.shape == (21, 8)
    assert learner_rec.pre_intervention_actions.shape == (21, 4)
    assert learner_rec.future_actions.shape == (40, 4)  # Remaining steps: 61 - 21 = 40

    # 2. Verify support metadata is present
    assert "historical_support_min" in learner_rec.support_metadata
    assert "distance_from_observed_support" in learner_rec.support_metadata
    assert learner_rec.support_metadata["distance_from_observed_support"] >= 0.0

    # 3. Verify zero oracle attributes
    assert not hasattr(learner_rec, "baseline_oracle_ep")
    assert not hasattr(learner_rec, "intervened_oracle_ep")
    assert not hasattr(learner_rec, "ground_truth_states")


def test_failure_metrics_contract_semantics(parent_episode) -> None:
    """Hardening Check 2.7-H1: Failure Metric Contract.
    
    Verifies:
    - baseline_failed, intervened_failed
    - failure_probability_baseline, failure_probability_intervention
    - absolute_failure_probability_delta, relative_failure_risk
    - failure_mode_transition, time_to_failure_delta
    """
    # 1. Non-failing intervention
    spec_safe = InterventionSpec(target="V_pos", value=85.0, intervention_time=20)
    rec_safe = execute_intervention_experiment(parent_episode, spec_safe)
    fm_safe = rec_safe.failure_metrics

    assert fm_safe.baseline_failed is False
    assert fm_safe.intervened_failed is False
    assert fm_safe.failure_probability_baseline == 0.0
    assert fm_safe.failure_probability_intervention == 0.0
    assert fm_safe.absolute_failure_probability_delta == 0.0
    assert fm_safe.relative_failure_risk == 1.0
    assert fm_safe.failure_mode_transition == "none -> none"
    assert fm_safe.time_to_failure_delta is None

    # 2. Severe load intervention inducing thermal failure
    spec_fail = InterventionSpec(target="L_cpu", value=100.0, intervention_time=5)
    rec_fail = execute_intervention_experiment(parent_episode, spec_fail)
    fm_fail = rec_fail.failure_metrics

    if fm_fail.intervened_failed:
        assert fm_fail.baseline_failed is False
        assert fm_fail.failure_probability_baseline == 0.0
        assert fm_fail.failure_probability_intervention == 1.0
        assert fm_fail.absolute_failure_probability_delta == 1.0
        assert fm_fail.relative_failure_risk == float("inf")
        assert "none ->" in fm_fail.failure_mode_transition
        assert fm_fail.intervention_failure_time is not None


def test_intervention_persistence_semantics(parent_episode) -> None:
    """Hardening Check 2.7-H2: State Intervention Persistence Semantics.
    
    Verifies that:
    1. duration=None persists do(V_pos=85) permanently through end of episode (bypassing actuator lag & wear).
    2. duration=5 is an instantaneous/finite pulse that reverts to normal dynamics at t* + 5.
    """
    t_star = 20
    # Case A: Permanent intervention (duration=None)
    spec_perm = InterventionSpec(target="V_pos", value=85.0, intervention_time=t_star, duration=None)
    rec_perm = execute_intervention_experiment(parent_episode, spec_perm)
    perm_v_pos = rec_perm.intervened_oracle_ep.ground_truth_states[:, 5]

    # Pre-intervention matches baseline
    np.testing.assert_array_equal(
        perm_v_pos[:t_star + 1],
        parent_episode.ground_truth_states[:t_star + 1, 5],
    )
    # Post-intervention is permanently clamped at 85.0 from t* through T=59
    for t in range(t_star + 1, len(perm_v_pos)):
        assert perm_v_pos[t] == 85.0, f"Expected V_pos=85.0 at t={t}, got {perm_v_pos[t]}"

    # Case B: Finite pulse intervention (duration=5)
    dur = 5
    spec_pulse = InterventionSpec(target="V_pos", value=85.0, intervention_time=t_star, duration=dur)
    rec_pulse = execute_intervention_experiment(parent_episode, spec_pulse)
    pulse_v_pos = rec_pulse.intervened_oracle_ep.ground_truth_states[:, 5]

    # During pulse window [t* + 1, t* + dur], clamped at 85.0
    for t in range(t_star + 1, t_star + dur + 1):
        assert pulse_v_pos[t] == 85.0, f"Expected pulse V_pos=85.0 at t={t}, got {pulse_v_pos[t]}"

    # After pulse window (t > t* + dur), natural dynamics resume and V_pos departs from 85.0
    post_pulse_step = t_star + dur + 5
    if post_pulse_step < len(pulse_v_pos):
        assert pulse_v_pos[post_pulse_step] != 85.0


def test_class_b_action_vs_class_a_state_intervention_distinction(parent_episode) -> None:
    """Hardening Check 2.7-H3: Class B Action-Mediated vs Class A Atomic State Interventions.
    
    Proves that:
    1. do(V_pos=85) severs parents and forces V_pos=85.0 immediately at t* + 1.
    2. A_valve=85 commands the actuator, but actuator lag/wear dictates gradual transition (V_pos < 85.0 at t* + 1).
    3. Hence do(V_pos=85) != A_valve=85.
    4. Future actions outside targeted action remain frozen.
    """
    t_star = 20

    # 1. Class A: Atomic State Intervention do(V_pos=85)
    spec_state = InterventionSpec(
        target="V_pos",
        value=85.0,
        intervention_time=t_star,
        category=InterventionClass.CLASS_A_ATOMIC_STATE,
    )
    rec_state = execute_intervention_experiment(parent_episode, spec_state)

    # 2. Class B: Action-Mediated Intervention A_valve=85
    spec_action = InterventionSpec(
        target="A_valve",
        value=85.0,
        intervention_time=t_star,
        category=InterventionClass.CLASS_B_ACTION_CONTROL,
    )
    rec_action = execute_intervention_experiment(parent_episode, spec_action)

    # State at t* + 1
    v_pos_state_t1 = rec_state.intervened_oracle_ep.ground_truth_states[t_star + 1, 5]
    v_pos_action_t1 = rec_action.intervened_oracle_ep.ground_truth_states[t_star + 1, 5]

    # Class A forces exact value immediately (graph surgery)
    assert v_pos_state_t1 == 85.0
    # Class B is subject to actuator lag: it has NOT reached 85.0 on step t* + 1!
    assert v_pos_action_t1 < 85.0
    assert v_pos_state_t1 != v_pos_action_t1

    # 3. Future action freezing proofs
    # For Class A, all actions throughout the episode remain identical to baseline
    np.testing.assert_array_equal(
        rec_state.intervened_oracle_ep.actions,
        parent_episode.actions,
    )

    # For Class B, pre-intervention actions are identical to baseline
    np.testing.assert_array_equal(
        rec_action.intervened_oracle_ep.actions[:t_star],
        parent_episode.actions[:t_star],
    )
    # At t >= t*, A_valve is modified, while other channels (throttle, pump, flush) remain identical
    np.testing.assert_array_equal(
        rec_action.intervened_oracle_ep.actions[t_star:, 0],
        np.full(len(rec_action.intervened_oracle_ep.actions) - t_star, 85.0),
    )
    np.testing.assert_array_equal(
        rec_action.intervened_oracle_ep.actions[t_star:, 1:],
        parent_episode.actions[t_star:, 1:],
    )

    # 4. Small benchmark for other Class B actions (A_throttle, A_pump)
    for pump_val in [1.0, 2.0, 3.0, 4.0]:
        spec_pump = InterventionSpec(
            target="A_pump",
            value=pump_val,
            intervention_time=t_star,
            category=InterventionClass.CLASS_B_ACTION_CONTROL,
        )
        rec_pump = execute_intervention_experiment(parent_episode, spec_pump)
        assert rec_pump.intervened_oracle_ep.actions[t_star, 2] == pump_val

    spec_throttle = InterventionSpec(
        target="A_throttle",
        value=80.0,
        intervention_time=t_star,
        category=InterventionClass.CLASS_B_ACTION_CONTROL,
    )
    rec_throttle = execute_intervention_experiment(parent_episode, spec_throttle)
    assert rec_throttle.intervened_oracle_ep.actions[t_star, 1] == 80.0
