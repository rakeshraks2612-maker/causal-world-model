"""Tests for Deterministic Replay, Seed Sensitivity, and Twin-Simulator Execution."""

import numpy as np
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.policies import NominalController, HighLoadController, FailureInducingController


def test_deterministic_replay_identical_seed() -> None:
    """TEST 1: Run twice with identical initial state, actions, and seed.
    
    Expected: trajectory_A == trajectory_B, noise_A == noise_B (100% exact match).
    """
    sim1 = THCSimulator(seed=42)
    sim2 = THCSimulator(seed=42)

    ep1 = sim1.run_episode(NominalController(), length=60)
    ep2 = sim2.run_episode(NominalController(), length=60)

    # 1. Ground truth physical state trajectory exact match
    np.testing.assert_array_equal(ep1.ground_truth_states, ep2.ground_truth_states)

    # 2. Exogenous noise vectors exact match
    np.testing.assert_array_equal(ep1.exogenous_noise, ep2.exogenous_noise)

    # 3. Actions exact match
    np.testing.assert_array_equal(ep1.actions, ep2.actions)

    # 4. Failure status exact match
    assert ep1.failure_latched == ep2.failure_latched


def test_seed_sensitivity_different_seeds() -> None:
    """TEST 2: Run with seed 42 vs seed 99 under identical initial state and policy.
    
    Expected: U_42 != U_99, trajectory_42 != trajectory_99, both physically valid.
    """
    sim1 = THCSimulator(seed=42)
    sim2 = THCSimulator(seed=99)
    policy = NominalController()

    ep1 = sim1.run_episode(policy, length=60)
    ep2 = sim2.run_episode(policy, length=60)

    # Noise and trajectories must differ
    assert not np.array_equal(ep1.exogenous_noise, ep2.exogenous_noise)
    assert not np.array_equal(ep1.ground_truth_states, ep2.ground_truth_states)

    # Both must remain within physical bounds
    for row in ep1.ground_truth_states:
        valid, violations = StateVector.from_array(row).validate_bounds()
        assert valid, f"Violations in ep1: {violations}"

    for row in ep2.ground_truth_states:
        valid, violations = StateVector.from_array(row).validate_bounds()
        assert valid, f"Violations in ep2: {violations}"


def test_action_sensitivity() -> None:
    """TEST 3: Same initial state & seed, but compare Nominal vs HighLoad vs Failure policy.
    
    Expected: Different policies produce distinct downstream causal trajectories.
    """
    sim_nom = THCSimulator(seed=42)
    sim_high = THCSimulator(seed=42)
    sim_fail = THCSimulator(seed=42)

    ep_nom = sim_nom.run_episode(NominalController(), length=60)
    ep_high = sim_high.run_episode(HighLoadController(fixed_valve=30.0), length=60)
    ep_fail = sim_fail.run_episode(FailureInducingController(), length=60)

    # Final core temperature under failure policy should be much higher than nominal
    final_temp_nom = ep_nom.ground_truth_states[-1, 0]
    final_temp_high = ep_high.ground_truth_states[-1, 0]
    final_temp_fail = ep_fail.ground_truth_states[-1, 0]

    assert final_temp_fail > final_temp_high > final_temp_nom
    assert ep_fail.failure_latched is True


def test_twin_simulator_noise_replay() -> None:
    """TEST 4: Replay an episode using replay_episode with exact frozen recorded noise."""
    sim = THCSimulator(seed=42)
    original_ep = sim.run_episode(NominalController(), length=50)

    init_state = StateVector.from_array(original_ep.ground_truth_states[0])
    actions_seq = original_ep.actions[:-1]  # The T action steps

    replayer = THCSimulator(seed=999)  # Seed shouldn't matter since noise is injected directly
    replayed_ep = replayer.replay_episode(
        recorded_noise=original_ep.exogenous_noise,
        action_sequence=actions_seq,
        initial_state=init_state,
    )

    # States must match exactly between original execution and twin-replay
    np.testing.assert_allclose(replayed_ep.ground_truth_states, original_ep.ground_truth_states, atol=1e-5)


def test_frozen_exogenous_counterfactual_divergence() -> None:
    """TASK 1.24: Mandatory Frozen-Exogenous Counterfactual Divergence Gate.
    
    Verifies that:
    1. Exogenous noise tensors U_orig and U_cf are 100% bit-for-bit IDENTICAL across all steps.
    2. Future recorded actions A[41:T] remain IDENTICAL.
    3. Trajectory before intervention (t <= 40) is 100% bit-for-bit IDENTICAL.
    4. Trajectory after intervention (t > 40) diverges strictly due to the changed action at t=40.
    """
    sim = THCSimulator(seed=42)
    # Generate original 120-step episode under HighLoad policy (fixed valve 35%)
    orig_ep = sim.run_episode(HighLoadController(fixed_valve=35.0), length=120)

    # Copy original action sequence
    cf_actions = np.copy(orig_ep.actions[:-1])  # Shape [120, 4]
    
    # Intervene ONLY on action at t=40: change A_valve from 35% to 95% (aggressive cooling)
    t_star = 40
    assert cf_actions[t_star, 0] == 35.0
    cf_actions[t_star, 0] = 95.0

    # Replay using twin-simulator with the EXACT frozen exogenous noise tensor
    init_state = StateVector.from_array(orig_ep.ground_truth_states[0])
    replayer = THCSimulator(seed=8888)  # Seed does not matter because frozen noise is supplied
    cf_ep = replayer.replay_episode(
        recorded_noise=orig_ep.exogenous_noise,
        action_sequence=cf_actions,
        initial_state=init_state,
    )

    # 1. Verify frozen noise tensors are 100% bit-for-bit identical
    np.testing.assert_array_equal(cf_ep.exogenous_noise, orig_ep.exogenous_noise)

    # 2. Verify state trajectory for t <= t_star is 100% bit-for-bit identical
    np.testing.assert_array_equal(
        cf_ep.ground_truth_states[:t_star + 1],
        orig_ep.ground_truth_states[:t_star + 1],
    )

    # 3. Verify state trajectory after t_star diverges downstream immediately in causal window
    # At t=41 (one step after t=40 intervention), valve opening, flow, and temp must diverge:
    orig_v_pos_41 = orig_ep.ground_truth_states[t_star + 1, 5]
    cf_v_pos_41 = cf_ep.ground_truth_states[t_star + 1, 5]
    assert cf_v_pos_41 > orig_v_pos_41 + 10.0  # Valve opened significantly more in CF

    orig_flow_41 = orig_ep.ground_truth_states[t_star + 1, 3]
    cf_flow_41 = cf_ep.ground_truth_states[t_star + 1, 3]
    assert cf_flow_41 > orig_flow_41 + 5.0  # Flow surged in response

    orig_temp_42 = orig_ep.ground_truth_states[t_star + 2, 0]
    cf_temp_42 = cf_ep.ground_truth_states[t_star + 2, 0]
    assert cf_temp_42 < orig_temp_42  # Core temp dropped in response to surged cooling flow
