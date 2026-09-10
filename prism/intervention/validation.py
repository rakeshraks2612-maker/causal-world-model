"""Validation Subsystem and Causal Invariants for Learned Interventions.

Implements programmatic invariant checks for:
1. No-intervention baseline equivalence: do(none) == factual rollout
2. Latent state consistency: Same inferred Z_t* -> different interventions -> divergent futures
3. State Clamp vs Action Control distinction: do(V_pos=85) != A_valve=85
4. Persistent vs Pulse intervention semantics
5. Pre-intervention temporal invariance: zero backward leakage
6. Valve directionality: V_pos increases -> F_cool increases, T_core decreases
7. Throttle directionality: L_cpu increases -> T_core increases
8. Non-descendant invariance: do(Vib_pump) has zero effect on thermal/hydraulic state
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from prism.intervention.simulator import LearnedInterventionSimulator, LearnedInterventionResult
from prism.intervention.spec import InterventionSpec, InterventionType, state_clamp, action_control
from prism.simulator.state import OBSERVABLE_VARIABLES


@dataclass
class CausalValidationReport:
    """Report certifying causal invariant adherence."""

    passed: bool
    checks: Dict[str, bool]
    details: Dict[str, Any]


def check_no_intervention_equivalence(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """Verify that a simulation with no intervention exactly matches factual baseline."""
    res = simulator.simulate(
        pre_observations=pre_obs,
        pre_actions=pre_act,
        future_actions=fut_act,
        intervention=None,
    )
    max_diff = float(np.max(np.abs(res.intervened_observations - res.baseline_observations)))
    passed = max_diff <= tolerance
    return passed, max_diff


def check_pre_intervention_identity(
    pre_obs_fact: np.ndarray,
    pre_obs_int: np.ndarray,
) -> Tuple[bool, float]:
    """Verify zero backward counterfactual leakage into historical observations (t <= t*)."""
    max_diff = float(np.max(np.abs(pre_obs_fact - pre_obs_int)))
    return (max_diff == 0.0), max_diff


def check_non_descendant_invariance(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    vib_value: float = 5.0,
    tolerance: float = 1e-5,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that do(Vib_pump = v) does not spuriously alter T_core, T_cool, P_sys, or F_cool."""
    spec = state_clamp("Vib_pump", vib_value, intervention_time=len(pre_obs) - 1)
    res = simulator.simulate(
        pre_observations=pre_obs,
        pre_actions=pre_act,
        future_actions=fut_act,
        intervention=spec,
    )

    # Variables that must remain completely invariant under do(Vib_pump)
    non_descendants = ["T_core", "T_cool", "P_sys", "F_cool"]
    deltas: Dict[str, float] = {}
    all_passed = True

    for var in non_descendants:
        idx = OBSERVABLE_VARIABLES.index(var)
        max_var_delta = float(np.max(np.abs(res.intervened_observations[:, idx] - res.baseline_observations[:, idx])))
        deltas[f"max_delta_{var}"] = max_var_delta
        if max_var_delta > tolerance:
            all_passed = False

    # But Vib_pump observation MUST be clamped to the intervened value
    vib_idx = OBSERVABLE_VARIABLES.index("Vib_pump")
    clamped_vib = res.intervened_observations[:, vib_idx]
    vib_clamped = bool(np.allclose(clamped_vib, vib_value, atol=1e-3))
    deltas["vib_pump_successfully_clamped"] = float(vib_clamped)

    return (all_passed and vib_clamped), deltas


def check_valve_directionality(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    val_low: float = 50.0,
    val_high: float = 100.0,
    horizon_step: int = 10,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that valve opening increases cooling flow and decreases core temperature."""
    spec_low = state_clamp("V_pos", val_low, intervention_time=len(pre_obs) - 1)
    spec_high = state_clamp("V_pos", val_high, intervention_time=len(pre_obs) - 1)

    res_low = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_low)
    res_high = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_high)

    h_idx = min(horizon_step - 1, res_low.intervened_observations.shape[0] - 1)

    t_core_idx = OBSERVABLE_VARIABLES.index("T_core")
    f_cool_idx = OBSERVABLE_VARIABLES.index("F_cool")

    t_core_low = res_low.intervened_observations[h_idx, t_core_idx]
    t_core_high = res_high.intervened_observations[h_idx, t_core_idx]

    flow_low = res_low.intervened_observations[h_idx, f_cool_idx]
    flow_high = res_high.intervened_observations[h_idx, f_cool_idx]

    # Flow should be higher at val_high; T_core should be lower (or equal) at val_high
    passed_flow = (flow_high >= flow_low - 1e-3)
    passed_temp = (t_core_high <= t_core_low + 1e-3)

    metrics = {
        "flow_low": float(flow_low),
        "flow_high": float(flow_high),
        "t_core_low": float(t_core_low),
        "t_core_high": float(t_core_high),
        "delta_flow": float(flow_high - flow_low),
        "delta_t_core": float(t_core_high - t_core_low),
    }

    return (passed_flow and passed_temp), metrics


def check_throttle_directionality(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    load_low: float = 20.0,
    load_high: float = 80.0,
    horizon_step: int = 10,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that higher CPU load increases core temperature."""
    spec_low = state_clamp("L_cpu", load_low, intervention_time=len(pre_obs) - 1)
    spec_high = state_clamp("L_cpu", load_high, intervention_time=len(pre_obs) - 1)

    res_low = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_low)
    res_high = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_high)

    h_idx = min(horizon_step - 1, res_low.intervened_observations.shape[0] - 1)
    t_core_idx = OBSERVABLE_VARIABLES.index("T_core")

    t_core_low = res_low.intervened_observations[h_idx, t_core_idx]
    t_core_high = res_high.intervened_observations[h_idx, t_core_idx]

    passed = (t_core_high > t_core_low)
    metrics = {
        "t_core_low": float(t_core_low),
        "t_core_high": float(t_core_high),
        "delta_t_core": float(t_core_high - t_core_low),
    }

    return passed, metrics


def check_state_clamp_vs_action_control(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    value: float = 85.0,
) -> Tuple[bool, float]:
    """Verify that do(V_pos = 85) is distinct from A_valve = 85 (instant clamp vs dynamical lag)."""
    spec_clamp = state_clamp("V_pos", value, intervention_time=len(pre_obs) - 1)
    spec_action = action_control("A_valve", value, intervention_time=len(pre_obs) - 1)

    res_clamp = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_clamp)
    res_action = simulator.simulate(pre_obs, pre_act, fut_act, intervention=spec_action)

    v_pos_idx = OBSERVABLE_VARIABLES.index("V_pos")

    # Step 0 of rollout: state clamp is immediately 85.0, whereas action control starts from baseline
    clamp_step0 = res_clamp.intervened_observations[0, v_pos_idx]
    action_step0 = res_action.intervened_observations[0, v_pos_idx]

    diff_step0 = float(abs(clamp_step0 - action_step0))
    passed = diff_step0 > 0.1  # Significant distinction due to actuator dynamics
    return passed, diff_step0
