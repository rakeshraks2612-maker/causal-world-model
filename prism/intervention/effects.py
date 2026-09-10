"""Causal Effect Computation and Metrics for Learned World Model Interventions.

Calculates:
- Multi-horizon causal deltas: Delta Y(t* + h) = Y_int - Y_base for h in {1, 5, 10, 20, 40}
- Peak metrics: peak T_core, max P_sys, min F_cool
- Predicted failure probabilities, modes, and time-to-failure
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES


# Default safety threshold limits for failure detection
SAFETY_THRESHOLDS: Dict[str, float] = {
    "T_core_max": 105.0,     # Core thermal runaway threshold (°C)
    "T_cool_max": 95.0,      # Coolant boiling threshold (°C)
    "P_sys_max": 6.0,        # Overpressure burst limit (bar)
    "F_cool_min": 1.0,       # Flow blockage threshold (L/min)
    "Vib_pump_max": 15.0,    # Mechanical cavitation/bearing failure (mm/s)
}


@dataclass
class HorizonCausalDelta:
    """Causal effect deltas at a specific forecast horizon step h."""

    horizon: int
    delta_t_core: float
    delta_t_cool: float
    delta_p_sys: float
    delta_f_cool: float
    delta_l_cpu: float
    delta_v_pos: float
    delta_vib_pump: float
    delta_p_elec: float
    baseline_failed: bool = False
    intervened_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearnedFailureMetrics:
    """Predicted failure outcomes and delta metrics."""

    baseline_failed: bool
    intervened_failed: bool
    failure_probability_baseline: float
    failure_probability_intervention: float
    absolute_failure_probability_delta: float
    relative_failure_risk: float
    baseline_failure_time: Optional[int] = None
    intervention_failure_time: Optional[int] = None
    time_to_failure_delta: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalEffectSummary:
    """Complete summary of causal effects between baseline and intervened rollouts."""

    horizon_effects: Dict[int, HorizonCausalDelta]
    mean_deltas: Dict[str, float]
    peak_t_core_base: float
    peak_t_core_int: float
    delta_peak_t_core: float
    max_pressure_base: float
    max_pressure_int: float
    delta_max_pressure: float
    min_flow_base: float
    min_flow_int: float
    delta_min_flow: float
    failure_metrics: LearnedFailureMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_effects": {str(h): eff.to_dict() for h, eff in self.horizon_effects.items()},
            "mean_deltas": self.mean_deltas,
            "peak_t_core_base": self.peak_t_core_base,
            "peak_t_core_int": self.peak_t_core_int,
            "delta_peak_t_core": self.delta_peak_t_core,
            "max_pressure_base": self.max_pressure_base,
            "max_pressure_int": self.max_pressure_int,
            "delta_max_pressure": self.delta_max_pressure,
            "min_flow_base": self.min_flow_base,
            "min_flow_int": self.min_flow_int,
            "delta_min_flow": self.delta_min_flow,
            "failure_metrics": self.failure_metrics.to_dict(),
        }


def _check_trajectory_failures(
    obs: np.ndarray,
    t_star: int,
    thresholds: Dict[str, float],
) -> Tuple[bool, Optional[int]]:
    """Determine if a trajectory violates safety thresholds and find failure step."""
    horizon = obs.shape[0]
    for step in range(horizon):
        t_core = obs[step, 0]
        t_cool = obs[step, 1]
        p_sys = obs[step, 2]
        f_cool = obs[step, 3]
        vib = obs[step, 6]

        if (
            t_core >= thresholds.get("T_core_max", 105.0)
            or t_cool >= thresholds.get("T_cool_max", 95.0)
            or p_sys >= thresholds.get("P_sys_max", 6.0)
            or f_cool <= thresholds.get("F_cool_min", 1.0)
            or vib >= thresholds.get("Vib_pump_max", 15.0)
        ):
            return True, t_star + step
    return False, None


def compute_causal_effects(
    baseline_obs: Tensor | np.ndarray,
    intervened_obs: Tensor | np.ndarray,
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
    t_star: int = 0,
    thresholds: Optional[Dict[str, float]] = None,
) -> CausalEffectSummary:
    """Compute multi-horizon causal effect deltas and peak metrics.
    
    Args:
        baseline_obs: Predicted baseline observations [H, 8] or [1, H, 8] in physical units
        intervened_obs: Predicted intervened observations [H, 8] or [1, H, 8] in physical units
        horizons: Target horizons h in timesteps
        t_star: Timestep at which intervention begins
        thresholds: Optional custom safety thresholds
        
    Returns:
        CausalEffectSummary containing all deltas, peak changes, and failure assessments
    """
    thresh = thresholds or SAFETY_THRESHOLDS

    if isinstance(baseline_obs, torch.Tensor):
        base_arr = baseline_obs.detach().cpu().numpy()
    else:
        base_arr = np.asarray(baseline_obs)

    if isinstance(intervened_obs, torch.Tensor):
        int_arr = intervened_obs.detach().cpu().numpy()
    else:
        int_arr = np.asarray(intervened_obs)

    if base_arr.ndim == 3:
        base_arr = base_arr[0]
    if int_arr.ndim == 3:
        int_arr = int_arr[0]

    horizon_len = min(len(base_arr), len(int_arr))

    # 1. Failure checks
    base_failed, base_fail_time = _check_trajectory_failures(base_arr, t_star, thresh)
    int_failed, int_fail_time = _check_trajectory_failures(int_arr, t_star, thresh)

    base_prob = 1.0 if base_failed else 0.0
    int_prob = 1.0 if int_failed else 0.0
    abs_delta_prob = int_prob - base_prob
    rel_risk = int_prob / max(1e-4, base_prob) if base_failed else (float("inf") if int_failed else 1.0)
    tt_delta = None
    if base_failed and int_failed and base_fail_time is not None and int_fail_time is not None:
        tt_delta = int_fail_time - base_fail_time

    fail_metrics = LearnedFailureMetrics(
        baseline_failed=base_failed,
        intervened_failed=int_failed,
        failure_probability_baseline=base_prob,
        failure_probability_intervention=int_prob,
        absolute_failure_probability_delta=abs_delta_prob,
        relative_failure_risk=rel_risk,
        baseline_failure_time=base_fail_time,
        intervention_failure_time=int_fail_time,
        time_to_failure_delta=tt_delta,
    )

    # 2. Multi-horizon deltas
    horizon_effects: Dict[int, HorizonCausalDelta] = {}
    for h in horizons:
        idx = h - 1  # 1-indexed horizon to 0-indexed step
        if idx < horizon_len:
            delta = int_arr[idx] - base_arr[idx]
            horizon_effects[h] = HorizonCausalDelta(
                horizon=h,
                delta_t_core=float(delta[0]),
                delta_t_cool=float(delta[1]),
                delta_p_sys=float(delta[2]),
                delta_f_cool=float(delta[3]),
                delta_l_cpu=float(delta[4]),
                delta_v_pos=float(delta[5]),
                delta_vib_pump=float(delta[6]),
                delta_p_elec=float(delta[7]),
                baseline_failed=base_failed,
                intervened_failed=int_failed,
            )

    # 3. Peak metrics across the rollout
    peak_t_core_base = float(np.max(base_arr[:, 0]))
    peak_t_core_int = float(np.max(int_arr[:, 0]))

    max_p_base = float(np.max(base_arr[:, 2]))
    max_p_int = float(np.max(int_arr[:, 2]))

    min_f_base = float(np.min(base_arr[:, 3]))
    min_f_int = float(np.min(int_arr[:, 3]))

    # 4. Trajectory-wide mean deltas
    mean_deltas = {
        "mean_delta_t_core": float(np.mean(int_arr[:, 0] - base_arr[:, 0])),
        "max_delta_t_core": float(np.max(int_arr[:, 0] - base_arr[:, 0])),
        "min_delta_t_core": float(np.min(int_arr[:, 0] - base_arr[:, 0])),
        "mean_delta_t_cool": float(np.mean(int_arr[:, 1] - base_arr[:, 1])),
        "mean_delta_p_sys": float(np.mean(int_arr[:, 2] - base_arr[:, 2])),
        "mean_delta_f_cool": float(np.mean(int_arr[:, 3] - base_arr[:, 3])),
        "mean_delta_vib_pump": float(np.mean(int_arr[:, 6] - base_arr[:, 6])),
    }

    return CausalEffectSummary(
        horizon_effects=horizon_effects,
        mean_deltas=mean_deltas,
        peak_t_core_base=peak_t_core_base,
        peak_t_core_int=peak_t_core_int,
        delta_peak_t_core=float(peak_t_core_int - peak_t_core_base),
        max_pressure_base=max_p_base,
        max_pressure_int=max_p_int,
        delta_max_pressure=float(max_p_int - max_p_base),
        min_flow_base=min_f_base,
        min_flow_int=min_f_int,
        delta_min_flow=float(min_f_int - min_f_base),
        failure_metrics=fail_metrics,
    )
