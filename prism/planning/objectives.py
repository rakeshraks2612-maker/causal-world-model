"""Planning Objectives, Utility Functions, and Safety Constraints (Phase 4)."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from prism.intervention.spec import InterventionSpec
from prism.intervention.simulator import LearnedInterventionResult
from prism.simulator.state import OBSERVABLE_VARIABLES


@dataclass
class SafetyConstraints:
    """Operational safety thresholds for physical infrastructure."""

    t_core_max: float = 95.0        # Max allowable steady-state core temp (°C)
    t_core_critical: float = 105.0   # Hard thermal runaway limit (°C)
    t_cool_max: float = 85.0        # Max coolant temp (°C)
    p_sys_max: float = 5.5          # Max system pressure (bar)
    f_cool_min: float = 8.0         # Min coolant flow (L/min)
    max_latent_novelty: float = 15.0 # Max allowable OOD latent distance

    def is_safe(
        self,
        peak_t_core: float,
        peak_t_cool: float,
        max_pressure: float,
        min_flow: float,
        latent_novelty: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """Check whether predicted metrics satisfy all safety constraints."""
        violations = []
        if peak_t_core > self.t_core_max:
            violations.append(f"T_core peak ({peak_t_core:.1f}°C) exceeds safety threshold ({self.t_core_max:.1f}°C)")
        if peak_t_cool > self.t_cool_max:
            violations.append(f"T_cool peak ({peak_t_cool:.1f}°C) exceeds safety threshold ({self.t_cool_max:.1f}°C)")
        if max_pressure > self.p_sys_max:
            violations.append(f"P_sys max ({max_pressure:.2f} bar) exceeds limit ({self.p_sys_max:.2f} bar)")
        if min_flow < self.f_cool_min:
            violations.append(f"F_cool min ({min_flow:.1f} L/min) below minimum ({self.f_cool_min:.1f} L/min)")
        if latent_novelty > self.max_latent_novelty:
            violations.append(f"Latent novelty ({latent_novelty:.1f}) exceeds OOD limit ({self.max_latent_novelty:.1f})")

        return len(violations) == 0, violations


@dataclass
class UtilityWeights:
    """Weights for multi-objective decision optimization."""

    throughput_weight: float = 1.0     # Reward for high CPU load L_cpu
    thermal_penalty: float = 0.5       # Penalty for elevated core temperature
    power_cost_weight: float = 0.2     # Penalty for electrical power draw
    control_effort_weight: float = 0.1 # Penalty for large control action changes
    failure_penalty: float = 1000.0    # Massive penalty for critical safety violation


@dataclass
class CandidateEvaluation:
    """Comprehensive evaluation score and metrics for a candidate intervention."""

    candidate_id: str
    spec: InterventionSpec
    is_safe: bool
    safety_violations: List[str]
    utility_score: float
    peak_t_core: float
    max_pressure: float
    min_flow: float
    mean_cpu_load: float
    mean_power: float
    failure_probability: float
    latent_novelty: float
    causal_delta_t_core: float
    causal_delta_f_cool: float
    simulation_result: Optional[LearnedInterventionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target": self.spec.target,
            "value": self.spec.value,
            "is_safe": self.is_safe,
            "safety_violations": self.safety_violations,
            "utility_score": float(self.utility_score),
            "peak_t_core": float(self.peak_t_core),
            "max_pressure": float(self.max_pressure),
            "min_flow": float(self.min_flow),
            "mean_cpu_load": float(self.mean_cpu_load),
            "mean_power": float(self.mean_power),
            "failure_probability": float(self.failure_probability),
            "latent_novelty": float(self.latent_novelty),
            "causal_delta_t_core": float(self.causal_delta_t_core),
            "causal_delta_f_cool": float(self.causal_delta_f_cool),
        }


class PlanningObjective:
    """Evaluates candidates using multi-objective utility scoring."""

    def __init__(
        self,
        constraints: Optional[SafetyConstraints] = None,
        weights: Optional[UtilityWeights] = None,
    ) -> None:
        self.constraints = constraints or SafetyConstraints()
        self.weights = weights or UtilityWeights()

    def evaluate_candidate(
        self,
        spec: InterventionSpec,
        sim_result: LearnedInterventionResult,
        latent_novelty: float = 0.0,
        baseline_actions: Optional[np.ndarray] = None,
    ) -> CandidateEvaluation:
        """Score candidate intervention rollout."""
        obs = sim_result.intervened_observations  # [H, 8]
        peak_t_core = float(np.max(obs[:, 0]))
        peak_t_cool = float(np.max(obs[:, 1]))
        max_p_sys = float(np.max(obs[:, 2]))
        min_f_cool = float(np.min(obs[:, 3]))
        mean_l_cpu = float(np.mean(obs[:, 4]))
        mean_power = float(np.mean(obs[:, 7]))

        is_safe, violations = self.constraints.is_safe(
            peak_t_core=peak_t_core,
            peak_t_cool=peak_t_cool,
            max_pressure=max_p_sys,
            min_flow=min_f_cool,
            latent_novelty=latent_novelty,
        )

        fail_prob = 1.0 if (peak_t_core >= self.constraints.t_core_critical or max_p_sys >= 6.0) else 0.0

        # Multi-objective utility:
        # U = w_load * Load - w_therm * max(0, T_core - 70) - w_power * Power - w_fail * fail_prob
        thermal_excess = max(0.0, peak_t_core - 70.0)
        utility = (
            self.weights.throughput_weight * (mean_l_cpu / 100.0)
            - self.weights.thermal_penalty * (thermal_excess / 10.0)
            - self.weights.power_cost_weight * (mean_power / 50.0)
            - (self.weights.failure_penalty * fail_prob if not is_safe else 0.0)
        )

        # Deltas at horizon h=20 or last horizon
        h_target = 20 if 20 in sim_result.effects.horizon_effects else list(sim_result.effects.horizon_effects.keys())[-1]
        eff_h = sim_result.effects.horizon_effects[h_target]

        cand_id = f"cand_{spec.target}_{int(spec.value)}"

        return CandidateEvaluation(
            candidate_id=cand_id,
            spec=spec,
            is_safe=is_safe,
            safety_violations=violations,
            utility_score=utility,
            peak_t_core=peak_t_core,
            max_pressure=max_p_sys,
            min_flow=min_f_cool,
            mean_cpu_load=mean_l_cpu,
            mean_power=mean_power,
            failure_probability=fail_prob,
            latent_novelty=latent_novelty,
            causal_delta_t_core=eff_h.delta_t_core,
            causal_delta_f_cool=eff_h.delta_f_cool,
            simulation_result=sim_result,
        )
