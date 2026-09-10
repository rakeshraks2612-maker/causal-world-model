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


from prism.planning.safety_constraints import (
    SafetySeverity,
    SafetyViolationReason,
    DecisionSafetyConfig,
    SafetyResult,
    SafetyConstraintEngine,
)
from prism.planning.cost_model import DecisionCostConfig, ActionCostModel, CostBreakdown


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
    safety_result: Optional[SafetyResult] = None
    cost_breakdown: Optional[CostBreakdown] = None
    simulation_result: Optional[LearnedInterventionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target": self.spec.target,
            "value": self.spec.value,
            "is_safe": self.is_safe,
            "safety_violations": self.safety_violations,
            "safety_result": self.safety_result.to_dict() if self.safety_result else None,
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
            "cost_breakdown": self.cost_breakdown.to_dict() if self.cost_breakdown else None,
        }



class PlanningObjective:
    """Evaluates candidates using the formal ActionCostModel."""

    def __init__(
        self,
        config: Optional[DecisionCostConfig] = None,
        constraints: Optional[SafetyConstraints] = None,
        weights: Optional[UtilityWeights] = None,
    ) -> None:
        if config is not None:
            self.config = config
            self.constraints = constraints or SafetyConstraints(
                t_core_max=self.config.thermal_hard_limit,
                p_sys_max=self.config.pressure_hard_limit,
                f_cool_min=self.config.flow_hard_limit,
                max_latent_novelty=self.config.latent_novelty_hard_limit,
            )
        elif constraints is not None:
            self.constraints = constraints
            self.config = DecisionCostConfig(
                thermal_hard_limit=constraints.t_core_max,
                pressure_hard_limit=constraints.p_sys_max,
                flow_hard_limit=constraints.f_cool_min,
                latent_novelty_hard_limit=constraints.max_latent_novelty,
            )
        else:
            self.config = DecisionCostConfig()
            self.constraints = SafetyConstraints(
                t_core_max=self.config.thermal_hard_limit,
                p_sys_max=self.config.pressure_hard_limit,
                f_cool_min=self.config.flow_hard_limit,
                max_latent_novelty=self.config.latent_novelty_hard_limit,
            )

        self.cost_model = ActionCostModel(self.config)
        self.weights = weights or UtilityWeights(
            throughput_weight=self.config.performance_weight,
            thermal_penalty=self.config.thermal_risk_weight,
            power_cost_weight=0.2,
            control_effort_weight=self.config.valve_actuation_weight,
        )

    def evaluate_candidate(
        self,
        spec: InterventionSpec,
        sim_result: LearnedInterventionResult,
        latent_novelty: float = 0.0,
        baseline_actions: Optional[np.ndarray | List[float]] = None,
        candidate_id: Optional[str] = None,
        is_compound_intervention: bool = False,
    ) -> CandidateEvaluation:
        """Score candidate intervention rollout using the formal cost model."""
        obs = sim_result.intervened_observations  # [H, 8]
        peak_t_core = float(np.max(obs[:, 0]))
        max_p_sys = float(np.max(obs[:, 2]))
        min_f_cool = float(np.min(obs[:, 3]))
        mean_l_cpu = float(np.mean(obs[:, 4]))
        mean_power = float(np.mean(obs[:, 7]))

        # Factual baseline action at t*
        if baseline_actions is not None:
            base_act = [float(x) for x in np.asarray(baseline_actions).flatten()[:4]]
        else:
            base_act = [50.0, 100.0, 2.0, 0.0]

        # Candidate action at t* initialized to baseline action
        cand_act = list(base_act)
        if spec is not None:
            if spec.target == "A_valve":
                cand_act[0] = float(spec.value)
            elif spec.target == "A_throttle":
                cand_act[1] = float(spec.value)
            elif spec.target == "A_pump":
                cand_act[2] = float(spec.value)
            elif spec.target == "A_flush":
                cand_act[3] = float(spec.value)

        breakdown = self.cost_model.evaluate_cost(
            peak_t_core=peak_t_core,
            max_pressure=max_p_sys,
            min_flow=min_f_cool,
            mean_cpu_load=mean_l_cpu,
            actions_at_t_star=cand_act,
            baseline_actions_at_t_star=base_act,
            latent_novelty=latent_novelty,
            is_compound_intervention=is_compound_intervention,
        )

        fail_prob = 1.0 if (peak_t_core >= 105.0 or max_p_sys >= 6.0) else 0.0

        # Deltas at horizon h=20 or last horizon
        h_target = 20 if 20 in sim_result.effects.horizon_effects else list(sim_result.effects.horizon_effects.keys())[-1]
        eff_h = sim_result.effects.horizon_effects[h_target]

        cand_id = candidate_id or (f"cand_{spec.target}_{int(spec.value)}" if spec is not None else "cand_do_nothing")


        return CandidateEvaluation(
            candidate_id=cand_id,
            spec=spec,
            is_safe=breakdown.is_safe,
            safety_violations=breakdown.safety_violations,
            utility_score=breakdown.net_utility,
            peak_t_core=peak_t_core,
            max_pressure=max_p_sys,
            min_flow=min_f_cool,
            mean_cpu_load=mean_l_cpu,
            mean_power=mean_power,
            failure_probability=fail_prob,
            latent_novelty=latent_novelty,
            causal_delta_t_core=eff_h.delta_t_core,
            causal_delta_f_cool=eff_h.delta_f_cool,
            safety_result=breakdown.safety_result,
            cost_breakdown=breakdown,
            simulation_result=sim_result,
        )

