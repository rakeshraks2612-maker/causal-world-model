"""Formal Action-Cost Model and Utility Objective Configuration (Task 5.2).

Defines:
1. DecisionCostConfig: Centralized, transparent configuration for operational costs,
   performance rewards, thermal/pressure risk penalties, and hard/soft boundaries.
2. ActionCostModel: Evaluator computing fine-grained cost breakdowns:
   - Performance benefit (Compute throughput)
   - Thermal risk penalty
   - Pressure risk penalty
   - Pump operating cost
   - Valve actuation displacement cost
   - Throttle loss penalty
   - Flush activation cost
   - Multi-action complexity penalty
3. Strict Lexicographical Safety Gating:
   Hard safety boundaries disqualify candidates before utility ranking.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class DecisionCostConfig:
    """Canonical configuration for operational costs, utility weights, and safety limits."""

    # 1. Hard Safety Limits (Strict Rejection Boundary)
    thermal_hard_limit: float = 95.0        # Max permissible steady-state core temp (°C)
    pressure_hard_limit: float = 5.5        # Max permissible system pressure (bar)
    flow_hard_limit: float = 8.0            # Min permissible coolant flow (L/min)
    latent_novelty_hard_limit: float = 15.0 # Max allowable OOD latent distance

    # 2. Soft Operating Limits (Risk Penalty Activation Thresholds)
    thermal_soft_limit: float = 75.0        # Thermal penalty begins above 75°C
    pressure_soft_limit: float = 4.5        # Pressure penalty begins above 4.5 bar
    flow_soft_limit: float = 15.0           # Sub-optimal flow penalty below 15 L/min

    # 3. Objective Weights
    performance_weight: float = 1.0         # Reward per unit of normalized compute throughput [0, 1]
    thermal_risk_weight: float = 0.5        # Penalty scaling for (T_core - T_soft) / 10
    pressure_risk_weight: float = 0.3       # Penalty scaling for (P_sys - P_soft) / 1.0
    pump_cost_weight: float = 0.10          # Energy cost scaling for pump stage (stage / 4)
    valve_actuation_weight: float = 0.05    # Mechanical wear penalty per 50% valve displacement
    throttle_cost_weight: float = 2.0       # Penalty scaling for sacrificing computational capacity
    flush_cost_weight: float = 0.50         # Operational cost for triggering coolant flush
    complexity_weight: float = 0.05         # Penalty for multi-action compound interventions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CostBreakdown:
    """Detailed decomposition of utility and cost components for an intervention."""

    performance_benefit: float
    thermal_penalty: float
    pressure_penalty: float
    pump_cost: float
    actuation_cost: float
    throttle_cost: float
    flush_cost: float
    complexity_penalty: float
    total_cost: float
    net_utility: float
    is_safe: bool
    safety_violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "performance_benefit": float(self.performance_benefit),
            "thermal_penalty": float(self.thermal_penalty),
            "pressure_penalty": float(self.pressure_penalty),
            "pump_cost": float(self.pump_cost),
            "actuation_cost": float(self.actuation_cost),
            "throttle_cost": float(self.throttle_cost),
            "flush_cost": float(self.flush_cost),
            "complexity_penalty": float(self.complexity_penalty),
            "total_cost": float(self.total_cost),
            "net_utility": float(self.net_utility),
            "is_safe": self.is_safe,
            "safety_violations": self.safety_violations,
        }


class ActionCostModel:
    """Evaluates multi-objective utility and verifies hard safety constraints."""

    def __init__(self, config: Optional[DecisionCostConfig] = None) -> None:
        self.config = config or DecisionCostConfig()

    def check_hard_safety(
        self,
        peak_t_core: float,
        max_pressure: float,
        min_flow: float,
        latent_novelty: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """Strict lexicographical hard constraint gate.
        
        Boundary semantics:
        - T_core >= thermal_hard_limit (95.0°C) -> FAIL
        - P_sys >= pressure_hard_limit (5.50 bar) -> FAIL
        - F_cool <= flow_hard_limit (8.00 L/min) -> FAIL
        - Novelty > latent_novelty_hard_limit (15.0) -> FAIL
        """
        violations = []
        if peak_t_core >= self.config.thermal_hard_limit:
            violations.append(f"T_core peak ({peak_t_core:.2f}°C) >= hard limit ({self.config.thermal_hard_limit:.2f}°C)")
        if max_pressure >= self.config.pressure_hard_limit:
            violations.append(f"P_sys max ({max_pressure:.2f} bar) >= hard limit ({self.config.pressure_hard_limit:.2f} bar)")
        if min_flow <= self.config.flow_hard_limit:
            violations.append(f"F_cool min ({min_flow:.2f} L/min) <= hard minimum ({self.config.flow_hard_limit:.2f} L/min)")
        if latent_novelty > self.config.latent_novelty_hard_limit:
            violations.append(f"Latent novelty ({latent_novelty:.2f}) > support limit ({self.config.latent_novelty_hard_limit:.2f})")

        return len(violations) == 0, violations

    def evaluate_cost(
        self,
        peak_t_core: float,
        max_pressure: float,
        min_flow: float,
        mean_cpu_load: float,
        actions_at_t_star: np.ndarray | List[float],
        baseline_actions_at_t_star: np.ndarray | List[float],
        latent_novelty: float = 0.0,
        is_compound_intervention: bool = False,
    ) -> CostBreakdown:
        """Compute full cost decomposition and net utility score."""
        is_safe, violations = self.check_hard_safety(
            peak_t_core=peak_t_core,
            max_pressure=max_pressure,
            min_flow=min_flow,
            latent_novelty=latent_novelty,
        )

        act = np.asarray(actions_at_t_star)
        base_act = np.asarray(baseline_actions_at_t_star)

        # 1. Performance Benefit: compute capacity delivered
        perf_benefit = self.config.performance_weight * (mean_cpu_load / 100.0)

        # 2. Thermal Risk Penalty (quadratic-linear above soft threshold)
        thermal_excess = max(0.0, peak_t_core - self.config.thermal_soft_limit)
        thermal_pen = self.config.thermal_risk_weight * ((thermal_excess / 10.0) ** 1.5)

        # 3. Pressure Risk Penalty
        pressure_excess = max(0.0, max_pressure - self.config.pressure_soft_limit)
        press_pen = self.config.pressure_risk_weight * (pressure_excess / 1.0)

        # 4. Pump Operating Resource Cost
        pump_stage = float(act[2])
        pump_cost = self.config.pump_cost_weight * (pump_stage / 4.0)

        # 5. Actuation Displacement Cost
        valve_displacement = abs(float(act[0]) - float(base_act[0]))
        act_cost = self.config.valve_actuation_weight * (valve_displacement / 50.0)

        # 6. Throttling Loss Penalty (cost of deliberately curbing compute)
        throttle_loss = max(0.0, (100.0 - mean_cpu_load) / 100.0)
        thr_cost = self.config.throttle_cost_weight * (throttle_loss ** 1.2)

        # 7. Flush Activation Cost
        flush_active = (float(act[3]) > 0.5)
        flush_cost = self.config.flush_cost_weight if flush_active else 0.0

        # 8. Intervention Complexity Penalty
        complexity_pen = self.config.complexity_weight if is_compound_intervention else 0.0

        total_cost = (
            thermal_pen
            + press_pen
            + pump_cost
            + act_cost
            + thr_cost
            + flush_cost
            + complexity_pen
        )

        # Net Utility = Performance Benefit - Total Operational Costs
        # Lexicographical Dominance: Unsafe actions receive prohibitive -1000.0 offset
        raw_utility = perf_benefit - total_cost
        net_utility = raw_utility if is_safe else (raw_utility - 1000.0)

        return CostBreakdown(
            performance_benefit=perf_benefit,
            thermal_penalty=thermal_pen,
            pressure_penalty=press_pen,
            pump_cost=pump_cost,
            actuation_cost=act_cost,
            throttle_cost=thr_cost,
            flush_cost=flush_cost,
            complexity_penalty=complexity_pen,
            total_cost=total_cost,
            net_utility=net_utility,
            is_safe=is_safe,
            safety_violations=violations,
        )
