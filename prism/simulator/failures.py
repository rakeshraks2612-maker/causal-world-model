"""Failure Evaluator and Safety Boundary Latching for THC-SCM System.

Evaluates 3 physical failure boundaries:
1. Thermal Runaway: T_core >= 105°C for >= 3 consecutive seconds
2. Hydraulic Overpressure: P_sys >= 5.5 bar
3. Pump Cavitation: F_cool < 2.0 L/min while V_pos > 50% and A_pump >= 2

Maintains permanent latching: once failed at step t_fail, state remains failed for the rest of the episode.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.parameters import SafetyThresholds


@dataclass
class FailureState:
    """Diagnostic failure status at a given step."""

    failed: bool = False
    failure_mode: Optional[str] = None
    failure_timestamp: Optional[int] = None
    consecutive_thermal_violations: int = 0
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureEvaluator:
    """Evaluates physical failure boundaries and latches failure state across time."""

    def __init__(self, thresholds: Optional[SafetyThresholds] = None) -> None:
        self.thresholds = thresholds or SafetyThresholds()
        self.state = FailureState()

    def reset(self) -> None:
        """Reset failure evaluator state for a new episode."""
        self.state = FailureState()

    def evaluate_step(self, step: int, state: StateVector, action: ActionVector) -> FailureState:
        """Evaluate physical safety boundaries at time step and update latched state.
        
        Args:
            step: Current discrete time step
            state: Physical 12-dimensional StateVector
            action: Current ActionVector
            
        Returns:
            Updated FailureState
        """
        # Calculate continuous composite risk score [0, 1]
        thermal_risk = max(0.0, min(1.0, (state.T_core - 70.0) / (self.thresholds.t_core_runaway_thresh - 70.0)))
        pressure_risk = max(0.0, min(1.0, (state.P_sys - 2.5) / (self.thresholds.p_sys_overpressure_thresh - 2.5)))
        flow_risk = max(0.0, min(1.0, (10.0 - state.F_cool) / 10.0)) if (state.V_pos > 40.0 and action.A_pump >= 2) else 0.0
        composite_risk = float(max(thermal_risk, pressure_risk, flow_risk))

        # Check thermal runaway violation counter
        if state.T_core >= self.thresholds.t_core_runaway_thresh:
            self.state.consecutive_thermal_violations += 1
        else:
            self.state.consecutive_thermal_violations = 0

        # If already failed, maintain latched failure status
        if self.state.failed:
            self.state.risk_score = 1.0
            return self.state

        # Check 1: Thermal Runaway
        if self.state.consecutive_thermal_violations >= self.thresholds.t_core_runaway_duration:
            self.state.failed = True
            self.state.failure_mode = "thermal_runaway"
            self.state.failure_timestamp = step
            self.state.risk_score = 1.0
            return self.state

        # Check 2: Hydraulic Overpressure Burst
        if state.P_sys >= self.thresholds.p_sys_overpressure_thresh:
            self.state.failed = True
            self.state.failure_mode = "hydraulic_overpressure"
            self.state.failure_timestamp = step
            self.state.risk_score = 1.0
            return self.state

        # Check 3: Pump Cavitation / Dry Run
        if (
            state.F_cool < self.thresholds.f_cool_cavitation_thresh
            and state.V_pos > self.thresholds.v_pos_cavitation_thresh
            and action.A_pump >= self.thresholds.a_pump_cavitation_thresh
        ):
            self.state.failed = True
            self.state.failure_mode = "pump_cavitation"
            self.state.failure_timestamp = step
            self.state.risk_score = 1.0
            return self.state

        self.state.risk_score = composite_risk
        return self.state
