"""Benchmark Control Policies for Experimental Trajectory Generation.

Provides 5 deterministic experimental policies:
1. NominalController: Balanced PI thermal feedback control
2. HighLoadController: Sustained high workload stress
3. AggressiveCoolingController: Maximum cooling throughput
4. FailureInducingController: Deliberately drives system toward boundary breach
5. RecoveryController: Automated safety response to elevated thermal/pressure alarms
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.observations import ObservationVector


class BasePolicy(ABC):
    """Abstract base class for THC-SCM control policies."""

    def reset(self) -> None:
        """Reset internal policy state if any."""
        pass

    @abstractmethod
    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        """Produce control ActionVector from current observation."""
        pass


class NominalController(BasePolicy):
    """Proportional-Integral feedback controller maintaining T_core ~ 68°C."""

    def __init__(self, target_temp: float = 68.0, kp: float = 2.2, ki: float = 0.05) -> None:
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.integral_err = 0.0

    def reset(self) -> None:
        self.integral_err = 0.0

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        # Use observable noisy T_core (or fallback to nominal if sensor is masked)
        t_meas = obs.T_core if not np.isnan(obs.T_core) else 68.0
        err = t_meas - self.target_temp
        self.integral_err = np.clip(self.integral_err + err * 1.0, -100.0, 100.0)

        # Base valve position 50% + PI correction
        valve_cmd = np.clip(50.0 + self.kp * err + self.ki * self.integral_err, 15.0, 95.0)

        return ActionVector(
            A_valve=float(valve_cmd),
            A_throttle=100.0,
            A_pump=2,
            A_flush=0,
        )


class HighLoadController(BasePolicy):
    """Sustained compute load with fixed moderate cooling."""

    def __init__(self, fixed_valve: float = 45.0, pump_stage: int = 2) -> None:
        self.fixed_valve = fixed_valve
        self.pump_stage = pump_stage

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        return ActionVector(
            A_valve=float(self.fixed_valve),
            A_throttle=100.0,
            A_pump=self.pump_stage,
            A_flush=0,
        )


class AggressiveCoolingController(BasePolicy):
    """Maximum cooling throughput with high pump setting."""

    def __init__(self, pump_stage: int = 3, valve_target: float = 90.0) -> None:
        self.pump_stage = pump_stage
        self.valve_target = valve_target

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        return ActionVector(
            A_valve=float(self.valve_target),
            A_throttle=85.0,
            A_pump=self.pump_stage,
            A_flush=0,
        )


class FailureInducingController(BasePolicy):
    """Throttles cooling while maximizing workload to force thermal runaway."""

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        # Starve cooling valve to 8%, run max load
        return ActionVector(
            A_valve=8.0,
            A_throttle=100.0,
            A_pump=1,
            A_flush=0,
        )


class RecoveryController(BasePolicy):
    """Automated safety supervisor that throttles workload and flushes lines upon alarm."""

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        t_meas = obs.T_core if not np.isnan(obs.T_core) else 70.0
        p_meas = obs.P_sys if not np.isnan(obs.P_sys) else 3.0

        if t_meas >= 92.0:
            # Critical thermal alarm: Emergency flush + throttle workload to 20%
            return ActionVector(
                A_valve=100.0,
                A_throttle=20.0,
                A_pump=3,
                A_flush=1 if step % 10 == 0 else 0,
            )
        elif t_meas >= 80.0:
            # Warning: Moderate throttle + boost valve
            return ActionVector(
                A_valve=85.0,
                A_throttle=50.0,
                A_pump=2,
                A_flush=0,
            )
        elif p_meas >= 5.0:
            # Pressure alarm: Step down pump
            return ActionVector(
                A_valve=70.0,
                A_throttle=60.0,
                A_pump=1,
                A_flush=0,
            )
        else:
            # Nominal
            return ActionVector(
                A_valve=55.0,
                A_throttle=100.0,
                A_pump=2,
                A_flush=0,
            )
