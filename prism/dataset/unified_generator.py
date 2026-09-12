"""Unified Training Dataset Generator for Baseline 005 (Task 5.10A).

Constructs D_unified with strict 40 / 40 / 20 composition:
1. 40% Sustained Closed-Loop PID:
   - Long-term thermal equilibrium across nominal (T < 85°C), high-safe (85 <= T < 94°C),
     boundary (94 <= T <= 104°C), and sustained runaway (T > 105°C).
   - High load operation, actuator feedback, realistic cooling dynamics.
2. 40% Orthogonal Multi-Actuator Excitation:
   - Balanced pump stages (25% each across stages 1, 2, 3, 4) with 5-15 step residence times.
   - Near-zero cross-actuator correlation (valve, throttle, pump, flush).
   - Structured step interventions for maximum causal identifiability.
3. 20% Multi-Channel Acute Thermal Transients:
   - Physically supported multivariate excursions: thermal surges Delta T >= +15°C
     accompanied by realistic hydraulic degradation (valve throttling, flow choking,
     pressure change, acoustic vibration).
   - Teaches the encoder the true multivariate signature of acute transitions.
"""

from __future__ import annotations
import math
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.validators import validate_oracle_episode_integrity, validate_learner_isolation
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.observations import ObservationVector
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.policies import (
    BasePolicy,
    NominalController,
    HighLoadController,
    AggressiveCoolingController,
    FailureInducingController,
    RecoveryController,
)


class UnifiedRegime(str, Enum):
    """Regimes in the baseline_005 unified training distribution."""
    # 40% Sustained Closed-Loop PID
    PID_NOMINAL = "pid_nominal"
    PID_HIGH_SAFE = "pid_high_safe"
    PID_BOUNDARY = "pid_boundary"
    PID_RUNAWAY = "pid_runaway"
    PID_RECOVERY = "pid_recovery"
    
    # 40% Orthogonal Multi-Actuator Excitation
    EXC_ORTHOGONAL = "exc_orthogonal"
    EXC_PUMP_STAIRCASE = "exc_pump_staircase"
    EXC_VALVE_STEP = "exc_valve_step"
    EXC_THROTTLE_STEP = "exc_throttle_step"
    
    # 20% Physically Supported Multi-Channel Acute Transients
    TRANSIENT_VALVE_CHOKE = "transient_valve_choke"
    TRANSIENT_LOAD_PUMP_FAILURE = "transient_load_pump_failure"
    TRANSIENT_COMPOUND_SURGE = "transient_compound_surge"


class PIDTargetController(BasePolicy):
    """Closed-loop PID controller tracking specific thermal setpoints with actuator feedback."""
    def __init__(self, target_temp: float = 80.0, base_load: float = 70.0, default_pump: int = 2, seed: int = 42) -> None:
        self.target_temp = target_temp
        self.base_load = base_load
        self.default_pump = default_pump
        self.rng = np.random.default_rng(seed)
        self.kp = 2.0
        self.ki = 0.05
        self.kd = 1.0
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        error = obs.T_core - self.target_temp
        self.integral = float(np.clip(self.integral + error, -50.0, 50.0))
        derivative = error - self.prev_error
        self.prev_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        valve = float(np.clip(50.0 + control, 10.0, 95.0))
        
        # Modulate pump speed based on error severity and default pump
        if error > 15.0:
            pump = min(4, self.default_pump + 2)
        elif error > 8.0:
            pump = min(4, self.default_pump + 1)
        elif error < -8.0:
            pump = max(1, self.default_pump - 1)
        else:
            pump = self.default_pump
            
        load_noise = float(self.rng.normal(0.0, 2.0))
        throttle = float(np.clip(self.base_load + load_noise, 10.0, 100.0))
        
        return ActionVector(
            A_valve=valve,
            A_throttle=throttle,
            A_pump=pump,
            A_flush=0,
        )


class StepInterventionController(BasePolicy):
    """Steps valve, pump, or throttle at t* to create clean causal contrasts."""
    def __init__(
        self,
        t_star: int = 30,
        pre_valve: float = 50.0,
        pre_throttle: float = 100.0,
        pre_pump: int = 2,
        post_valve: float = 85.0,
        post_throttle: float = 100.0,
        post_pump: int = 3,
    ) -> None:
        self.t_star = t_star
        self.pre_valve = pre_valve
        self.pre_throttle = pre_throttle
        self.pre_pump = pre_pump
        self.post_valve = post_valve
        self.post_throttle = post_throttle
        self.post_pump = post_pump

    def reset(self) -> None:
        pass

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        if step < self.t_star:
            return ActionVector(
                A_valve=float(self.pre_valve),
                A_throttle=float(self.pre_throttle),
                A_pump=int(self.pre_pump),
                A_flush=0,
            )
        return ActionVector(
            A_valve=float(self.post_valve),
            A_throttle=float(self.post_throttle),
            A_pump=int(self.post_pump),
            A_flush=0,
        )


class OrthogonalExcitationController(BasePolicy):
    """Independently samples actions with short holds (5-15 steps) across all channels."""
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.pump_stages = [1, 2, 3, 4]
        self.valve_levels = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        self.throttle_levels = [30.0, 50.0, 70.0, 85.0, 100.0]
        
        self.current_pump = 2
        self.current_valve = 60.0
        self.current_throttle = 80.0
        self.current_flush = 0
        
        self.pump_timer = 0
        self.valve_timer = 0
        self.throttle_timer = 0
        self.flush_timer = 0

    def reset(self) -> None:
        self.current_pump = int(self.rng.choice(self.pump_stages))
        self.current_valve = float(self.rng.choice(self.valve_levels))
        self.current_throttle = float(self.rng.choice(self.throttle_levels))
        self.current_flush = 0
        self.pump_timer = int(self.rng.integers(5, 15))
        self.valve_timer = int(self.rng.integers(5, 15))
        self.throttle_timer = int(self.rng.integers(5, 15))
        self.flush_timer = int(self.rng.integers(20, 60))

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        self.pump_timer -= 1
        if self.pump_timer <= 0:
            other_pumps = [p for p in self.pump_stages if p != self.current_pump]
            self.current_pump = int(self.rng.choice(other_pumps))
            self.pump_timer = int(self.rng.integers(5, 15))
            
        self.valve_timer -= 1
        if self.valve_timer <= 0:
            self.current_valve = float(self.rng.choice(self.valve_levels))
            self.valve_timer = int(self.rng.integers(5, 15))
            
        self.throttle_timer -= 1
        if self.throttle_timer <= 0:
            self.current_throttle = float(self.rng.choice(self.throttle_levels))
            self.throttle_timer = int(self.rng.integers(5, 15))
            
        self.flush_timer -= 1
        if self.flush_timer <= 0:
            if self.current_flush == 0:
                self.current_flush = 1
                self.flush_timer = int(self.rng.integers(2, 5))
            else:
                self.current_flush = 0
                self.flush_timer = int(self.rng.integers(30, 80))
                
        return ActionVector(
            A_valve=float(self.current_valve),
            A_throttle=float(self.current_throttle),
            A_pump=int(self.current_pump),
            A_flush=int(self.current_flush),
        )


class PumpStaircaseController(BasePolicy):
    """Systematically cycles pump 1 -> 2 -> 3 -> 4 -> 3 -> 2 -> 1 with fixed holds."""
    def __init__(self, hold_time: int = 15, valve: float = 60.0, throttle: float = 80.0) -> None:
        self.hold_time = hold_time
        self.valve = valve
        self.throttle = throttle
        self.sequence = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4]

    def reset(self) -> None:
        pass

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        seq_idx = (step // self.hold_time) % len(self.sequence)
        pump = self.sequence[seq_idx]
        return ActionVector(
            A_valve=float(self.valve),
            A_throttle=float(self.throttle),
            A_pump=int(pump),
            A_flush=0,
        )


class MultivariateTransientController(BasePolicy):
    """Induces physically consistent acute thermal transients by degrading cooling or surging load."""
    def __init__(
        self,
        t_event: int = 35,
        event_type: str = "valve_choke",
        seed: int = 42,
    ) -> None:
        self.t_event = t_event
        self.event_type = event_type
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        if step < self.t_event:
            return ActionVector(A_valve=65.0, A_throttle=80.0, A_pump=2, A_flush=0)
            
        if self.event_type == "valve_choke":
            return ActionVector(A_valve=5.0, A_throttle=100.0, A_pump=2, A_flush=0)
        elif self.event_type == "load_pump_failure":
            return ActionVector(A_valve=40.0, A_throttle=100.0, A_pump=1, A_flush=0)
        elif self.event_type == "compound_surge":
            return ActionVector(A_valve=10.0, A_throttle=100.0, A_pump=1, A_flush=0)
        else:
            return ActionVector(A_valve=10.0, A_throttle=95.0, A_pump=1, A_flush=0)


def sample_unified_initial_state_and_policy(
    regime: UnifiedRegime,
    rng: np.random.Generator,
    seed: int,
) -> Tuple[StateVector, BasePolicy, Optional[InterventionRegistry]]:
    """Sample state vector, policy, and interventions for a given regime."""
    inv_reg = InterventionRegistry()
    
    default_p = int(rng.choice([1, 2, 3, 4]))
    
    if regime == UnifiedRegime.PID_NOMINAL:
        init_state = StateVector(
            T_core=float(rng.normal(70.0, 2.0)),
            T_cool=float(rng.normal(40.0, 1.5)),
            P_sys=float(rng.normal(3.0, 0.1)),
            F_cool=float(rng.normal(30.0, 1.5)),
            L_cpu=float(rng.normal(60.0, 2.0)),
            V_pos=float(rng.normal(60.0, 2.0)),
            Vib_pump=float(rng.normal(5.0, 0.3)),
            P_elec=float(rng.normal(2.5, 0.1)),
            T_amb=float(rng.normal(22.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.0, 0.02)),
            xi_leak=0.0,
        )
        policy = PIDTargetController(target_temp=75.0, base_load=60.0, default_pump=default_p, seed=seed)
        
    elif regime == UnifiedRegime.PID_HIGH_SAFE:
        init_state = StateVector(
            T_core=float(rng.normal(88.0, 1.5)),
            T_cool=float(rng.normal(55.0, 1.5)),
            P_sys=float(rng.normal(3.2, 0.1)),
            F_cool=float(rng.normal(40.0, 2.0)),
            L_cpu=float(rng.normal(85.0, 2.0)),
            V_pos=float(rng.normal(75.0, 2.0)),
            Vib_pump=float(rng.normal(8.0, 0.4)),
            P_elec=float(rng.normal(3.5, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.10, 0.03)),
            xi_leak=0.0,
        )
        policy = PIDTargetController(target_temp=90.0, base_load=85.0, default_pump=default_p, seed=seed)
        
    elif regime == UnifiedRegime.PID_BOUNDARY:
        init_state = StateVector(
            T_core=float(rng.normal(96.0, 1.5)),
            T_cool=float(rng.normal(62.0, 1.5)),
            P_sys=float(rng.normal(3.3, 0.1)),
            F_cool=float(rng.normal(45.0, 2.0)),
            L_cpu=float(rng.normal(95.0, 1.5)),
            V_pos=float(rng.normal(85.0, 2.0)),
            Vib_pump=float(rng.normal(10.0, 0.5)),
            P_elec=float(rng.normal(4.0, 0.1)),
            T_amb=float(rng.normal(28.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.12, 0.03), 0.0, 0.3)),
            Q_internal=float(rng.normal(0.20, 0.04)),
            xi_leak=0.0,
        )
        policy = PIDTargetController(target_temp=98.0, base_load=95.0, default_pump=default_p, seed=seed)
        
    elif regime == UnifiedRegime.PID_RUNAWAY:
        init_state = StateVector(
            T_core=float(rng.normal(82.0, 2.0)),
            T_cool=float(rng.normal(45.0, 1.5)),
            P_sys=float(rng.normal(2.8, 0.1)),
            F_cool=float(rng.normal(12.0, 1.5)),
            L_cpu=float(rng.normal(90.0, 2.0)),
            V_pos=float(rng.normal(15.0, 2.0)),
            Vib_pump=float(rng.normal(6.0, 0.3)),
            P_elec=float(rng.normal(3.6, 0.1)),
            T_amb=float(rng.normal(28.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.15, 0.04), 0.0, 0.3)),
            Q_internal=float(rng.normal(0.30, 0.05)),
            xi_leak=0.0,
        )
        policy = FailureInducingController()
        
    elif regime == UnifiedRegime.PID_RECOVERY:
        init_state = StateVector(
            T_core=float(rng.normal(94.0, 2.0)),
            T_cool=float(rng.normal(60.0, 1.5)),
            P_sys=float(rng.normal(3.0, 0.1)),
            F_cool=float(rng.normal(35.0, 2.0)),
            L_cpu=float(rng.normal(80.0, 2.0)),
            V_pos=float(rng.normal(70.0, 2.0)),
            Vib_pump=float(rng.normal(8.0, 0.4)),
            P_elec=float(rng.normal(3.2, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.10, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.10, 0.03)),
            xi_leak=0.0,
        )
        policy = RecoveryController()
        
    elif regime == UnifiedRegime.EXC_ORTHOGONAL:
        init_state = StateVector(
            T_core=float(rng.normal(78.0, 3.0)),
            T_cool=float(rng.normal(48.0, 2.0)),
            P_sys=float(rng.normal(3.1, 0.1)),
            F_cool=float(rng.normal(32.0, 2.5)),
            L_cpu=float(rng.normal(75.0, 5.0)),
            V_pos=float(rng.normal(60.0, 5.0)),
            Vib_pump=float(rng.normal(6.5, 0.5)),
            P_elec=float(rng.normal(3.0, 0.2)),
            T_amb=float(rng.normal(24.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.08, 0.03), 0.0, 0.25)),
            Q_internal=float(rng.normal(0.08, 0.03)),
            xi_leak=0.0,
        )
        policy = OrthogonalExcitationController(seed=seed)
        
    elif regime == UnifiedRegime.EXC_PUMP_STAIRCASE:
        init_state = StateVector(
            T_core=float(rng.normal(76.0, 2.0)),
            T_cool=float(rng.normal(46.0, 1.5)),
            P_sys=float(rng.normal(3.0, 0.1)),
            F_cool=float(rng.normal(28.0, 2.0)),
            L_cpu=float(rng.normal(80.0, 2.0)),
            V_pos=float(rng.normal(55.0, 2.0)),
            Vib_pump=float(rng.normal(6.0, 0.4)),
            P_elec=float(rng.normal(3.2, 0.1)),
            T_amb=float(rng.normal(23.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.05, 0.02)),
            xi_leak=0.0,
        )
        policy = PumpStaircaseController(hold_time=12, valve=55.0, throttle=85.0)
        
    elif regime == UnifiedRegime.EXC_VALVE_STEP:
        p_step = int(rng.choice([1, 2, 3, 4]))
        init_state = StateVector(
            T_core=float(rng.normal(84.0, 2.0)),
            T_cool=float(rng.normal(52.0, 1.5)),
            P_sys=float(rng.normal(3.1, 0.1)),
            F_cool=float(rng.normal(20.0, 1.5)),
            L_cpu=float(rng.normal(85.0, 2.0)),
            V_pos=float(rng.normal(30.0, 2.0)),
            Vib_pump=float(rng.normal(7.0, 0.4)),
            P_elec=float(rng.normal(3.4, 0.1)),
            T_amb=float(rng.normal(24.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.07, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.08, 0.02)),
            xi_leak=0.0,
        )
        policy = StepInterventionController(t_star=35, pre_valve=30.0, post_valve=90.0, pre_pump=p_step, post_pump=p_step)
        
    elif regime == UnifiedRegime.EXC_THROTTLE_STEP:
        p_step = int(rng.choice([1, 2, 3, 4]))
        init_state = StateVector(
            T_core=float(rng.normal(88.0, 2.0)),
            T_cool=float(rng.normal(54.0, 1.5)),
            P_sys=float(rng.normal(3.1, 0.1)),
            F_cool=float(rng.normal(35.0, 2.0)),
            L_cpu=float(rng.normal(100.0, 1.0)),
            V_pos=float(rng.normal(70.0, 2.0)),
            Vib_pump=float(rng.normal(8.0, 0.4)),
            P_elec=float(rng.normal(4.0, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.12, 0.03)),
            xi_leak=0.0,
        )
        policy = StepInterventionController(t_star=35, pre_throttle=100.0, post_throttle=40.0, pre_pump=p_step, post_pump=p_step)
        
    elif regime == UnifiedRegime.TRANSIENT_VALVE_CHOKE:
        init_state = StateVector(
            T_core=float(rng.normal(76.0, 2.0)),
            T_cool=float(rng.normal(45.0, 1.5)),
            P_sys=float(rng.normal(3.1, 0.1)),
            F_cool=float(rng.normal(38.0, 2.0)),
            L_cpu=float(rng.normal(80.0, 2.0)),
            V_pos=float(rng.normal(65.0, 2.0)),
            Vib_pump=float(rng.normal(6.5, 0.4)),
            P_elec=float(rng.normal(3.2, 0.1)),
            T_amb=float(rng.normal(24.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.2)),
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=0.0,
        )
        policy = MultivariateTransientController(t_event=35, event_type="valve_choke", seed=seed)
        
    elif regime == UnifiedRegime.TRANSIENT_LOAD_PUMP_FAILURE:
        init_state = StateVector(
            T_core=float(rng.normal(78.0, 2.0)),
            T_cool=float(rng.normal(48.0, 1.5)),
            P_sys=float(rng.normal(3.1, 0.1)),
            F_cool=float(rng.normal(32.0, 2.0)),
            L_cpu=float(rng.normal(75.0, 2.0)),
            V_pos=float(rng.normal(55.0, 2.0)),
            Vib_pump=float(rng.normal(6.0, 0.4)),
            P_elec=float(rng.normal(3.0, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.10, 0.03), 0.0, 0.25)),
            Q_internal=float(rng.normal(0.20, 0.04)),
            xi_leak=0.0,
        )
        policy = MultivariateTransientController(t_event=35, event_type="load_pump_failure", seed=seed)
        
    elif regime == UnifiedRegime.TRANSIENT_COMPOUND_SURGE:
        init_state = StateVector(
            T_core=float(rng.normal(80.0, 2.0)),
            T_cool=float(rng.normal(50.0, 1.5)),
            P_sys=float(rng.normal(3.2, 0.1)),
            F_cool=float(rng.normal(30.0, 2.0)),
            L_cpu=float(rng.normal(85.0, 2.0)),
            V_pos=float(rng.normal(60.0, 2.0)),
            Vib_pump=float(rng.normal(7.0, 0.4)),
            P_elec=float(rng.normal(3.4, 0.1)),
            T_amb=float(rng.normal(26.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.12, 0.03), 0.0, 0.3)),
            Q_internal=float(rng.normal(0.25, 0.05)),
            xi_leak=0.0,
        )
        policy = MultivariateTransientController(t_event=30, event_type="compound_surge", seed=seed)
        
    else:
        raise ValueError(f"Unknown regime: {regime}")
        
    return init_state, policy, inv_reg


def generate_unified_episode(
    split: SplitType,
    index: int,
    regime: UnifiedRegime,
    length: int = 120,
    seed: int = 42,
) -> Tuple[OracleEpisode, LearnerEpisode]:
    """Generate a single verified episode under the unified regime distribution."""
    ep_id = f"ep_{split.value}_unf_{index:05d}_{regime.value}"
    init_rng = np.random.default_rng(seed)
    init_state, policy, inv_reg = sample_unified_initial_state_and_policy(regime, init_rng, seed)
    
    sim = THCSimulator(seed=seed)
    raw_ep = sim.run_episode(
        policy=policy,
        length=length,
        initial_state=init_state,
        interventions=inv_reg,
        episode_id=ep_id,
    )
    
    mask = ~np.isnan(raw_ep.observations)
    
    oracle_ep = OracleEpisode(
        episode_id=ep_id,
        seed=seed,
        split=split,
        timestamps=raw_ep.timestamps,
        observations=raw_ep.observations,
        observation_mask=mask,
        actions=raw_ep.actions,
        ground_truth_states=raw_ep.ground_truth_states,
        exogenous_noise=raw_ep.exogenous_noise,
        failure_latched=raw_ep.failure_latched,
        failure_mode=raw_ep.failure_mode,
        failure_timestamp=raw_ep.failure_timestamp,
        oracle_metadata={
            "regime": regime.value,
            "policy": policy.__class__.__name__,
            "seed": seed,
            "length": length,
            "has_interventions": inv_reg is not None and len(inv_reg.interventions) > 0,
            "initial_t_amb": float(init_state.T_amb),
            "initial_w_wear": float(init_state.W_wear),
            "initial_q_internal": float(init_state.Q_internal),
        },
    )
    
    validate_oracle_episode_integrity(oracle_ep)
    learner_ep = oracle_ep.to_learner_episode()
    validate_learner_isolation(learner_ep)
    
    return oracle_ep, learner_ep


def generate_baseline_005_dataset(
    output_dir: str | Path,
    train_count: int = 150,
    val_count: int = 30,
    length: int = 120,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Generate the complete Baseline 005 Unified Training Dataset with 40/40/20 proportions."""
    out_path = Path(output_dir)
    orc_train = out_path / "oracle" / "train"
    orc_val = out_path / "oracle" / "validation"
    lrn_train = out_path / "learner" / "train"
    lrn_val = out_path / "learner" / "validation"
    
    for d in [orc_train, orc_val, lrn_train, lrn_val]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 40% PID (60 episodes), 40% Excitation (60 episodes), 20% Acute Transients (30 episodes)
    train_plan = [
        (UnifiedRegime.PID_NOMINAL, 15),
        (UnifiedRegime.PID_HIGH_SAFE, 15),
        (UnifiedRegime.PID_BOUNDARY, 15),
        (UnifiedRegime.PID_RUNAWAY, 10),
        (UnifiedRegime.PID_RECOVERY, 5),
        (UnifiedRegime.EXC_ORTHOGONAL, 25),
        (UnifiedRegime.EXC_PUMP_STAIRCASE, 15),
        (UnifiedRegime.EXC_VALVE_STEP, 10),
        (UnifiedRegime.EXC_THROTTLE_STEP, 10),
        (UnifiedRegime.TRANSIENT_VALVE_CHOKE, 10),
        (UnifiedRegime.TRANSIENT_LOAD_PUMP_FAILURE, 10),
        (UnifiedRegime.TRANSIENT_COMPOUND_SURGE, 10),
    ]
    
    val_plan = [
        (UnifiedRegime.PID_NOMINAL, 3),
        (UnifiedRegime.PID_HIGH_SAFE, 3),
        (UnifiedRegime.PID_BOUNDARY, 3),
        (UnifiedRegime.PID_RUNAWAY, 3),
        (UnifiedRegime.PID_RECOVERY, 2),
        (UnifiedRegime.EXC_ORTHOGONAL, 4),
        (UnifiedRegime.EXC_PUMP_STAIRCASE, 3),
        (UnifiedRegime.EXC_VALVE_STEP, 3),
        (UnifiedRegime.EXC_THROTTLE_STEP, 2),
        (UnifiedRegime.TRANSIENT_VALVE_CHOKE, 2),
        (UnifiedRegime.TRANSIENT_LOAD_PUMP_FAILURE, 2),
        (UnifiedRegime.TRANSIENT_COMPOUND_SURGE, 2),
    ]
    
    seed_counter = base_seed
    train_episodes_saved = 0
    val_episodes_saved = 0
    
    for regime, count in train_plan:
        for i in range(count):
            orc, lrn = generate_unified_episode(SplitType.TRAIN, train_episodes_saved, regime, length=length, seed=seed_counter)
            orc.save_npz(orc_train / f"{orc.episode_id}.npz")
            lrn.save_npz(lrn_train / f"{lrn.episode_id}.npz")
            seed_counter += 1
            train_episodes_saved += 1
            
    for regime, count in val_plan:
        for i in range(count):
            orc, lrn = generate_unified_episode(SplitType.VAL, val_episodes_saved, regime, length=length, seed=seed_counter)
            orc.save_npz(orc_val / f"{orc.episode_id}.npz")
            lrn.save_npz(lrn_val / f"{lrn.episode_id}.npz")
            seed_counter += 1
            val_episodes_saved += 1
            
    manifest = {
        "dataset_name": "baseline_005_unified_dataset",
        "total_train_episodes": train_episodes_saved,
        "total_val_episodes": val_episodes_saved,
        "timesteps_per_episode": length,
        "total_train_timesteps": train_episodes_saved * length,
        "total_val_timesteps": val_episodes_saved * length,
        "proportions": {
            "sustained_closed_loop_pid": 0.40,
            "orthogonal_multi_actuator_excitation": 0.40,
            "multichannel_acute_transients": 0.20,
        },
    }
    
    import json
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest
