"""Tests for Structural Causal Dynamics and Physical Invariants."""

import pytest
import numpy as np
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.noise import NoiseVector
from prism.simulator.parameters import SystemConstants
from prism.simulator.dynamics import StructuralDynamics


@pytest.fixture
def dynamics() -> StructuralDynamics:
    return StructuralDynamics(SystemConstants.load_default())


def test_pump_action_increases_head_and_flow(dynamics: StructuralDynamics) -> None:
    """Verify A_pump increases hydrostatic pressure and flow rate under identical valve."""
    state = StateVector.default_nominal()
    noise = NoiseVector.zeros()

    act_stage1 = ActionVector(A_valve=50.0, A_throttle=50.0, A_pump=1, A_flush=0)
    act_stage4 = ActionVector(A_valve=50.0, A_throttle=50.0, A_pump=4, A_flush=0)

    next_s1 = dynamics.step(0, state, act_stage1, noise)
    next_s4 = dynamics.step(0, state, act_stage4, noise)

    assert next_s4.P_sys > next_s1.P_sys
    assert next_s4.F_cool > next_s1.F_cool


def test_flush_action_effects(dynamics: StructuralDynamics) -> None:
    """Verify A_flush=1 surges flow, drops coolant temp, and relieves wear."""
    state = StateVector(
        T_core=90.0,
        T_cool=60.0,
        P_sys=3.5,
        F_cool=20.0,
        L_cpu=80.0,
        V_pos=50.0,
        Vib_pump=5.0,
        P_elec=2.5,
        T_amb=30.0,
        W_wear=0.5,
        Q_internal=0.2,
        xi_leak=5.0,
    )
    noise = NoiseVector.zeros()

    act_normal = ActionVector(A_valve=50.0, A_throttle=80.0, A_pump=2, A_flush=0)
    act_flush = ActionVector(A_valve=50.0, A_throttle=80.0, A_pump=2, A_flush=1)

    next_normal = dynamics.step(0, state, act_normal, noise)
    next_flush = dynamics.step(0, state, act_flush, noise)

    # Flush surges flow
    assert next_flush.F_cool > next_normal.F_cool
    # Flush drops coolant temperature
    assert next_flush.T_cool < next_normal.T_cool
    # Flush reduces cumulative wear
    assert next_flush.W_wear < next_normal.W_wear


def test_leak_degrades_pressure_and_flow(dynamics: StructuralDynamics) -> None:
    """Verify latent micro-leak rate xi_leak reduces system pressure and flow."""
    state_healthy = StateVector.default_nominal()
    state_leaking = StateVector(
        T_core=68.5,
        T_cool=34.0,
        P_sys=3.15,
        F_cool=32.0,
        L_cpu=50.0,
        V_pos=55.0,
        Vib_pump=4.2,
        P_elec=1.85,
        T_amb=25.0,
        W_wear=0.05,
        Q_internal=0.15,
        xi_leak=35.0,  # High leak
    )
    noise = NoiseVector.zeros()
    act = ActionVector.default_nominal()

    next_healthy = dynamics.step(0, state_healthy, act, noise)
    next_leaking = dynamics.step(0, state_leaking, act, noise)

    assert next_leaking.F_cool < next_healthy.F_cool
    assert next_leaking.P_sys < next_healthy.P_sys


def test_wear_slows_actuator_response(dynamics: StructuralDynamics) -> None:
    """Verify mechanical wear W_wear increases valve actuator response lag."""
    state_new = StateVector.default_nominal()  # W_wear = 0.05
    state_worn = StateVector(
        T_core=68.5,
        T_cool=34.0,
        P_sys=3.15,
        F_cool=32.0,
        L_cpu=50.0,
        V_pos=20.0,  # Starting from 20%
        Vib_pump=4.2,
        P_elec=1.85,
        T_amb=25.0,
        W_wear=0.90,  # Severe wear
        Q_internal=0.15,
        xi_leak=0.0,
    )
    state_fresh = StateVector(
        T_core=68.5,
        T_cool=34.0,
        P_sys=3.15,
        F_cool=32.0,
        L_cpu=50.0,
        V_pos=20.0,  # Starting from 20%
        Vib_pump=4.2,
        P_elec=1.85,
        T_amb=25.0,
        W_wear=0.0,  # Zero wear
        Q_internal=0.15,
        xi_leak=0.0,
    )
    noise = NoiseVector.zeros()
    # Step target valve opening to 100%
    act = ActionVector(A_valve=100.0, A_throttle=50.0, A_pump=2, A_flush=0)

    next_fresh = dynamics.step(0, state_fresh, act, noise)
    next_worn = dynamics.step(0, state_worn, act, noise)

    # Fresh valve opens faster in one second than worn valve
    assert next_fresh.V_pos > next_worn.V_pos
