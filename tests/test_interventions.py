"""Tests for Graph-Surgical Interventions and Causal Sanity Checks."""

import pytest
import numpy as np
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.noise import NoiseVector
from prism.simulator.parameters import SystemConstants
from prism.simulator.dynamics import StructuralDynamics
from prism.simulator.interventions import InterventionRegistry, Intervention


@pytest.fixture
def dynamics() -> StructuralDynamics:
    return StructuralDynamics(SystemConstants.load_default())


def test_action_vs_intervention_distinction(dynamics: StructuralDynamics) -> None:
    """CRITICAL TEST: Action A_valve=85% != Intervention do(V_pos=85%).
    
    A_valve=85% must go through actuator response lag tau_valve(W_wear).
    do(V_pos=85%) instantly clamps the physical valve to 85% by graph surgery.
    """
    # Start with initial valve position at 20% and high wear
    state = StateVector(
        T_core=70.0,
        T_cool=35.0,
        P_sys=3.0,
        F_cool=15.0,
        L_cpu=50.0,
        V_pos=20.0,
        Vib_pump=4.0,
        P_elec=1.8,
        T_amb=25.0,
        W_wear=0.8,  # High wear
        Q_internal=0.1,
        xi_leak=0.0,
    )
    noise = NoiseVector.zeros()

    # 1. Natural Action A_valve = 85%
    act_natural = ActionVector(A_valve=85.0, A_throttle=50.0, A_pump=2, A_flush=0)
    next_action_only = dynamics.step(0, state, act_natural, noise, interventions=None)

    # 2. Intervention do(V_pos = 85%) with natural action kept at 20%
    act_intervened = ActionVector(A_valve=20.0, A_throttle=50.0, A_pump=2, A_flush=0)
    inv_reg = InterventionRegistry.create_single("V_pos", 85.0, step=0)
    next_intervention = dynamics.step(0, state, act_intervened, noise, interventions=inv_reg)

    # The forced intervention immediately sets V_pos to exactly 85%
    assert next_intervention.V_pos == 85.0

    # The natural action only moved partially from 20% toward 85% due to wear lag
    assert next_action_only.V_pos < 50.0
    assert next_action_only.V_pos != next_intervention.V_pos

    # Downstream flow must also differ significantly
    assert next_intervention.F_cool > next_action_only.F_cool


def test_anti_spurious_causal_sanity(dynamics: StructuralDynamics) -> None:
    """CAUSAL SANITY TEST: do(Vib_pump = 0.5) has ZERO effect on thermal dynamics.
    
    Although vibration correlates observationally with pump strain and failure,
    forcibly clamping vibration must not alter core temperature or coolant dynamics.
    """
    state = StateVector.default_nominal()
    noise = NoiseVector.zeros()
    act = ActionVector.default_nominal()

    # Natural baseline step
    next_baseline = dynamics.step(0, state, act, noise, interventions=None)

    # Intervene only on vibration
    inv_reg = InterventionRegistry.create_single("Vib_pump", 0.5, step=0)
    next_intervened = dynamics.step(0, state, act, noise, interventions=inv_reg)

    # Vibration is forced to 0.5
    assert next_intervened.Vib_pump == 0.5
    assert next_baseline.Vib_pump != 0.5

    # Core temperature, flow, pressure, and coolant temperature remain EXACTLY identical!
    assert next_intervened.T_core == next_baseline.T_core
    assert next_intervened.T_cool == next_baseline.T_cool
    assert next_intervened.F_cool == next_baseline.F_cool
    assert next_intervened.P_sys == next_baseline.P_sys
    assert next_intervened.P_elec == next_baseline.P_elec


def test_multi_value_anti_spurious_intervention_range(dynamics: StructuralDynamics) -> None:
    """TASK 1.21: Multi-value intervention range test on Vib_pump.
    
    Verifies that for arbitrary intervention values do(Vib_pump in {0.0, 0.5, 2.0, 5.0, 15.0}),
    all non-descendant physical variables remain strictly bit-for-bit identical to baseline.
    """
    state = StateVector.default_nominal()
    noise = NoiseVector.zeros()
    act = ActionVector.default_nominal()

    next_baseline = dynamics.step(0, state, act, noise, interventions=None)

    test_values = [0.0, 0.5, 2.0, 5.0, 15.0, 28.5]
    for val in test_values:
        inv_reg = InterventionRegistry.create_single("Vib_pump", val, step=0)
        next_intervened = dynamics.step(0, state, act, noise, interventions=inv_reg)

        # Vibration is clamped exactly to target value
        assert next_intervened.Vib_pump == val

        # Every single non-descendant state variable is 100% bit-for-bit invariant
        assert next_intervened.T_core == next_baseline.T_core
        assert next_intervened.T_cool == next_baseline.T_cool
        assert next_intervened.P_sys == next_baseline.P_sys
        assert next_intervened.F_cool == next_baseline.F_cool
        assert next_intervened.L_cpu == next_baseline.L_cpu
        assert next_intervened.V_pos == next_baseline.V_pos
        assert next_intervened.P_elec == next_baseline.P_elec
        assert next_intervened.T_amb == next_baseline.T_amb
        assert next_intervened.W_wear == next_baseline.W_wear
        assert next_intervened.Q_internal == next_baseline.Q_internal
        assert next_intervened.xi_leak == next_baseline.xi_leak
