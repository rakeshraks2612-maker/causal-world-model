"""Tests for Safety Boundaries and Failure Latching Semantics."""

from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.failures import FailureEvaluator
from prism.simulator.parameters import SafetyThresholds


def test_thermal_runaway_requires_consecutive_duration() -> None:
    """Verify thermal runaway requires 3 consecutive steps above 105°C."""
    evaluator = FailureEvaluator(SafetyThresholds(t_core_runaway_thresh=105.0, t_core_runaway_duration=3))
    act = ActionVector.default_nominal()

    hot_state = StateVector(
        T_core=108.0,
        T_cool=70.0,
        P_sys=3.0,
        F_cool=20.0,
        L_cpu=90.0,
        V_pos=50.0,
        Vib_pump=5.0,
        P_elec=3.0,
        T_amb=30.0,
        W_wear=0.2,
        Q_internal=0.2,
        xi_leak=0.0,
    )
    safe_state = StateVector.default_nominal()

    # Step 1 hot: not yet failed
    f1 = evaluator.evaluate_step(1, hot_state, act)
    assert not f1.failed
    assert f1.consecutive_thermal_violations == 1

    # Step 2 hot: not yet failed
    f2 = evaluator.evaluate_step(2, hot_state, act)
    assert not f2.failed
    assert f2.consecutive_thermal_violations == 2

    # Step 3 safe: counter resets
    f3 = evaluator.evaluate_step(3, safe_state, act)
    assert not f3.failed
    assert f3.consecutive_thermal_violations == 0

    # Step 4, 5, 6 hot: reaches 3 consecutive violations -> fails on step 6!
    evaluator.evaluate_step(4, hot_state, act)
    evaluator.evaluate_step(5, hot_state, act)
    f6 = evaluator.evaluate_step(6, hot_state, act)
    assert f6.failed
    assert f6.failure_mode == "thermal_runaway"
    assert f6.failure_timestamp == 6


def test_hydraulic_overpressure_and_latching() -> None:
    """Verify hydraulic overpressure triggers immediately and latches permanently."""
    evaluator = FailureEvaluator(SafetyThresholds(p_sys_overpressure_thresh=5.5))
    act = ActionVector.default_nominal()

    overpressure_state = StateVector(
        T_core=68.0,
        T_cool=30.0,
        P_sys=5.8,  # > 5.5 bar!
        F_cool=40.0,
        L_cpu=50.0,
        V_pos=90.0,
        Vib_pump=8.0,
        P_elec=2.0,
        T_amb=25.0,
        W_wear=0.1,
        Q_internal=0.1,
        xi_leak=0.0,
    )
    nominal_state = StateVector.default_nominal()

    f_burst = evaluator.evaluate_step(10, overpressure_state, act)
    assert f_burst.failed
    assert f_burst.failure_mode == "hydraulic_overpressure"
    assert f_burst.failure_timestamp == 10

    # Next steps return to safe pressure, but status remains latched as failed
    f_latched = evaluator.evaluate_step(11, nominal_state, act)
    assert f_latched.failed
    assert f_latched.failure_mode == "hydraulic_overpressure"
    assert f_latched.failure_timestamp == 10


def test_pump_cavitation_failure() -> None:
    """Verify cavitation triggers when valve/pump are high but flow is starved."""
    evaluator = FailureEvaluator()
    act_high_pump = ActionVector(A_valve=80.0, A_throttle=50.0, A_pump=3, A_flush=0)

    cavitation_state = StateVector(
        T_core=70.0,
        T_cool=35.0,
        P_sys=2.0,
        F_cool=1.2,  # < 2.0 L/min cavitation threshold
        L_cpu=50.0,
        V_pos=75.0,  # > 50%
        Vib_pump=6.0,
        P_elec=2.0,
        T_amb=25.0,
        W_wear=0.1,
        Q_internal=0.1,
        xi_leak=0.0,
    )

    f_cav = evaluator.evaluate_step(15, cavitation_state, act_high_pump)
    assert f_cav.failed
    assert f_cav.failure_mode == "pump_cavitation"
    assert f_cav.failure_timestamp == 15
