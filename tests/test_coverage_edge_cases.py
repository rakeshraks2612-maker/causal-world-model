"""Edge cases and serialization roundtrip tests to ensure 100% test completeness."""

import tempfile
from pathlib import Path
import numpy as np
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.observations import ObservationVector
from prism.simulator.noise import NoiseVector
from prism.simulator.episode import Episode
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.policies import (
    NominalController,
    AggressiveCoolingController,
    RecoveryController,
)


def test_episode_npz_serialization_roundtrip() -> None:
    """Verify save_npz and load_npz perfectly reconstruct Episode."""
    sim = THCSimulator(seed=42)
    ep = sim.run_episode(NominalController(), length=30, episode_id="test_ep_01")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_episode.npz"
        ep.save_npz(file_path)

        loaded_ep = Episode.load_npz(file_path)
        assert loaded_ep.episode_id == ep.episode_id
        assert loaded_ep.seed == ep.seed
        assert loaded_ep.length == ep.length
        assert loaded_ep.failure_latched == ep.failure_latched
        np.testing.assert_array_equal(loaded_ep.ground_truth_states, ep.ground_truth_states)
        np.testing.assert_array_equal(loaded_ep.observations, ep.observations)
        np.testing.assert_array_equal(loaded_ep.actions, ep.actions)
        np.testing.assert_array_equal(loaded_ep.exogenous_noise, ep.exogenous_noise)


def test_recovery_controller_triggers_alarm_responses() -> None:
    """Verify RecoveryController responds appropriately to hot, warm, and high-pressure states."""
    ctrl = RecoveryController()

    # Normal observation
    obs_normal = ObservationVector.from_state(StateVector.default_nominal())
    act_normal = ctrl.select_action(0, obs_normal)
    assert act_normal.A_throttle == 100.0

    # Warm observation (T_core = 84°C)
    state_warm = StateVector(
        T_core=84.0,
        T_cool=40.0,
        P_sys=3.0,
        F_cool=25.0,
        L_cpu=70.0,
        V_pos=60.0,
        Vib_pump=4.0,
        P_elec=2.0,
        T_amb=25.0,
        W_wear=0.1,
        Q_internal=0.1,
        xi_leak=0.0,
    )
    obs_warm = ObservationVector.from_state(state_warm)
    act_warm = ctrl.select_action(1, obs_warm)
    assert act_warm.A_throttle == 50.0  # Throttled

    # Critical thermal alarm (T_core = 95°C)
    state_crit = StateVector(
        T_core=95.0,
        T_cool=55.0,
        P_sys=3.0,
        F_cool=20.0,
        L_cpu=90.0,
        V_pos=60.0,
        Vib_pump=5.0,
        P_elec=3.0,
        T_amb=25.0,
        W_wear=0.1,
        Q_internal=0.1,
        xi_leak=0.0,
    )
    obs_crit = ObservationVector.from_state(state_crit)
    act_crit = ctrl.select_action(10, obs_crit)  # step 10 triggers flush
    assert act_crit.A_throttle == 20.0
    assert act_crit.A_valve == 100.0
    assert act_crit.A_flush == 1


def test_intervention_duration_and_clear() -> None:
    """Verify Intervention duration windows and clear() functionality."""
    reg = InterventionRegistry()
    inv = Intervention(target="V_pos", value=90.0, start_step=5, duration=3)
    reg.add(inv)

    assert not reg.has_intervention("V_pos", step=4)
    assert reg.has_intervention("V_pos", step=5)
    assert reg.has_intervention("V_pos", step=7)
    assert not reg.has_intervention("V_pos", step=8)
    assert reg.get_intervened_value("V_pos", step=6) == 90.0

    reg.clear()
    assert not reg.has_intervention("V_pos", step=6)


def test_action_and_noise_conversions() -> None:
    """Verify ActionVector and NoiseVector roundtrip conversions."""
    act = ActionVector.default_nominal()
    arr = act.to_array()
    act_rec = ActionVector.from_array(arr)
    assert act_rec == act

    d = act.to_dict()
    act_dict = ActionVector.from_dict(d)
    assert act_dict == act

    noise = NoiseVector.zeros()
    noise_arr = noise.to_array()
    noise_rec = NoiseVector.from_array(noise_arr)
    assert noise_rec == noise
    assert len(noise.to_dict()) == 12
