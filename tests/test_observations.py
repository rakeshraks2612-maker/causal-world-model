"""Tests for Observation Model, Sensor Noise, and Telemetry Masking."""

import numpy as np
from prism.simulator.state import StateVector, OBSERVABLE_VARIABLES, LATENT_VARIABLES
from prism.simulator.observations import ObservationVector
from prism.simulator.simulator import THCSimulator
from prism.simulator.policies import NominalController


def test_latent_variables_excluded_from_observations() -> None:
    """Verify that observations contain ONLY the 8 observable variables."""
    state = StateVector.default_nominal()
    obs = ObservationVector.from_state(state)

    for latent_var in LATENT_VARIABLES:
        assert not hasattr(obs, latent_var), f"Observation leaked latent variable: {latent_var}"

    for obs_var in OBSERVABLE_VARIABLES:
        assert hasattr(obs, obs_var), f"Observation missing observable variable: {obs_var}"


def test_sensor_noise_distribution() -> None:
    """Verify sensor noise perturbs observations around true state."""
    state = StateVector.default_nominal()
    rng = np.random.default_rng(42)

    samples = [ObservationVector.from_state(state, rng=rng).T_core for _ in range(500)]
    mean_obs = float(np.mean(samples))
    std_obs = float(np.std(samples))

    # Mean should be close to true T_core (68.5) and std close to 0.40
    assert abs(mean_obs - state.T_core) < 0.1
    assert abs(std_obs - 0.40) < 0.05


def test_missing_telemetry_masking() -> None:
    """Verify missing telemetry sets observation to NaN while ground truth is preserved."""
    sim = THCSimulator(seed=42)
    missing = {"P_sys", "F_cool"}

    ep = sim.run_episode(
        policy=NominalController(),
        length=20,
        missing_channels=missing,
    )

    # Observations for P_sys (index 2) and F_cool (index 3) must be NaN
    assert np.all(np.isnan(ep.observations[:, 2]))
    assert np.all(np.isnan(ep.observations[:, 3]))

    # Other observable channels must NOT be NaN
    assert not np.any(np.isnan(ep.observations[:, 0]))  # T_core
    assert not np.any(np.isnan(ep.observations[:, 1]))  # T_cool

    # Ground truth state for P_sys and F_cool must remain completely intact and non-NaN!
    assert not np.any(np.isnan(ep.ground_truth_states[:, 2]))
    assert not np.any(np.isnan(ep.ground_truth_states[:, 3]))
