"""Tests for RNG Manager and Exogenous Stochastic Innovations."""

import numpy as np
from prism.simulator.noise import RNGManager, NoiseVector


def test_rng_manager_reproducibility() -> None:
    """Verify that two RNGManagers with identical seed produce identical random numbers."""
    rng1 = RNGManager(seed=42)
    rng2 = RNGManager(seed=42)

    sigmas = {"sigma_p_elec": 0.02, "sigma_q": 0.04}
    n1 = rng1.sample_process_innovations(sigmas)
    n2 = rng2.sample_process_innovations(sigmas)

    np.testing.assert_array_equal(n1.to_array(), n2.to_array())

    # Ambient step
    t1, u1 = rng1.sample_ambient_step(25.0, 0.02, 27.0, 0.15, 1.0)
    t2, u2 = rng2.sample_ambient_step(25.0, 0.02, 27.0, 0.15, 1.0)
    assert t1 == t2
    assert u1 == u2


def test_rng_manager_seed_difference() -> None:
    """Verify that different seeds produce distinct stochastic trajectories."""
    rng1 = RNGManager(seed=42)
    rng2 = RNGManager(seed=99)

    sigmas = {"sigma_p_elec": 0.02, "sigma_q": 0.04}
    n1 = rng1.sample_process_innovations(sigmas)
    n2 = rng2.sample_process_innovations(sigmas)

    assert not np.array_equal(n1.to_array(), n2.to_array())
