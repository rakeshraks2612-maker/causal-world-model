"""Tests for StateVector and ActionVector Schemas."""

import pytest
import numpy as np
from prism.simulator.state import (
    StateVector,
    OBSERVABLE_VARIABLES,
    LATENT_VARIABLES,
    ALL_STATE_VARIABLES,
)
from prism.simulator.actions import ActionVector


def test_state_vector_dimensions() -> None:
    """Verify 12-dimensional state schema with 8 observable and 4 latent."""
    assert len(OBSERVABLE_VARIABLES) == 8
    assert len(LATENT_VARIABLES) == 4
    assert len(ALL_STATE_VARIABLES) == 12

    state = StateVector.default_nominal()
    arr = state.to_array()
    assert arr.shape == (12,)
    assert state.observable_array().shape == (8,)
    assert state.latent_array().shape == (4,)


def test_state_vector_roundtrip() -> None:
    """Verify array and dict conversions are lossless."""
    state = StateVector.default_nominal()
    arr = state.to_array()
    reconstructed = StateVector.from_array(arr)
    assert reconstructed == state

    d = state.to_dict()
    reconstructed_dict = StateVector.from_dict(d)
    assert reconstructed_dict == state


def test_state_vector_validation() -> None:
    """Verify invalid inputs raise appropriate errors."""
    with pytest.raises(KeyError):
        StateVector.from_dict({"T_core": "invalid"})  # missing keys

    with pytest.raises(ValueError):
        d = StateVector.default_nominal().to_dict()
        d["T_core"] = float("nan")
        StateVector.from_dict(d)  # NaN value

    with pytest.raises(ValueError):
        StateVector.from_array(np.zeros(11))  # wrong length


def test_action_vector_bounds() -> None:
    """Verify action vector range constraints."""
    # Valid nominal
    act = ActionVector(A_valve=50.0, A_throttle=100.0, A_pump=2, A_flush=0)
    assert act.A_valve == 50.0

    # Invalid A_valve
    with pytest.raises(ValueError):
        ActionVector(A_valve=120.0, A_throttle=100.0, A_pump=2, A_flush=0)

    # Invalid A_throttle
    with pytest.raises(ValueError):
        ActionVector(A_valve=50.0, A_throttle=5.0, A_pump=2, A_flush=0)

    # Invalid A_pump
    with pytest.raises(ValueError):
        ActionVector(A_valve=50.0, A_throttle=100.0, A_pump=5, A_flush=0)

    # Invalid A_flush
    with pytest.raises(ValueError):
        ActionVector(A_valve=50.0, A_throttle=100.0, A_pump=2, A_flush=3)
