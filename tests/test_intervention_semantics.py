"""Unit Tests for Intervention Semantics and Invariants."""

import numpy as np
import torch
import pytest

from prism.intervention.spec import state_clamp, action_control, InterventionSpec
from prism.intervention.operator import InterventionOperator, OBSERVABLE_TO_IDX
from prism.intervention.effects import compute_causal_effects
from prism.intervention.validation import (
    check_pre_intervention_identity,
    check_non_descendant_invariance,
)
from prism.simulator.state import OBSERVABLE_VARIABLES


def test_persistent_vs_pulse_semantics():
    """Test that persistent and pulse interventions exhibit correct temporal activation profiles."""
    spec_perm = state_clamp("V_pos", 85.0, intervention_time=10, duration=None)
    spec_pulse = state_clamp("V_pos", 85.0, intervention_time=10, duration=5)

    # Before intervention
    assert not spec_perm.is_active(9)
    assert not spec_pulse.is_active(9)

    # During active window
    assert spec_perm.is_active(10)
    assert spec_pulse.is_active(10)
    assert spec_perm.is_active(14)
    assert spec_pulse.is_active(14)

    # After pulse duration
    assert spec_perm.is_active(15)
    assert not spec_pulse.is_active(15)
    assert spec_perm.is_active(50)
    assert not spec_pulse.is_active(50)


def test_pre_intervention_identity():
    """Test invariant that pre-intervention observations are strictly identical between factual and counterfactual."""
    pre_obs_fact = np.random.randn(21, 8)
    pre_obs_int = np.copy(pre_obs_fact)

    passed, max_diff = check_pre_intervention_identity(pre_obs_fact, pre_obs_int)
    assert passed
    assert max_diff == 0.0


def test_causal_effects_calculation():
    """Test compute_causal_effects on synthetic baseline and intervened trajectories."""
    # Baseline: constant 70.0 core temp, 30.0 flow
    base_obs = np.zeros((40, 8))
    base_obs[:, 0] = 70.0
    base_obs[:, 3] = 30.0

    # Intervened: 60.0 core temp (-10), 45.0 flow (+15)
    int_obs = np.zeros((40, 8))
    int_obs[:, 0] = 60.0
    int_obs[:, 3] = 45.0

    effects = compute_causal_effects(base_obs, int_obs, horizons=(1, 5, 10, 20, 40), t_star=20)

    # Check horizon deltas
    h10 = effects.horizon_effects[10]
    assert np.isclose(h10.delta_t_core, -10.0)
    assert np.isclose(h10.delta_f_cool, 15.0)

    # Check peak metrics
    assert np.isclose(effects.peak_t_core_base, 70.0)
    assert np.isclose(effects.peak_t_core_int, 60.0)
    assert np.isclose(effects.delta_peak_t_core, -10.0)
    assert np.isclose(effects.min_flow_base, 30.0)
    assert np.isclose(effects.min_flow_int, 45.0)
    assert np.isclose(effects.delta_min_flow, 15.0)
