"""Unit Tests for Intervention Operator and Specification."""

import pytest
import numpy as np
import torch

from prism.intervention.spec import (
    InterventionSpec,
    InterventionType,
    state_clamp,
    action_control,
)
from prism.intervention.operator import (
    InterventionOperator,
    OBSERVABLE_TO_IDX,
    ACTION_TO_IDX,
)
from prism.training.normalization import ObservationNormalizer, NormalizationStats


def test_intervention_spec_validation():
    """Test valid creation and boundary rejection in InterventionSpec."""
    spec = state_clamp("V_pos", 85.0, intervention_time=20)
    assert spec.target == "V_pos"
    assert spec.value == 85.0
    assert spec.intervention_time == 20
    assert spec.duration is None
    assert spec.is_state_clamp

    # Out of bounds value
    with pytest.raises(ValueError, match="V_pos value must be in"):
        state_clamp("V_pos", 150.0)

    # Invalid target
    with pytest.raises(ValueError, match="Unknown intervention target"):
        state_clamp("NonExistentVar", 10.0)

    # Negative intervention time
    with pytest.raises(ValueError, match="must be non-negative"):
        state_clamp("V_pos", 50.0, intervention_time=-1)

    # Invalid action bounds
    with pytest.raises(ValueError, match="A_throttle value must be in"):
        action_control("A_throttle", 5.0)


def test_intervention_operator_action_modification():
    """Test that InterventionOperator correctly updates action sequences."""
    # Future actions: 10 steps, 4 action channels
    base_actions = torch.zeros(1, 10, 4)
    base_actions[:, :, 0] = 50.0  # A_valve
    base_actions[:, :, 1] = 100.0 # A_throttle

    # Pulse intervention starting at t*=20 for 3 steps
    spec = action_control("A_valve", 85.0, intervention_time=20, duration=3)
    op = InterventionOperator([spec])

    # Case 1: Horizon starting at t_star=20
    mod_acts = op.get_modified_actions(base_actions, t_star=20)
    
    # Steps 0, 1, 2 (t=20, 21, 22) should be modified to 85.0
    assert torch.allclose(mod_acts[0, 0:3, 0], torch.tensor([85.0, 85.0, 85.0]))
    # Steps 3..9 (t=23..29) should remain 50.0
    assert torch.allclose(mod_acts[0, 3:, 0], torch.full((7,), 50.0))
    # A_throttle should remain untouched
    assert torch.allclose(mod_acts[0, :, 1], torch.full((10,), 100.0))


def test_intervention_operator_observation_clamping():
    """Test that InterventionOperator clamps observation channels correctly."""
    base_obs = torch.zeros(1, 10, 8)
    base_obs[:, :, OBSERVABLE_TO_IDX["V_pos"]] = 50.0

    spec = state_clamp("V_pos", 85.0, intervention_time=20)
    op = InterventionOperator([spec])

    clamped_obs = op.apply_observation_clamps(base_obs, t_star=20)
    # V_pos channel should be 85.0 everywhere
    assert torch.allclose(clamped_obs[0, :, OBSERVABLE_TO_IDX["V_pos"]], torch.full((10,), 85.0))
    # Other channels should remain 0.0
    assert torch.allclose(clamped_obs[0, :, OBSERVABLE_TO_IDX["T_core"]], torch.zeros(10))


def test_leaf_variable_non_causal_action_invariance():
    """Test that intervening on leaf variable Vib_pump leaves action tensors untouched."""
    base_actions = torch.ones(1, 10, 4) * 42.0
    spec = state_clamp("Vib_pump", 5.0, intervention_time=20)
    op = InterventionOperator([spec])

    mod_acts = op.get_modified_actions(base_actions, t_star=20)
    # Actions must be completely identical
    assert torch.allclose(mod_acts, base_actions)
