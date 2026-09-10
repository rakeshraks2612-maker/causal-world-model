"""Tests for Action Alignment and Causal Flow in Latent Transition."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel


def test_action_at_t_affects_transition_t_plus_1_only():
    """Verify that changing action A_t modifies prediction at t+1, but has zero effect on Z_<=t and transitions at t' < t."""
    torch.manual_seed(42)
    B, T, obs_dim, act_dim = 1, 8, 8, 4
    
    config = WorldModelConfig()
    model = CausalWorldModel(config)
    model.eval()
    
    # Base sequence
    obs = torch.randn(B, T, obs_dim)
    mask = torch.ones(B, T, obs_dim)
    act_base = torch.zeros(B, T, act_dim)
    
    inputs_base = ModelInputs(observations=obs, observation_mask=mask, actions=act_base)
    
    with torch.no_grad():
        post_base, _ = model.encode(inputs_base)
        z_base = post_base.mean[:, :-1]          # [1, T-1, d_z]
        trans_base = model.transition_step(z_base, act_base[:, :-1]) # [1, T-1, d_z]
        pred_base = model.decode(trans_base.mean)                    # [1, T-1, 8]
        
    # Now modify action at t_star = 3 only
    t_star = 3
    act_mod = act_base.clone()
    act_mod[:, t_star, 0] += 50.0  # e.g., valve command changed at step 3
    
    with torch.no_grad():
        # Using the SAME historical latent state z_base at t=0..T-2, we feed act_mod:
        trans_mod = model.transition_step(z_base, act_mod[:, :-1])
        pred_mod = model.decode(trans_mod.mean)
        
    # Check predictions at t < t_star (indices 0, 1, 2 representing predictions for t=1, 2, 3)
    # These must be IDENTICAL
    for t_idx in range(t_star):
        assert torch.allclose(pred_mod.mean[:, t_idx], pred_base.mean[:, t_idx], atol=1e-6), (
            f"Prediction at step {t_idx+1} changed unexpectedly when action was modified at step {t_star}!"
        )
        
    # Check prediction at t = t_star (index 3, representing prediction for t=4)
    # This MUST change because A_3 changed!
    diff_t_plus_1 = torch.norm(pred_mod.mean[:, t_star] - pred_base.mean[:, t_star]).item()
    assert diff_t_plus_1 > 1e-4, "Action at t_star did not change prediction at t_star + 1!"
    
    # Check predictions at t > t_star (indices 4, 5, 6)
    # In single-step transition evaluation with fixed z_base, actions at t > t_star are unchanged, so those single-step predictions match
    for t_idx in range(t_star + 1, T - 1):
        assert torch.allclose(pred_mod.mean[:, t_idx], pred_base.mean[:, t_idx], atol=1e-6), (
            f"Prediction at step {t_idx+1} changed unexpectedly when single-step action was modified at step {t_star}!"
        )


def test_autoregressive_rollout_action_causality():
    """Verify that in multi-step rollout, action at t_star affects all future states t > t_star."""
    torch.manual_seed(42)
    B, H, d_z, act_dim = 1, 10, 32, 4
    
    config = WorldModelConfig()
    model = CausalWorldModel(config)
    model.eval()
    
    initial_z = torch.randn(B, d_z)
    future_actions_base = torch.zeros(B, H, act_dim)
    
    t_star = 2
    future_actions_mod = future_actions_base.clone()
    future_actions_mod[:, t_star, 0] = 80.0  # Intervention at step t_star
    
    traj_base = model.rollout_manager.rollout_deterministic(initial_z, future_actions_base)
    traj_mod = model.rollout_manager.rollout_deterministic(initial_z, future_actions_mod)
    
    # Pre-intervention steps (h < t_star) must be identical
    for h in range(t_star):
        assert torch.allclose(traj_mod.latent_mean[:, h], traj_base.latent_mean[:, h], atol=1e-6)
        assert torch.allclose(traj_mod.observations.mean[:, h], traj_base.observations.mean[:, h], atol=1e-6)
        
    # Post-intervention steps (h >= t_star) must diverge due to causal propagation from modified action at t_star
    for h in range(t_star, H):
        latent_diff = torch.norm(traj_mod.latent_mean[:, h] - traj_base.latent_mean[:, h]).item()
        assert latent_diff > 1e-4, f"Rollout at horizon {h} did not diverge after action intervention at {t_star}!"
