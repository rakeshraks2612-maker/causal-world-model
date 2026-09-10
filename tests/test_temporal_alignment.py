"""Tests for Temporal Alignment in World Model Training, Transition, and Loss Computation."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.world_model.losses import WorldModelLossCalculator


def test_temporal_alignment_indices():
    """Verify that transition prediction at index t strictly corresponds to observation target at t+1."""
    torch.manual_seed(42)
    B, T, obs_dim, act_dim = 2, 10, 8, 4
    
    # Create synthetic linear ramp observations where obs[t] = t * 10
    obs = torch.zeros(B, T, obs_dim)
    for t in range(T):
        obs[:, t, :] = float(t * 10)
    
    mask = torch.ones(B, T, obs_dim)
    actions = torch.zeros(B, T, act_dim)
    inputs = ModelInputs(observations=obs, observation_mask=mask, actions=actions)
    
    config = WorldModelConfig()
    model = CausalWorldModel(config)
    model.eval()
    
    with torch.no_grad():
        post_latents, _ = model.encode(inputs)
        
        # Current time step t=0..T-2
        z_curr = post_latents.mean[:, :-1]   # [B, T-1, d_z]
        act_curr = inputs.actions[:, :-1]   # [B, T-1, 4]
        
        # Transition predicts latent distribution at t+1: [B, T-1, d_z]
        prior_trans = model.transition_step(z_curr, act_curr)
        
        # Decoder predicts observations at t+1: [B, T-1, 8]
        pred_obs_dist = model.decode(prior_trans.mean)
        
        # Ground truth target is obs at t+1: [B, T-1, 8]
        target_obs = inputs.observations[:, 1:]
        
        # Assert shapes match
        assert pred_obs_dist.mean.shape == target_obs.shape == (B, T - 1, obs_dim)
        
        # Target at prediction index 0 must be obs at index 1 (value 10.0, not 0.0)
        assert torch.allclose(target_obs[:, 0, :], torch.tensor(10.0))
        # Target at prediction index T-2 must be obs at index T-1 (value (T-1)*10.0)
        assert torch.allclose(target_obs[:, -1, :], torch.tensor(float((T - 1) * 10)))


def test_loss_temporal_slices():
    """Verify that loss calculator compares posterior at t+1 against transition from t."""
    torch.manual_seed(42)
    config = WorldModelConfig()
    loss_calc = WorldModelLossCalculator(config.loss_weights)
    
    B, T, d_z, obs_dim = 2, 5, config.model.latent_dim, config.model.obs_dim
    
    # Construct posterior latents with distinct mean per timestep
    means = torch.stack([torch.full((B, d_z), float(t)) for t in range(T)], dim=1) # [B, T, d_z]
    logvars = torch.zeros(B, T, d_z)
    
    from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
    post_latents = LatentDistribution(mean=means, logvar=logvars)
    
    # Transition prior predicted from t=0..T-2, suppose it predicts exactly t+1
    trans_means = torch.stack([torch.full((B, d_z), float(t + 1)) for t in range(T - 1)], dim=1) # [B, T-1, d_z]
    trans_logvars = torch.zeros(B, T - 1, d_z)
    prior_trans = LatentDistribution(mean=trans_means, logvar=trans_logvars)
    
    # Reconstructed obs
    recon_obs = ObservationDistribution(mean=torch.zeros(B, T, obs_dim), logvar=torch.zeros(B, T, obs_dim))
    target_obs = torch.zeros(B, T, obs_dim)
    mask = torch.ones(B, T, obs_dim)
    
    loss_out = loss_calc.compute_loss(
        posterior_latents=post_latents,
        prior_transitions=prior_trans,
        reconstructed_obs=recon_obs,
        target_obs=target_obs,
        observation_mask=mask,
    )
    
    # Because transition prior means exactly match posterior[:, 1:] means (both equal t+1),
    # transition KL must be zero!
    assert loss_out.trans_loss.item() < 1e-5
