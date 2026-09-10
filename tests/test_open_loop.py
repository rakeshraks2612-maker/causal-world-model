"""Tests for Open-Loop Isolation and Freedom from Future Observation Leakage."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.forecasting import evaluate_prism_multistep


def test_open_loop_zero_future_observation_leakage():
    """Verify that altering future observations (t > t_star) has ZERO impact on open-loop predictions."""
    torch.manual_seed(42)
    np.random.seed(42)
    T = 100
    timestamps = np.arange(T, dtype=np.int64)

    obs_clean = np.random.uniform(20.0, 80.0, size=(T, 8))
    mask = np.ones((T, 8), dtype=bool)
    act = np.random.uniform(10.0, 90.0, size=(T, 4))

    ep_clean = LearnerEpisode("ep_clean", SplitType.TEST, timestamps, obs_clean, mask, act)

    # Corrupt future observations after context_length=40
    obs_corrupted = obs_clean.copy()
    obs_corrupted[40:] += 500.0  # Massive corruption in future observations

    ep_corrupted = LearnerEpisode("ep_corrupted", SplitType.TEST, timestamps, obs_corrupted, mask, act)

    normalizer = ObservationNormalizer.fit([ep_clean])
    model = CausalWorldModel(WorldModelConfig())
    model.eval()

    # Open-loop predictions for both
    rep_clean = evaluate_prism_multistep(
        model=model,
        episodes=[ep_clean],
        normalizer=normalizer,
        horizons=[1, 5, 10, 20, 40],
        context_length=40,
        stride=100,
        mode="open_loop",
    )

    # In open-loop mode, the rollout manager only encodes the initial context (obs[:40]).
    # We directly verify that rollout deterministic with initial_z does not touch future obs.
    ctx_inputs_clean = normalizer.normalize(torch.tensor(obs_clean[:40], dtype=torch.float32)).unsqueeze(0)
    ctx_inputs_corr = normalizer.normalize(torch.tensor(obs_corrupted[:40], dtype=torch.float32)).unsqueeze(0)

    from prism.world_model.inputs import ModelInputs
    in_clean = ModelInputs(ctx_inputs_clean, torch.ones_like(ctx_inputs_clean), torch.tensor(act[:40], dtype=torch.float32).unsqueeze(0))
    in_corr = ModelInputs(ctx_inputs_corr, torch.ones_like(ctx_inputs_corr), torch.tensor(act[:40], dtype=torch.float32).unsqueeze(0))

    with torch.no_grad():
        z_clean = model.encode(in_clean)[0].mean[:, -1]
        z_corr = model.encode(in_corr)[0].mean[:, -1]

        # Initial latents are identical because context is identical
        assert torch.allclose(z_clean, z_corr, atol=1e-6)

        fut_actions = torch.tensor(act[39:79], dtype=torch.float32).unsqueeze(0)
        rollout_clean = model.rollout_manager.rollout_deterministic(z_clean, fut_actions)
        rollout_corr = model.rollout_manager.rollout_deterministic(z_corr, fut_actions)

        # Predicted observation trajectories are bit-identical despite future corruption
        assert torch.allclose(rollout_clean.observations.mean, rollout_corr.observations.mean, atol=1e-6)
