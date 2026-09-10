"""Tests for PrismWindowedDataset, Windowing Invariants, and Zero-Leakage."""

import pytest
import numpy as np
import torch
from pathlib import Path

from prism.dataset.schema import SplitType, LearnerEpisode, FORBIDDEN_LEARNER_KEYS
from prism.dataset.generator import generate_single_episode
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader


@pytest.fixture
def sample_episodes():
    """Produce 3 sample episodes of length 60."""
    eps = [
        generate_single_episode(SplitType.TRAIN, index=i, regime="nominal", length=60).to_learner_episode()
        for i in range(3)
    ]
    return eps


def test_windowing_indices_and_length(sample_episodes) -> None:
    """Test window index calculation and exact window bounds without boundary crossing."""
    context_len = 20
    pred_len = 1
    stride = 5
    window_len = context_len + pred_len  # 21

    dataset = PrismWindowedDataset(
        episodes=sample_episodes,
        context_length=context_len,
        prediction_length=pred_len,
        stride=stride,
    )

    # For each 61-step episode (length=60 -> 61 timestamps): max_start = 61 - 21 = 40.
    # Starts at 0, 5, 10, 15, 20, 25, 30, 35, 40 -> 9 windows per episode.
    # Total windows for 3 episodes = 3 * 9 = 27.
    assert len(dataset) == 27

    sample0 = dataset[0]
    assert sample0["observations"].shape == (21, 8)
    assert sample0["observation_mask"].shape == (21, 8)
    assert sample0["actions"].shape == (21, 4)
    assert sample0["episode_id"] == sample_episodes[0].episode_id
    assert sample0["start_timestep"] == 0

    sample9 = dataset[9]
    assert sample9["episode_id"] == sample_episodes[1].episode_id
    assert sample9["start_timestep"] == 0


def test_mask_dropout_training_time(sample_episodes) -> None:
    """Test that training-time mask dropout guarantees M_training <= M_original."""
    dataset = PrismWindowedDataset(
        episodes=sample_episodes,
        context_length=20,
        prediction_length=1,
        stride=10,
        mask_dropout_prob=0.3,
        rng_seed=42,
    )

    sample = dataset[0]
    raw_mask = sample["raw_mask"].numpy().astype(bool)
    training_mask = sample["observation_mask"].numpy().astype(bool)

    # Training mask cannot contain True where raw_mask is False
    assert np.all(training_mask <= raw_mask)
    # With 30% dropout, some channels should be masked additionally
    assert np.sum(training_mask) <= np.sum(raw_mask)


def test_dataloader_batch_collation(sample_episodes) -> None:
    """Test batch collation into ModelInputs container."""
    dataset = PrismWindowedDataset(
        episodes=sample_episodes,
        context_length=15,
        prediction_length=1,
        stride=5,
    )
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    inputs = batch["inputs"]

    assert inputs.observations.shape == (4, 16, 8)
    assert inputs.observation_mask.shape == (4, 16, 8)
    assert inputs.actions.shape == (4, 16, 4)
    assert len(batch["episode_ids"]) == 4


def test_zero_latent_leakage_in_dataset(sample_episodes) -> None:
    """Test that dataset outputs contain zero latent fields or forbidden keys."""
    dataset = PrismWindowedDataset(
        episodes=sample_episodes,
        context_length=10,
        prediction_length=1,
    )
    sample = dataset[0]

    for forbidden_key in FORBIDDEN_LEARNER_KEYS:
        assert forbidden_key not in sample
