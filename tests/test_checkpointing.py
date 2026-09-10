"""Tests for Checkpoint Persistence, State Loading, and Inference Invariance."""

import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.training.checkpointing import save_checkpoint, load_checkpoint
from prism.training.normalization import ObservationNormalizer, NormalizationStats


def test_checkpoint_save_and_load_roundtrip(tmp_path) -> None:
    """Test saving and loading a checkpoint, asserting identical inference outputs."""
    config = WorldModelConfig()
    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    stats = NormalizationStats(means=[25.0] * 8, stds=[2.0] * 8)
    normalizer = ObservationNormalizer(stats=stats)

    ckpt_path = tmp_path / "test_ckpt.pt"

    # Save checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        global_step=120,
        val_loss=1.234,
        val_mae=0.567,
        config=config,
        normalizer=normalizer,
        save_path=ckpt_path,
        seed=42,
    )
    assert ckpt_path.exists()

    # Create new model instance with random initialization
    new_model = CausalWorldModel(config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    # Load weights
    meta = load_checkpoint(ckpt_path, new_model, new_optimizer)

    assert meta.epoch == 5
    assert meta.global_step == 120
    assert meta.val_mae == 0.567
    assert meta.normalization_stats is not None

    # Test identical inference outputs
    test_inputs = ModelInputs(
        observations=torch.randn(2, 10, 8),
        observation_mask=torch.ones(2, 10, 8),
        actions=torch.randn(2, 10, 4),
    )

    model.eval()
    new_model.eval()

    with torch.no_grad():
        dist_orig, _ = model.encode(test_inputs)
        dist_loaded, _ = new_model.encode(test_inputs)

    np.testing.assert_allclose(dist_orig.mean.numpy(), dist_loaded.mean.numpy(), atol=1e-6)
    np.testing.assert_allclose(dist_orig.logvar.numpy(), dist_loaded.logvar.numpy(), atol=1e-6)
