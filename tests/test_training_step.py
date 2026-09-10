"""Tests for World Model Training Step, Loss Computation, and Overfit Diagnostic."""

import pytest
import numpy as np
import torch

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader
from prism.training.normalization import ObservationNormalizer
from prism.training.trainer import WorldModelTrainer


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset of 2 episodes."""
    eps = [
        generate_single_episode(SplitType.TRAIN, index=i, regime="nominal", length=40).to_learner_episode()
        for i in range(2)
    ]
    norm = ObservationNormalizer.fit(eps)
    ds = PrismWindowedDataset(
        episodes=eps,
        context_length=15,
        prediction_length=1,
        stride=5,
        normalizer=norm,
    )
    return ds, norm


def test_single_training_epoch(tiny_dataset) -> None:
    """Test executing a training epoch and updating parameters."""
    ds, norm = tiny_dataset
    loader = create_dataloader(ds, batch_size=2, shuffle=True)

    config = WorldModelConfig()
    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        normalizer=norm,
        config=config,
    )

    initial_params = [p.clone().detach() for p in model.parameters()]
    metrics = trainer.train_epoch()

    assert "loss/train_total" in metrics
    assert np.isfinite(metrics["loss/train_total"])

    # Verify at least some parameters were updated
    changed = False
    for p_init, p_new in zip(initial_params, model.parameters()):
        if not torch.equal(p_init, p_new):
            changed = True
            break
    assert changed, "Parameters were not updated during training step!"


def test_tiny_subset_overfit_diagnostic(tiny_dataset) -> None:
    """Critical Diagnostic (Task 3.2.17): Verify that World Model drives loss downward on a tiny batch."""
    ds, norm = tiny_dataset
    loader = create_dataloader(ds, batch_size=len(ds), shuffle=False)

    config = WorldModelConfig()
    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        normalizer=norm,
        config=config,
    )

    # Initial loss
    batch = next(iter(loader))
    with torch.no_grad():
        initial_loss = model.compute_loss(batch["inputs"]).total_loss.item()

    # Train for 40 steps on this single tiny batch
    final_loss = trainer.overfit_diagnostic(num_epochs=40)

    # Loss must decrease substantially
    assert final_loss < initial_loss
    assert final_loss < initial_loss * 0.7  # Significant loss reduction


def test_trainer_fit_and_evaluate(tiny_dataset, tmp_path) -> None:
    """Test full trainer.fit() and trainer.evaluate() execution."""
    ds, norm = tiny_dataset
    loader = create_dataloader(ds, batch_size=2, shuffle=True)

    config = WorldModelConfig()
    config.training.early_stopping_patience = 2
    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        normalizer=norm,
        config=config,
        save_dir=tmp_path / "checkpoints",
    )

    result = trainer.fit(max_epochs=3)
    assert "best_val_mae" in result
    assert result["final_epoch"] >= 1

    val_loss, summary = trainer.evaluate(loader)
    assert np.isfinite(val_loss)
    assert summary.overall_mae > 0.0
    assert "T_core" in summary.per_channel

