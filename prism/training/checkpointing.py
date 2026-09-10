"""Model Checkpointing and State Persistence Subsystem."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import yaml

from prism.world_model.config import WorldModelConfig
from prism.training.normalization import NormalizationStats, ObservationNormalizer


@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    random_seed: int
    val_loss: float
    val_mae: float
    config: Dict[str, Any]
    normalization_stats: Optional[Dict[str, Any]] = None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    val_mae: float,
    config: WorldModelConfig,
    normalizer: Optional[ObservationNormalizer],
    save_path: str | Path,
    seed: int = 42,
) -> None:
    """Save comprehensive training checkpoint to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    norm_dict = normalizer.stats.to_dict() if normalizer and normalizer.stats else None

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "random_seed": seed,
        "config": config.to_dict(),
        "normalization_stats": norm_dict,
    }

    torch.save(checkpoint_dict, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> CheckpointMetadata:
    """Load model weights and training metadata from checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    meta = CheckpointMetadata(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        random_seed=int(checkpoint.get("random_seed", 42)),
        val_loss=float(checkpoint.get("val_loss", 0.0)),
        val_mae=float(checkpoint.get("val_mae", 0.0)),
        config=checkpoint.get("config", {}),
        normalization_stats=checkpoint.get("normalization_stats"),
    )
    return meta
