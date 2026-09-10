"""Training Pipeline for PRISM Baseline 003 (Intervention-Aware World Model).

Trains on data/intervention_aware/ with:
- 100 Train episodes across 5 balanced regimes (Nominal, High Load, Runaway, Recovery, Step Interventions)
- Normalization statistics computed strictly on the training partition
- Checkpoints saved to artifacts/baseline_003/
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.dataset import PrismWindowedDataset
from prism.training.normalization import ObservationNormalizer
from prism.training.trainer import WorldModelTrainer


def main() -> None:
    data_dir = Path("data/intervention_aware/learner")
    save_dir = Path("artifacts/baseline_003")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("      TRAINING PRISM BASELINE 003: INTERVENTION-AWARE WORLD MODEL        ")
    print("=========================================================================")

    # 1. Load episodes
    train_files = sorted(list((data_dir / "train").glob("*.npz")))
    val_files = sorted(list((data_dir / "validation").glob("*.npz")))

    print(f"Loading {len(train_files)} training episodes and {len(val_files)} validation episodes...")
    train_episodes = [LearnerEpisode.load_npz(f) for f in train_files]
    val_episodes = [LearnerEpisode.load_npz(f) for f in val_files]

    # 2. Compute training normalization statistics
    normalizer = ObservationNormalizer.fit(train_episodes)
    normalizer.save_yaml(save_dir / "normalization.yaml")
    print(f"✓ Saved normalization parameters to {save_dir / 'normalization.yaml'}")

    # 3. Build datasets and loaders
    context_len = 40
    train_dataset = PrismWindowedDataset(
        episodes=train_episodes,
        context_length=context_len,
        prediction_length=1,
        stride=2,
        normalizer=normalizer,
    )
    val_dataset = PrismWindowedDataset(
        episodes=val_episodes,
        context_length=context_len,
        prediction_length=1,
        stride=4,
        normalizer=normalizer,
    )

    print(f"Generated {len(train_dataset)} training windows and {len(val_dataset)} validation windows.")

    from prism.training.batching import create_dataloader

    train_loader = create_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    # 4. Model Configuration
    config = WorldModelConfig()
    config.model.latent_dim = 64
    config.encoder.hidden_dim = 64
    config.encoder.num_layers = 2
    config.transition.hidden_dim = 64
    config.transition.num_layers = 2
    config.transition.residual = True
    config.decoder.hidden_dim = 64
    config.decoder.num_layers = 2
    config.decoder.learn_variance = True
    config.loss_weights.lambda_obs = 1.0
    config.loss_weights.lambda_trans = 1.0
    config.loss_weights.beta_kl = 0.001
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-5
    config.training.max_epochs = 35
    config.training.early_stopping_patience = 8

    import yaml
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # 5. Instantiate Model and Trainer
    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=config,
        save_dir=save_dir,
    )

    print("\nStarting model training...")
    history = trainer.fit()

    print(f"\n=========================================================================")
    print(f"✓ Baseline 003 Training Completed!")
    print(f"Best Checkpoint: {save_dir / 'best.pt'}")
    print(f"=========================================================================")


if __name__ == "__main__":
    main()
