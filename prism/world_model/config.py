"""PRISM World Model Configuration and Hyperparameter Schema."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ModelArchitectureConfig:
    obs_dim: int = 8
    action_dim: int = 4
    latent_dim: int = 32
    min_std: float = 1e-3
    max_std: float = 10.0


@dataclass
class EncoderConfig:
    type: str = "gru"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


@dataclass
class TransitionConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    residual: bool = True


@dataclass
class DecoderConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learn_variance: bool = True


@dataclass
class LossWeightsConfig:
    lambda_obs: float = 1.0
    lambda_trans: float = 1.0
    beta_kl: float = 0.01
    lambda_rollout: float = 0.0
    rollout_horizon: int = 1


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 10.0
    max_epochs: int = 50
    early_stopping_patience: int = 10


@dataclass
class RolloutConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])
    num_samples: int = 50
    confidence_level: float = 0.90


@dataclass
class WorldModelConfig:
    """Master configuration schema for PRISM Causal World Model."""

    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> WorldModelConfig:
        path = Path(yaml_path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            model=ModelArchitectureConfig(**data.get("model", {})),
            encoder=EncoderConfig(**data.get("encoder", {})),
            transition=TransitionConfig(**data.get("transition", {})),
            decoder=DecoderConfig(**data.get("decoder", {})),
            loss_weights=LossWeightsConfig(**data.get("loss_weights", {})),
            training=TrainingConfig(**data.get("training", {})),
            rollout=RolloutConfig(**data.get("rollout", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.__dict__,
            "encoder": self.encoder.__dict__,
            "transition": self.transition.__dict__,
            "decoder": self.decoder.__dict__,
            "loss_weights": self.loss_weights.__dict__,
            "training": self.training.__dict__,
            "rollout": self.rollout.__dict__,
        }
