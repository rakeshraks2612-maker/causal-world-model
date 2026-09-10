"""PRISM Uncertainty-Aware Causal World Model Package."""

from prism.world_model.config import (
    WorldModelConfig,
    ModelArchitectureConfig,
    EncoderConfig,
    TransitionConfig,
    DecoderConfig,
    LossWeightsConfig,
    TrainingConfig,
    RolloutConfig,
)
from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import (
    LatentDistribution,
    ObservationDistribution,
    LatentState,
)
from prism.world_model.encoder import BaseEncoder, GRUEncoder
from prism.world_model.transition import BaseTransition, MLPTransition
from prism.world_model.decoder import BaseDecoder, MLPDecoder
from prism.world_model.uncertainty import (
    UncertaintyMetrics,
    compute_prediction_intervals,
    evaluate_calibration,
    compute_ood_uncertainty_score,
)
from prism.world_model.rollout import RolloutTrajectory, RolloutManager
from prism.world_model.losses import LossOutput, WorldModelLossCalculator
from prism.world_model.model import CausalWorldModel

__all__ = [
    "WorldModelConfig",
    "ModelArchitectureConfig",
    "EncoderConfig",
    "TransitionConfig",
    "DecoderConfig",
    "LossWeightsConfig",
    "TrainingConfig",
    "RolloutConfig",
    "ModelInputs",
    "LatentDistribution",
    "ObservationDistribution",
    "LatentState",
    "BaseEncoder",
    "GRUEncoder",
    "BaseTransition",
    "MLPTransition",
    "BaseDecoder",
    "MLPDecoder",
    "UncertaintyMetrics",
    "compute_prediction_intervals",
    "evaluate_calibration",
    "compute_ood_uncertainty_score",
    "RolloutTrajectory",
    "RolloutManager",
    "LossOutput",
    "WorldModelLossCalculator",
    "CausalWorldModel",
]
