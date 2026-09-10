"""PRISM Dataset Management Subsystem.

Exposes schema contracts, generators, split managers, and integrity validators.
"""

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.splits import get_partition_seed, generate_episode_id
from prism.dataset.validators import (
    validate_learner_isolation,
    validate_split_disjointness,
    validate_temporal_monotonicity,
    validate_mask_consistency,
    validate_oracle_episode_integrity,
    DatasetValidationError,
)
from prism.dataset.manifest import DatasetManifest
from prism.dataset.generator import generate_single_episode, generate_split_dataset
from prism.dataset.scenarios import OODRegime, generate_ood_episode, generate_confounding_episode

from prism.dataset.counterfactuals import (
    CounterfactualSpec,
    CounterfactualFailureMetrics,
    CounterfactualHorizonEffect,
    OracleLatentTruth,
    OracleAbductionState,
    LearnerCounterfactualRecord,
    OracleCounterfactualRecord,
    execute_counterfactual_experiment,
    COUNTERFACTUAL_ACTION_MATRIX,
)

__all__ = [
    "SplitType",
    "LearnerEpisode",
    "OracleEpisode",
    "get_partition_seed",
    "generate_episode_id",
    "validate_learner_isolation",
    "validate_split_disjointness",
    "validate_temporal_monotonicity",
    "validate_mask_consistency",
    "validate_oracle_episode_integrity",
    "DatasetValidationError",
    "DatasetManifest",
    "generate_single_episode",
    "generate_split_dataset",
    "OODRegime",
    "generate_ood_episode",
    "generate_confounding_episode",
    "CounterfactualSpec",
    "CounterfactualFailureMetrics",
    "CounterfactualHorizonEffect",
    "OracleLatentTruth",
    "OracleAbductionState",
    "LearnerCounterfactualRecord",
    "OracleCounterfactualRecord",
    "execute_counterfactual_experiment",
    "COUNTERFACTUAL_ACTION_MATRIX",
]
