"""Dataset Integrity and Anti-Leakage Validators.

Provides automated testable validations for:
- Strict Learner/Oracle isolation (Zero-leakage guarantee)
- Episode-level split disjointness
- Temporal monotonicity and step continuity
- Observation mask / NaN consistency
- Tensor shape matching
"""

from __future__ import annotations
from typing import List, Set, Tuple, Dict, Any
import numpy as np

from prism.dataset.schema import LearnerEpisode, OracleEpisode, FORBIDDEN_LEARNER_KEYS
from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES


class DatasetValidationError(Exception):
    """Raised when a dataset integrity or leakage invariant is breached."""
    pass


def validate_learner_isolation(episode: LearnerEpisode) -> bool:
    """Verify that a LearnerEpisode contains zero latent or ground-truth leakage."""
    # Check observation shape and column count
    if episode.observations.shape[1] != 8:
        raise DatasetValidationError(
            f"Learner observation dimension must be exactly 8, got {episode.observations.shape[1]}"
        )

    # Check for forbidden keys in metadata
    for forbidden in FORBIDDEN_LEARNER_KEYS:
        if forbidden in episode.metadata:
            raise DatasetValidationError(
                f"CRITICAL LEAKAGE: LearnerEpisode metadata contains forbidden latent key '{forbidden}'"
            )

    # Verify no oracle properties exist
    for forbidden_attr in ["ground_truth_states", "exogenous_noise", "latent_states"]:
        if hasattr(episode, forbidden_attr):
            raise DatasetValidationError(
                f"CRITICAL LEAKAGE: LearnerEpisode has forbidden oracle attribute '{forbidden_attr}'"
            )

    return True


def validate_split_disjointness(
    train_episodes: List[LearnerEpisode | OracleEpisode],
    val_episodes: List[LearnerEpisode | OracleEpisode],
    test_episodes: List[LearnerEpisode | OracleEpisode],
) -> bool:
    """Verify episode-level split disjointness (Train ∩ Val ∩ Test = ∅)."""
    train_ids = {ep.episode_id for ep in train_episodes}
    val_ids = {ep.episode_id for ep in val_episodes}
    test_ids = {ep.episode_id for ep in test_episodes}

    train_val_overlap = train_ids.intersection(val_ids)
    if train_val_overlap:
        raise DatasetValidationError(f"Split leakage detected: Train and Val share episodes: {train_val_overlap}")

    train_test_overlap = train_ids.intersection(test_ids)
    if train_test_overlap:
        raise DatasetValidationError(f"Split leakage detected: Train and Test share episodes: {train_test_overlap}")

    val_test_overlap = val_ids.intersection(test_ids)
    if val_test_overlap:
        raise DatasetValidationError(f"Split leakage detected: Val and Test share episodes: {val_test_overlap}")

    return True


def validate_temporal_monotonicity(timestamps: np.ndarray) -> bool:
    """Verify timestamps are strictly monotonically increasing."""
    if len(timestamps) < 2:
        return True
    diffs = np.diff(timestamps)
    if not np.all(diffs > 0):
        raise DatasetValidationError(f"Timestamps are not strictly monotonically increasing: min diff = {np.min(diffs)}")
    return True


def validate_mask_consistency(observations: np.ndarray, mask: np.ndarray) -> bool:
    """Verify that observation_mask == True iff observation is not NaN."""
    nan_mask = np.isnan(observations)
    expected_mask = ~nan_mask
    if not np.array_equal(mask, expected_mask):
        raise DatasetValidationError("Observation mask is inconsistent with NaN patterns in observations array!")
    return True


def validate_oracle_episode_integrity(ep: OracleEpisode) -> bool:
    """Comprehensive integrity check for OracleEpisode."""
    t_len = len(ep.timestamps)
    validate_temporal_monotonicity(ep.timestamps)
    validate_mask_consistency(ep.observations, ep.observation_mask)

    if ep.observations.shape != (t_len, 8):
        raise DatasetValidationError(f"Observations shape {ep.observations.shape} != ({t_len}, 8)")
    if ep.actions.shape != (t_len, 4):
        raise DatasetValidationError(f"Actions shape {ep.actions.shape} != ({t_len}, 4)")
    if ep.ground_truth_states.shape != (t_len, 12):
        raise DatasetValidationError(f"Ground truth states shape {ep.ground_truth_states.shape} != ({t_len}, 12)")
    if ep.exogenous_noise.shape != (t_len, 12):
        raise DatasetValidationError(f"Exogenous noise shape {ep.exogenous_noise.shape} != ({t_len}, 12)")

    return True
