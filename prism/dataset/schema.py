"""Dataset Schema and Learner vs Oracle Boundary Definitions.

Establishes strict two-tier data contracts:
1. LearnerEpisode: Strictly what the future learning model receives (noisy observations, masks, actions).
   Contains zero ground-truth latent state or exogenous noise.
2. OracleEpisode: Comprehensive evaluation record retained by the benchmark evaluator (complete 12-dim state,
   12-dim exogenous noise, failure telemetry).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np

from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES, ALL_STATE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES
from prism.simulator.noise import NOISE_VARIABLES


class SplitType(str, Enum):
    """Dataset partition types."""
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    OOD = "ood"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"
    STRESS = "stress"


FORBIDDEN_LEARNER_KEYS: Set[str] = {
    "T_amb",
    "W_wear",
    "Q_internal",
    "xi_leak",
    "latent_states",
    "ground_truth_states",
    "exogenous_noise",
    "u_amb",
    "u_w",
    "u_q_int",
    "u_xi",
}


@dataclass
class LearnerEpisode:
    """The strict data representation exposed to learning models.
    
    Fields:
        episode_id: Unique identifier for the episode
        split: Partition membership (TRAIN, VAL, TEST, etc.)
        timestamps: Discrete simulation timestamps [T]
        observations: Telemetry measurements [T, 8] (NaN for masked/missing sensors)
        observation_mask: Boolean indicator [T, 8] (True = observed, False = missing/masked)
        actions: Executed control actions [T, 4]
        metadata: Sanitized non-oracle metadata (policy name, length, missingness type)
    """

    episode_id: str
    split: SplitType
    timestamps: np.ndarray  # Shape: [T]
    observations: np.ndarray  # Shape: [T, 8]
    observation_mask: np.ndarray  # Shape: [T, 8]
    actions: np.ndarray  # Shape: [T, 4]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce strict validation of shapes, types, and absence of latent information."""
        if not isinstance(self.timestamps, np.ndarray) or self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D numpy array, got {type(self.timestamps)}")
        t_len = len(self.timestamps)

        if self.observations.shape != (t_len, 8):
            raise ValueError(f"observations shape must be ({t_len}, 8), got {self.observations.shape}")
        if self.observation_mask.shape != (t_len, 8):
            raise ValueError(f"observation_mask shape must be ({t_len}, 8), got {self.observation_mask.shape}")
        if self.actions.shape != (t_len, 4):
            raise ValueError(f"actions shape must be ({t_len}, 4), got {self.actions.shape}")

        # Check for forbidden latent metadata leakage
        for k in self.metadata:
            if k in FORBIDDEN_LEARNER_KEYS:
                raise ValueError(f"CRITICAL LEAKAGE: LearnerEpisode metadata contains forbidden latent key '{k}'")

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def save_npz(self, file_path: str | Path) -> None:
        """Serialize learner episode to .npz archive."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            episode_id=np.array(self.episode_id),
            split=np.array(self.split.value if isinstance(self.split, SplitType) else str(self.split)),
            timestamps=self.timestamps,
            observations=self.observations,
            observation_mask=self.observation_mask,
            actions=self.actions,
            metadata=np.array(self.metadata, dtype=object),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> LearnerEpisode:
        """Load learner episode from .npz archive."""
        data = np.load(file_path, allow_pickle=True)
        meta_arr = data["metadata"]
        meta = meta_arr.item() if meta_arr.ndim == 0 else dict(meta_arr)
        return cls(
            episode_id=str(data["episode_id"]),
            split=SplitType(str(data["split"])),
            timestamps=data["timestamps"],
            observations=data["observations"],
            observation_mask=data["observation_mask"],
            actions=data["actions"],
            metadata=meta,
        )


@dataclass
class OracleEpisode:
    """The complete oracle record retained solely by the benchmark evaluator."""

    episode_id: str
    seed: int
    split: SplitType
    timestamps: np.ndarray  # Shape: [T]
    observations: np.ndarray  # Shape: [T, 8]
    observation_mask: np.ndarray  # Shape: [T, 8]
    actions: np.ndarray  # Shape: [T, 4]
    ground_truth_states: np.ndarray  # Shape: [T, 12] (8 observable + 4 latent)
    exogenous_noise: np.ndarray  # Shape: [T, 12] (Realized innovation sequence)
    failure_latched: bool = False
    failure_mode: Optional[str] = None
    failure_timestamp: Optional[int] = None
    oracle_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def to_learner_episode(self) -> LearnerEpisode:
        """Strip all ground truth state and noise, producing an isolated LearnerEpisode."""
        # Sanitize metadata
        sanitized_meta = {
            "policy": self.oracle_metadata.get("policy", "Unknown"),
            "length": self.length,
            "regime": self.oracle_metadata.get("regime", "nominal"),
            "missingness_type": self.oracle_metadata.get("missingness_type", "none"),
        }
        return LearnerEpisode(
            episode_id=self.episode_id,
            split=self.split,
            timestamps=np.copy(self.timestamps),
            observations=np.copy(self.observations),
            observation_mask=np.copy(self.observation_mask),
            actions=np.copy(self.actions),
            metadata=sanitized_meta,
        )

    def save_npz(self, file_path: str | Path) -> None:
        """Serialize oracle episode to .npz archive."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            episode_id=np.array(self.episode_id),
            seed=np.array(self.seed),
            split=np.array(self.split.value if isinstance(self.split, SplitType) else str(self.split)),
            timestamps=self.timestamps,
            observations=self.observations,
            observation_mask=self.observation_mask,
            actions=self.actions,
            ground_truth_states=self.ground_truth_states,
            exogenous_noise=self.exogenous_noise,
            failure_latched=np.array(self.failure_latched),
            failure_mode=np.array(self.failure_mode if self.failure_mode is not None else ""),
            failure_timestamp=np.array(self.failure_timestamp if self.failure_timestamp is not None else -1),
            oracle_metadata=np.array(self.oracle_metadata, dtype=object),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> OracleEpisode:
        """Load oracle episode from .npz archive."""
        data = np.load(file_path, allow_pickle=True)
        fail_mode_raw = str(data["failure_mode"])
        fail_mode = fail_mode_raw if fail_mode_raw != "" else None
        fail_ts_raw = int(data["failure_timestamp"])
        fail_ts = fail_ts_raw if fail_ts_raw >= 0 else None
        meta_arr = data["oracle_metadata"]
        meta = meta_arr.item() if meta_arr.ndim == 0 else dict(meta_arr)

        return cls(
            episode_id=str(data["episode_id"]),
            seed=int(data["seed"]),
            split=SplitType(str(data["split"])),
            timestamps=data["timestamps"],
            observations=data["observations"],
            observation_mask=data["observation_mask"],
            actions=data["actions"],
            ground_truth_states=data["ground_truth_states"],
            exogenous_noise=data["exogenous_noise"],
            failure_latched=bool(data["failure_latched"]),
            failure_mode=fail_mode,
            failure_timestamp=fail_ts,
            oracle_metadata=meta,
        )
