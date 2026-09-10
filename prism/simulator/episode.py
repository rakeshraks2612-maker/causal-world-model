"""Episode Data Structure, Recording, and Serialization for THC-SCM System.

Stores complete standardized episode trajectories containing:
- Timestamps
- Noisy partial observations
- Control actions
- Ground-truth 12-dimensional physical states
- Realized exogenous noise innovations
- Latched failure diagnostics & metadata
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from prism.simulator.state import OBSERVABLE_VARIABLES, ALL_STATE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES
from prism.simulator.noise import NOISE_VARIABLES


@dataclass
class Episode:
    """Standardized recorded episode trajectory."""

    episode_id: str
    seed: int
    timestamps: np.ndarray  # Shape: [T]
    observations: np.ndarray  # Shape: [T, 8]
    actions: np.ndarray  # Shape: [T, 4]
    ground_truth_states: np.ndarray  # Shape: [T, 12]
    exogenous_noise: np.ndarray  # Shape: [T, 12]
    failure_latched: bool = False
    failure_mode: Optional[str] = None
    failure_timestamp: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the number of time steps in this episode."""
        return int(len(self.timestamps))

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to serializable dictionary."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "length": self.length,
            "failure_latched": self.failure_latched,
            "failure_mode": self.failure_mode,
            "failure_timestamp": self.failure_timestamp,
            "metadata": self.metadata,
        }

    def save_npz(self, file_path: str | Path) -> None:
        """Save episode arrays and metadata to a compressed .npz archive."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            observations=self.observations,
            actions=self.actions,
            ground_truth_states=self.ground_truth_states,
            exogenous_noise=self.exogenous_noise,
            episode_id=np.array(self.episode_id),
            seed=np.array(self.seed),
            failure_latched=np.array(self.failure_latched),
            failure_mode=np.array(self.failure_mode if self.failure_mode is not None else ""),
            failure_timestamp=np.array(self.failure_timestamp if self.failure_timestamp is not None else -1),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> Episode:
        """Load episode from a compressed .npz archive."""
        data = np.load(file_path, allow_pickle=True)
        fail_mode_raw = str(data["failure_mode"])
        fail_mode = fail_mode_raw if fail_mode_raw != "" else None
        fail_ts_raw = int(data["failure_timestamp"])
        fail_ts = fail_ts_raw if fail_ts_raw >= 0 else None

        return cls(
            episode_id=str(data["episode_id"]),
            seed=int(data["seed"]),
            timestamps=data["timestamps"],
            observations=data["observations"],
            actions=data["actions"],
            ground_truth_states=data["ground_truth_states"],
            exogenous_noise=data["exogenous_noise"],
            failure_latched=bool(data["failure_latched"]),
            failure_mode=fail_mode,
            failure_timestamp=fail_ts,
        )
