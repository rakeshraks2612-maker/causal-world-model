"""Training-Set-Only Normalization Subsystem.

Strict Invariants:
- Normalization statistics (mu_train, sigma_train) MUST be calculated using TRAIN split only.
- Validation, Test, OOD, Intervention, Counterfactual splits NEVER alter normalization statistics.
- Zero-variance channels are safely regularized (std < 1e-6 -> 1.0).
- Reversible transformation: denormalize(normalize(x)) == x.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import Tensor
import yaml

from prism.dataset.schema import LearnerEpisode
from prism.simulator.state import OBSERVABLE_VARIABLES


@dataclass
class NormalizationStats:
    """Statistical parameters per observable channel."""

    means: List[float]
    stds: List[float]
    channel_names: List[str] = field(default_factory=lambda: list(OBSERVABLE_VARIABLES))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_names": self.channel_names,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizationStats:
        return cls(
            means=[float(m) for m in data["means"]],
            stds=[float(s) for s in data["stds"]],
            channel_names=list(data.get("channel_names", OBSERVABLE_VARIABLES)),
        )


class ObservationNormalizer:
    """Normalizes the 8 observable sensor channels: x_norm = (x - mu) / sigma."""

    def __init__(self, stats: Optional[NormalizationStats] = None) -> None:
        self.stats = stats
        self._mean_tensor: Optional[Tensor] = None
        self._std_tensor: Optional[Tensor] = None

        if stats is not None:
            self._update_tensors()

    def _update_tensors(self) -> None:
        if self.stats is not None:
            self._mean_tensor = torch.tensor(self.stats.means, dtype=torch.float32)
            self._std_tensor = torch.tensor(self.stats.stds, dtype=torch.float32)

    @classmethod
    def fit(cls, train_episodes: List[LearnerEpisode]) -> ObservationNormalizer:
        """Compute mean and std strictly across valid unmasked training observations."""
        if not train_episodes:
            raise ValueError("Cannot fit normalizer on empty training episode list")

        # Collect unmasked measurements per channel
        all_obs = []
        all_masks = []

        for ep in train_episodes:
            all_obs.append(ep.observations)
            all_masks.append(ep.observation_mask)

        obs_arr = np.concatenate(all_obs, axis=0)      # [Total_T, 8]
        mask_arr = np.concatenate(all_masks, axis=0)    # [Total_T, 8]

        means: List[float] = []
        stds: List[float] = []

        for ch in range(8):
            valid_vals = obs_arr[mask_arr[:, ch] & ~np.isnan(obs_arr[:, ch]), ch]
            if len(valid_vals) > 0:
                mean_val = float(np.mean(valid_vals))
                std_val = float(np.std(valid_vals))
                # Safe clamp for near-zero variance
                if std_val < 1e-6:
                    std_val = 1.0
            else:
                mean_val = 0.0
                std_val = 1.0

            means.append(mean_val)
            stds.append(std_val)

        stats = NormalizationStats(means=means, stds=stds)
        return cls(stats=stats)

    def normalize(self, x: Tensor | np.ndarray) -> Tensor:
        """Apply z-score normalization: (x - mu) / sigma."""
        if self._mean_tensor is None or self._std_tensor is None:
            raise RuntimeError("Normalizer has not been fitted or loaded with statistics")

        is_numpy = isinstance(x, np.ndarray)
        t = torch.tensor(x, dtype=torch.float32) if is_numpy else x
        
        mean = self._mean_tensor.to(t.device)
        std = self._std_tensor.to(t.device)

        norm_t = (t - mean) / std
        return norm_t

    def denormalize(self, x_norm: Tensor | np.ndarray) -> Tensor:
        """Invert z-score normalization: x = x_norm * sigma + mu."""
        if self._mean_tensor is None or self._std_tensor is None:
            raise RuntimeError("Normalizer has not been fitted or loaded with statistics")

        is_numpy = isinstance(x_norm, np.ndarray)
        t = torch.tensor(x_norm, dtype=torch.float32) if is_numpy else x_norm

        mean = self._mean_tensor.to(t.device)
        std = self._std_tensor.to(t.device)

        denorm_t = t * std + mean
        return denorm_t

    def save_yaml(self, file_path: str | Path) -> None:
        """Save normalization statistics to YAML file."""
        if self.stats is None:
            raise RuntimeError("Cannot save uninitialized normalizer")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.stats.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, file_path: str | Path) -> ObservationNormalizer:
        """Load normalization statistics from YAML file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        stats = NormalizationStats.from_dict(data)
        return cls(stats=stats)
