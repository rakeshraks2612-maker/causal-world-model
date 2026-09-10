"""World Model Input Specifications and Validation Contracts.

Enforces:
- Observation dimension: exactly 8 observable sensors
- Action dimension: exactly 4 control actions
- Mask dimension: exactly 8 boolean/binary indicators
- Absolute prohibition of oracle/latent variables in input tensors
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch import Tensor

from prism.dataset.schema import LearnerEpisode, FORBIDDEN_LEARNER_KEYS
from prism.dataset.validators import DatasetValidationError


@dataclass
class ModelInputs:
    """Batch input tensor container for World Model operations."""

    observations: Tensor        # Shape: [B, T, 8] (float32)
    observation_mask: Tensor    # Shape: [B, T, 8] (bool or float32: 1.0 = valid, 0.0 = masked)
    actions: Tensor             # Shape: [B, T, 4] (float32)
    timestamps: Optional[Tensor] = None  # Shape: [B, T] (float32)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and dimensions."""
        if self.observations.ndim != 3:
            raise ValueError(f"observations must be 3D [B, T, 8], got shape {self.observations.shape}")
        if self.observations.shape[-1] != 8:
            raise ValueError(f"observations last dimension must be 8, got {self.observations.shape[-1]}")

        if self.observation_mask.ndim != 3:
            raise ValueError(f"observation_mask must be 3D [B, T, 8], got shape {self.observation_mask.shape}")
        if self.observation_mask.shape != self.observations.shape:
            raise ValueError(f"observation_mask shape {self.observation_mask.shape} != observations shape {self.observations.shape}")

        if self.actions.ndim != 3:
            raise ValueError(f"actions must be 3D [B, T, 4], got shape {self.actions.shape}")
        if self.actions.shape[-1] != 4:
            raise ValueError(f"actions last dimension must be 4, got {self.actions.shape[-1]}")
        if self.actions.shape[:2] != self.observations.shape[:2]:
            raise ValueError(f"actions [B, T] shape {self.actions.shape[:2]} != observations [B, T] shape {self.observations.shape[:2]}")

        for k in self.metadata:
            if k in FORBIDDEN_LEARNER_KEYS:
                raise DatasetValidationError(f"CRITICAL LEAKAGE: Forbidden latent key '{k}' found in ModelInputs metadata!")

    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]

    @property
    def seq_len(self) -> int:
        return self.observations.shape[1]

    @classmethod
    def from_learner_episode(
        cls,
        episode: LearnerEpisode,
        device: Optional[torch.device] = None,
    ) -> ModelInputs:
        """Convert a single LearnerEpisode into a batch-of-1 ModelInputs tensor."""
        # Replace NaNs in masked observations with zeros for safe neural input
        obs_clean = np.nan_to_num(episode.observations, nan=0.0)
        
        obs_t = torch.tensor(obs_clean, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(episode.observation_mask, dtype=torch.float32).unsqueeze(0)
        act_t = torch.tensor(episode.actions, dtype=torch.float32).unsqueeze(0)
        ts_t = torch.tensor(episode.timestamps, dtype=torch.float32).unsqueeze(0)

        if device is not None:
            obs_t = obs_t.to(device)
            mask_t = mask_t.to(device)
            act_t = act_t.to(device)
            ts_t = ts_t.to(device)

        return cls(
            observations=obs_t,
            observation_mask=mask_t,
            actions=act_t,
            timestamps=ts_t,
            metadata=episode.metadata,
        )

    @classmethod
    def from_batch(
        cls,
        episodes: List[LearnerEpisode],
        device: Optional[torch.device] = None,
    ) -> ModelInputs:
        """Collate multiple LearnerEpisodes of identical length into a single ModelInputs batch."""
        if not episodes:
            raise ValueError("Cannot create ModelInputs from empty episode list")

        b_obs = [np.nan_to_num(ep.observations, nan=0.0) for ep in episodes]
        b_mask = [ep.observation_mask for ep in episodes]
        b_act = [ep.actions for ep in episodes]
        b_ts = [ep.timestamps for ep in episodes]

        obs_t = torch.tensor(np.stack(b_obs, axis=0), dtype=torch.float32)
        mask_t = torch.tensor(np.stack(b_mask, axis=0), dtype=torch.float32)
        act_t = torch.tensor(np.stack(b_act, axis=0), dtype=torch.float32)
        ts_t = torch.tensor(np.stack(b_ts, axis=0), dtype=torch.float32)

        if device is not None:
            obs_t = obs_t.to(device)
            mask_t = mask_t.to(device)
            act_t = act_t.to(device)
            ts_t = ts_t.to(device)

        return cls(
            observations=obs_t,
            observation_mask=mask_t,
            actions=act_t,
            timestamps=ts_t,
        )
