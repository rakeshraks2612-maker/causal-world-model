"""Windowed Dataset Loader for PRISM Sequential World Model Training.

Enforces:
- Sequential Windowing without crossing episode boundaries
- Train-set only normalization
- Preservation of native missing telemetry masks with optional training-time dropout masking (M_training <= M_original)
- Zero oracle / latent variable leakage
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from prism.dataset.schema import LearnerEpisode, SplitType, FORBIDDEN_LEARNER_KEYS
from prism.training.normalization import ObservationNormalizer


class PrismWindowedDataset(Dataset):
    """PyTorch Dataset generating sequential temporal windows from LearnerEpisodes."""

    def __init__(
        self,
        episodes: List[LearnerEpisode],
        context_length: int = 40,
        prediction_length: int = 1,
        stride: int = 2,
        normalizer: Optional[ObservationNormalizer] = None,
        mask_dropout_prob: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.episodes = episodes
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = context_length + prediction_length
        self.stride = stride
        self.normalizer = normalizer
        self.mask_dropout_prob = mask_dropout_prob
        self.rng = np.random.default_rng(rng_seed)

        # Build index of valid windows: (episode_idx, start_timestep)
        self.windows: List[Tuple[int, int]] = []
        self._index_windows()

    def _index_windows(self) -> None:
        """Construct window indices ensuring ZERO boundary crossing between episodes."""
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep.timestamps)
            if ep_len >= self.window_length:
                max_start = ep_len - self.window_length
                for start_t in range(0, max_start + 1, self.stride):
                    self.windows.append((ep_idx, start_t))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ep_idx, start_t = self.windows[idx]
        ep = self.episodes[ep_idx]
        end_t = start_t + self.window_length

        # 1. Extract raw slice
        raw_obs = np.copy(ep.observations[start_t:end_t])       # [L, 8]
        raw_mask = np.copy(ep.observation_mask[start_t:end_t])   # [L, 8]
        actions = np.copy(ep.actions[start_t:end_t])             # [L, 4]

        # 2. Training-time mask dropout: M_training <= M_original
        training_mask = np.copy(raw_mask)
        if self.mask_dropout_prob > 0.0:
            dropout_mask = self.rng.binomial(1, 1.0 - self.mask_dropout_prob, size=raw_mask.shape).astype(bool)
            training_mask = training_mask & dropout_mask

        # 3. Clean NaNs in observations (for masked sensors)
        clean_obs = np.nan_to_num(raw_obs, nan=0.0)

        # 4. Apply Normalization if provided
        if self.normalizer is not None:
            norm_obs_t = self.normalizer.normalize(torch.tensor(clean_obs, dtype=torch.float32))
            norm_obs = norm_obs_t.numpy()
        else:
            norm_obs = clean_obs

        # Set masked values to 0.0 in input tensor
        norm_obs = norm_obs * training_mask.astype(np.float32)

        return {
            "observations": torch.tensor(norm_obs, dtype=torch.float32),
            "observation_mask": torch.tensor(training_mask, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "raw_observations": torch.tensor(clean_obs, dtype=torch.float32),
            "raw_mask": torch.tensor(raw_mask, dtype=torch.float32),
            "episode_id": ep.episode_id,
            "start_timestep": start_t,
        }

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        context_length: int = 40,
        prediction_length: int = 1,
        stride: int = 2,
        normalizer: Optional[ObservationNormalizer] = None,
        mask_dropout_prob: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> PrismWindowedDataset:
        """Load all .npz files from a learner split directory into a windowed dataset."""
        dir_path = Path(data_dir)
        files = sorted(dir_path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz learner files found in {data_dir}")

        episodes = [LearnerEpisode.load_npz(f) for f in files]
        return cls(
            episodes=episodes,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            normalizer=normalizer,
            mask_dropout_prob=mask_dropout_prob,
            rng_seed=rng_seed,
        )
