"""Batch Collation and DataLoader Utilities."""

from __future__ import annotations
from typing import Dict, List, Any
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from prism.world_model.inputs import ModelInputs
from prism.training.dataset import PrismWindowedDataset


def collate_windowed_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dataset window samples into batch tensors and ModelInputs."""
    obs_list = [item["observations"] for item in batch_items]
    mask_list = [item["observation_mask"] for item in batch_items]
    act_list = [item["actions"] for item in batch_items]
    raw_obs_list = [item["raw_observations"] for item in batch_items]
    raw_mask_list = [item["raw_mask"] for item in batch_items]

    obs_b = torch.stack(obs_list, dim=0)          # [B, L, 8]
    mask_b = torch.stack(mask_list, dim=0)        # [B, L, 8]
    act_b = torch.stack(act_list, dim=0)          # [B, L, 4]
    raw_obs_b = torch.stack(raw_obs_list, dim=0)  # [B, L, 8]
    raw_mask_b = torch.stack(raw_mask_list, dim=0) # [B, L, 8]

    model_inputs = ModelInputs(
        observations=obs_b,
        observation_mask=mask_b,
        actions=act_b,
    )

    return {
        "inputs": model_inputs,
        "raw_observations": raw_obs_b,
        "raw_mask": raw_mask_b,
        "episode_ids": [item["episode_id"] for item in batch_items],
        "start_timesteps": [item["start_timestep"] for item in batch_items],
    }


def create_dataloader(
    dataset: PrismWindowedDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create standard DataLoader for PRISM training and evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_windowed_batch,
    )
