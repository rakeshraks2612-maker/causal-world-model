"""Split Management and Deterministic Seed Allocation for PRISM Datasets.

Assigns independent, reproducible seed ranges per split partition to guarantee:
- Absolute episode ID disjointness
- Zero RNG overlap between Train, Validation, and Test
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from prism.dataset.schema import SplitType


# Pre-allocated disjoint seed ranges per partition
SEED_RANGES: Dict[SplitType, Tuple[int, int]] = {
    SplitType.TRAIN: (100_000, 199_999),
    SplitType.VAL: (200_000, 249_999),
    SplitType.TEST: (250_000, 299_999),
    SplitType.OOD: (300_000, 349_999),
    SplitType.INTERVENTION: (350_000, 399_999),
    SplitType.COUNTERFACTUAL: (400_000, 449_999),
    SplitType.STRESS: (450_000, 499_999),
}


def get_partition_seed(split: SplitType, index: int) -> int:
    """Compute a deterministic seed for an episode within a partition."""
    start_seed, end_seed = SEED_RANGES[split]
    seed = start_seed + index
    if seed > end_seed:
        raise ValueError(f"Index {index} exceeds preallocated seed range for split {split.value}")
    return seed


def generate_episode_id(split: SplitType, index: int) -> str:
    """Generate a clean, standardized episode ID."""
    return f"ep_{split.value}_{index:05d}"
