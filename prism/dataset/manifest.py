"""Dataset Manifest Generator and Serialization.

Provides machine-readable dataset metadata for provenance, versioning, and reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class DatasetManifest:
    """Standardized metadata manifest for PRISM datasets."""

    dataset_version: str
    simulator_version: str
    generation_timestamp: str
    random_seed_policy: str
    total_episodes: int
    split_counts: Dict[str, int]
    regime_counts: Dict[str, int]
    schema_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        dataset_version: str = "1.0.0",
        simulator_version: str = "1.1.0",
        split_counts: Optional[Dict[str, int]] = None,
        regime_counts: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetManifest:
        splits = split_counts or {}
        regimes = regime_counts or {}
        total = sum(splits.values())
        return cls(
            dataset_version=dataset_version,
            simulator_version=simulator_version,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            random_seed_policy="deterministic_hierarchical_seed_sequence",
            total_episodes=total,
            split_counts=splits,
            regime_counts=regimes,
            schema_version="1.0.0",
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str | Path) -> DatasetManifest:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
