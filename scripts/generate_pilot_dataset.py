"""Pilot Dataset Generator and Distribution Auditor for Tasks 2.1-2.4.

Generates:
- 20 Train episodes
- 5 Validation episodes
- 5 Test episodes
in both data/pilot/learner/ and data/pilot/oracle/ formats, validating all invariants and printing summary statistics.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_split_dataset
from prism.dataset.manifest import DatasetManifest
from prism.dataset.validators import validate_split_disjointness
from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES


def run_pilot_generation(output_dir: str = "data/pilot") -> None:
    out_path = Path(output_dir)
    print(f"Generating pilot dataset to {out_path}...")

    regime_distribution = {
        "nominal": 0.70,
        "moderate_load": 0.15,
        "moderate_ambient": 0.10,
        "moderate_wear": 0.05,
    }

    # Generate Train (20), Val (5), Test (5)
    train_oracle, train_learner = generate_split_dataset(
        SplitType.TRAIN, count=20, regime_distribution=regime_distribution, output_dir=out_path, length=120
    )
    val_oracle, val_learner = generate_split_dataset(
        SplitType.VAL, count=5, regime_distribution=regime_distribution, output_dir=out_path, length=120
    )
    test_oracle, test_learner = generate_split_dataset(
        SplitType.TEST, count=5, regime_distribution=regime_distribution, output_dir=out_path, length=120
    )

    # Validate split disjointness
    validate_split_disjointness(train_learner, val_learner, test_learner)
    print("✓ Episode split disjointness verified (Train ∩ Val ∩ Test = ∅)")

    # Count regimes
    regime_counts = {}
    for ep in train_oracle + val_oracle + test_oracle:
        r = ep.oracle_metadata.get("regime", "unknown")
        regime_counts[r] = regime_counts.get(r, 0) + 1

    # Save manifest
    manifest = DatasetManifest.create(
        dataset_version="0.1.0-pilot",
        simulator_version="1.1.0",
        split_counts={"train": len(train_learner), "validation": len(val_learner), "test": len(test_learner)},
        regime_counts=regime_counts,
        metadata={"length": 120, "pilot": True},
    )
    manifest.save_json(out_path / "manifest.json")
    print(f"✓ Saved dataset manifest to {out_path / 'manifest.json'}")

    # Compute and display distribution statistics across Train
    all_train_obs = np.vstack([ep.observations for ep in train_learner])
    print("\n--- Training Distribution Observable Statistics (N = 20 episodes, 2400 time steps) ---")
    for idx, var_name in enumerate(OBSERVABLE_VARIABLES):
        col = all_train_obs[:, idx]
        print(f"  {var_name:12s} | Mean: {np.nanmean(col):6.2f} | Std: {np.nanstd(col):5.2f} | Min: {np.nanmin(col):6.2f} | Max: {np.nanmax(col):6.2f}")

    all_train_states = np.vstack([ep.ground_truth_states for ep in train_oracle])
    print("\n--- Training Distribution Latent Variables (Oracle Audit Only) ---")
    for idx, var_name in enumerate(LATENT_VARIABLES, start=8):
        col = all_train_states[:, idx]
        print(f"  {var_name:12s} | Mean: {np.mean(col):6.2f} | Std: {np.std(col):5.2f} | Min: {np.min(col):6.2f} | Max: {np.max(col):6.2f}")

    print("\nPilot dataset successfully generated and verified.")


if __name__ == "__main__":
    run_pilot_generation()
