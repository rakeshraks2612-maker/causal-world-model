"""Execution Script for Task 3.3B OOD Uncertainty Diagnostics.

Investigates why single-model predictive sigma does not reliably increase under OOD conditions.
Calculates latent distribution shifts, Mahalanobis distances, particle disagreement, and correlations.
Outputs artifacts/rollout/uncertainty_diagnostic_report.json.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.seed import set_seed
from prism.training.normalization import ObservationNormalizer
from prism.training.checkpointing import load_checkpoint
from prism.evaluation.uncertainty_diagnostics import compute_uncertainty_diagnostics


def run_ood_uncertainty_diagnostics(
    checkpoint_dir: str = "artifacts/baseline_002",
    output_file: str = "artifacts/rollout/uncertainty_diagnostic_report.json",
) -> None:
    set_seed(42)
    ckpt_path = Path(checkpoint_dir)
    out_file = Path(output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 115)
    print("TASK 3.3B — OOD UNCERTAINTY & LATENT SHIFT DIAGNOSTIC SUITE")
    print("=" * 115)

    # 1. Load Frozen baseline_002 Checkpoint and Normalizer
    norm_path = ckpt_path / "normalization.yaml"
    normalizer = ObservationNormalizer.load_yaml(norm_path)

    cfg_path = ckpt_path / "config.yaml"
    wm_config = WorldModelConfig.from_yaml(cfg_path)
    model = CausalWorldModel(wm_config)
    load_checkpoint(ckpt_path / "best.pt", model)
    model.eval()

    print(f"Loaded frozen baseline_002 checkpoint from {ckpt_path / 'best.pt'}")

    # 2. Load Train, Test, and OOD Episodes
    train_eps = [LearnerEpisode.load_npz(f) for f in sorted(Path("data/pilot/learner/train").glob("*.npz"))]
    test_eps = [LearnerEpisode.load_npz(f) for f in sorted(Path("data/pilot/learner/test").glob("*.npz"))]

    ood_root = Path("data/pilot/learner/ood")
    ood_regimes: dict[str, list[LearnerEpisode]] = {}
    for d in sorted(ood_root.iterdir()):
        if d.is_dir():
            eps = [LearnerEpisode.load_npz(f) for f in sorted(d.glob("*.npz"))]
            if eps:
                ood_regimes[d.name] = eps

    print(f"Loaded: {len(train_eps)} Train, {len(test_eps)} Test, and {len(ood_regimes)} OOD regimes.")

    # 3. Compute Comprehensive Diagnostics
    diag_report = compute_uncertainty_diagnostics(
        model=model,
        train_episodes=train_eps,
        test_episodes=test_eps,
        ood_regimes=ood_regimes,
        normalizer=normalizer,
        num_particles=50,
        context_length=40,
        stride=5,
        horizon=40,
    )

    # 4. Save JSON Report
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diag_report.to_dict(), f, indent=2)

    print(f"\nSaved uncertainty diagnostic report to {out_file}")

    # 5. Format and Print Diagnostic Table
    print("\n" + "=" * 125)
    print("TASK 3.3B — OOD UNCERTAINTY & LATENT DIAGNOSTIC TABLE")
    print("=" * 125)
    header = (
        f"{'Regime':<24} | {'Forecast Error':<14} | {'Latent Distance':<16} | "
        f"{'Latent σ':<10} | {'Particle σ':<10} | {'Error–Latent Dist':<18} | {'Error–Particle σ':<16}"
    )
    print(header)
    print("-" * 125)

    for reg_name, diag in diag_report.regime_diagnostics.items():
        err_str = f"{diag.forecast_error_h40:.4f}"
        dist_str = f"{diag.latent_mahalanobis_distance:.3f} (d_M)"
        lat_sig_str = f"{diag.latent_std_norm:.3f}"
        part_sig_str = f"{diag.particle_obs_sigma_h40:.3f}"
        corr_d_str = f"{diag.error_latent_distance_corr:+.3f}"
        corr_s_str = f"{diag.error_particle_sigma_corr:+.3f}"

        clean_reg = reg_name.replace("ood_", "").replace("_", " ").title()
        print(f"{clean_reg:<24} | {err_str:>14} | {dist_str:>16} | {lat_sig_str:>10} | {part_sig_str:>10} | {corr_d_str:>18} | {corr_s_str:>16}")

    print("=" * 125)
    print(f"\nDIAGNOSTIC VERDICT: {diag_report.diagnosis_verdict}")
    print(f"RATIONALE: {diag_report.diagnosis_rationale}\n")


if __name__ == "__main__":
    run_ood_uncertainty_diagnostics()
