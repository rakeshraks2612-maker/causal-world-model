"""Master Evaluation and Experiment Runner for Task 3.3.

Executes:
1. Multi-Step Open-Loop & Teacher-Forced Forecasting across horizons h in {1, 5, 10, 20, 40} for Persistence, MLP Dynamics, and PRISM.
2. Monte Carlo 50-Particle Ensemble Rollouts, Predictive Uncertainty, 90% Confidence Intervals, and Empirical Coverage.
3. OOD Generalization Evaluation across OOD-1 through OOD-5 regimes without retraining.
4. Error-vs-Uncertainty Correlation Analysis.
5. Causal Action Branching for Valve and Throttle Sweeps.
6. Generates JSON artifacts and high-resolution diagnostic plots in artifacts/rollout/.
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
import matplotlib.pyplot as plt

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.seed import set_seed
from prism.training.normalization import ObservationNormalizer
from prism.training.checkpointing import load_checkpoint
from prism.training.metrics import MLPDynamicsBaseline
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader

from prism.evaluation.forecasting import (
    evaluate_prism_multistep,
    evaluate_mlp_multistep,
    evaluate_persistence_multistep,
)
from prism.evaluation.uncertainty_metrics import evaluate_monte_carlo_uncertainty
from prism.evaluation.ood import evaluate_ood_regimes
from prism.evaluation.action_branching import evaluate_action_branching


def run_task_3_3_experiments(
    checkpoint_dir: str = "artifacts/baseline_002",
    output_dir: str = "artifacts/rollout",
) -> None:
    out_path = Path(output_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint_dir)
    set_seed(42)

    print("=" * 115)
    print("TASK 3.3 — MULTI-STEP LATENT ROLLOUT & UNCERTAINTY CALIBRATION")
    print("=" * 115)

    # 1. Load Frozen baseline_002 Model and Normalizer
    norm_path = ckpt_path / "normalization.yaml"
    normalizer = ObservationNormalizer.load_yaml(norm_path)

    cfg_path = ckpt_path / "config.yaml"
    wm_config = WorldModelConfig.from_yaml(cfg_path)
    prism_model = CausalWorldModel(wm_config)
    load_checkpoint(ckpt_path / "best.pt", prism_model)
    prism_model.eval()

    print(f"Loaded frozen baseline_002 checkpoint from {ckpt_path / 'best.pt'}")

    # 2. Load Train, Validation, Test, and OOD Episodes
    train_eps = [LearnerEpisode.load_npz(f) for f in sorted(Path("data/pilot/learner/train").glob("*.npz"))]
    test_eps = [LearnerEpisode.load_npz(f) for f in sorted(Path("data/pilot/learner/test").glob("*.npz"))]
    print(f"Loaded {len(train_eps)} Train episodes, {len(test_eps)} untouched Test episodes.")

    # 3. Train Supervised MLP Dynamics Baseline
    print("Training Supervised MLP Dynamics Baseline for multi-step comparison...")
    mlp_model = MLPDynamicsBaseline(obs_dim=8, action_dim=4, hidden_dim=64)
    opt_mlp = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_ds = PrismWindowedDataset(train_eps, context_length=40, prediction_length=1, stride=2, normalizer=normalizer)
    train_loader = create_dataloader(train_ds, batch_size=32, shuffle=True)

    mlp_model.train()
    for _ in range(25):
        for batch in train_loader:
            inputs = batch["inputs"]
            curr_obs = inputs.observations[:, :-1].reshape(-1, 8)
            curr_act = inputs.actions[:, :-1].reshape(-1, 4)
            next_target = inputs.observations[:, 1:].reshape(-1, 8)
            step_mask = inputs.observation_mask[:, 1:].reshape(-1, 8)

            opt_mlp.zero_grad()
            pred_next = mlp_model(curr_obs, curr_act)
            loss = torch.mean(((pred_next - next_target) ** 2) * step_mask)
            loss.backward()
            opt_mlp.step()
    mlp_model.eval()

    # 4. Multi-Step Forecasting Benchmark (h in {1, 5, 10, 20, 40})
    horizons = [1, 5, 10, 20, 40]
    print("\nExecuting Multi-Step Open-Loop and Teacher-Forced Forecasting on Test split...")

    pers_ol = evaluate_persistence_multistep(test_eps, normalizer, horizons=horizons, context_length=40, stride=5)
    mlp_ol = evaluate_mlp_multistep(mlp_model, test_eps, normalizer, horizons=horizons, context_length=40, stride=5)
    prism_ol = evaluate_prism_multistep(prism_model, test_eps, normalizer, horizons=horizons, context_length=40, stride=5, mode="open_loop", deterministic=True)
    prism_tf = evaluate_prism_multistep(prism_model, test_eps, normalizer, horizons=horizons, context_length=40, stride=5, mode="teacher_forced", deterministic=True)

    multistep_data = {
        "persistence_open_loop": pers_ol.to_dict(),
        "mlp_dynamics_open_loop": mlp_ol.to_dict(),
        "prism_open_loop": prism_ol.to_dict(),
        "prism_teacher_forced": prism_tf.to_dict(),
    }
    with open(out_path / "multistep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(multistep_data, f, indent=2)

    # 5. Monte Carlo Uncertainty Evaluation (N=50 particles)
    print("\nExecuting Monte Carlo 50-Particle Uncertainty & Calibration Evaluation...")
    unc_report = evaluate_monte_carlo_uncertainty(
        model=prism_model,
        episodes=test_eps,
        normalizer=normalizer,
        horizons=horizons,
        num_particles=50,
        context_length=40,
        stride=5,
    )
    with open(out_path / "uncertainty_metrics.json", "w", encoding="utf-8") as f:
        json.dump(unc_report.to_dict(), f, indent=2)

    # 6. OOD Generalization Evaluation
    print("\nExecuting OOD Generalization Evaluation across OOD-1..5 Regimes...")
    ood_report = evaluate_ood_regimes(
        model=prism_model,
        test_episodes=test_eps,
        ood_root_dir="data/pilot/learner/ood",
        normalizer=normalizer,
        horizons=horizons,
        num_particles=50,
    )
    with open(out_path / "ood_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ood_report.to_dict(), f, indent=2)

    # 7. Action Branching Evaluation
    print("\nExecuting Action Branching & Causal Simulation Experiments...")
    ref_episode = test_eps[0]
    branch_report = evaluate_action_branching(
        model=prism_model,
        reference_episode=ref_episode,
        normalizer=normalizer,
        horizon=40,
        context_length=40,
    )
    with open(out_path / "action_branches.json", "w", encoding="utf-8") as f:
        json.dump(branch_report.to_dict(), f, indent=2)

    # 8. Render Visual Artifacts / Plots
    print("\nRendering high-resolution diagnostic plots...")

    # Plot 1: Error vs Horizon (Persistence vs MLP vs PRISM Open-Loop vs PRISM Teacher-Forced)
    plt.figure(figsize=(10, 6), dpi=150)
    h_vals = horizons
    p_maes = [pers_ol.by_horizon[h].train_normalized_aggregate_mae for h in h_vals]
    mlp_maes = [mlp_ol.by_horizon[h].train_normalized_aggregate_mae for h in h_vals]
    prism_maes = [prism_ol.by_horizon[h].train_normalized_aggregate_mae for h in h_vals]
    tf_maes = [prism_tf.by_horizon[h].train_normalized_aggregate_mae for h in h_vals]

    plt.plot(h_vals, p_maes, "o--", color="#e74c3c", label="Persistence (Open-Loop)", linewidth=2)
    plt.plot(h_vals, mlp_maes, "s--", color="#e67e22", label="MLP Dynamics (Open-Loop)", linewidth=2)
    plt.plot(h_vals, prism_maes, "o-", color="#2980b9", label="PRISM World Model (Open-Loop)", linewidth=2.5)
    plt.plot(h_vals, tf_maes, "^:", color="#27ae60", label="PRISM (Teacher-Forced)", linewidth=1.8)

    plt.xlabel("Forecasting Horizon h (steps)", fontsize=12)
    plt.ylabel("Train-Normalized Aggregate MAE", fontsize=12)
    plt.title("Multi-Step Error Growth vs Forecasting Horizon", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(plots_path / "error_vs_horizon.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Thermal Rollout (T_core and T_cool across horizons)
    plt.figure(figsize=(12, 5), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(h_vals, [pers_ol.by_horizon[h].per_channel["T_core"].mae for h in h_vals], "o--", color="#e74c3c", label="Persistence")
    plt.plot(h_vals, [mlp_ol.by_horizon[h].per_channel["T_core"].mae for h in h_vals], "s--", color="#e67e22", label="MLP Dynamics")
    plt.plot(h_vals, [prism_ol.by_horizon[h].per_channel["T_core"].mae for h in h_vals], "o-", color="#2980b9", label="PRISM Open-Loop")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("T_core MAE (°C)")
    plt.title("Core Temperature (T_core) Error Growth")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h_vals, [pers_ol.by_horizon[h].per_channel["T_cool"].mae for h in h_vals], "o--", color="#e74c3c", label="Persistence")
    plt.plot(h_vals, [mlp_ol.by_horizon[h].per_channel["T_cool"].mae for h in h_vals], "s--", color="#e67e22", label="MLP Dynamics")
    plt.plot(h_vals, [prism_ol.by_horizon[h].per_channel["T_cool"].mae for h in h_vals], "o-", color="#2980b9", label="PRISM Open-Loop")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("T_cool MAE (°C)")
    plt.title("Coolant Temperature (T_cool) Error Growth")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_path / "thermal_rollout.png", bbox_inches="tight")
    plt.close()

    # Plot 3: Uncertainty & Interval Width vs Horizon
    plt.figure(figsize=(10, 6), dpi=150)
    widths_tcore = [unc_report.by_horizon[h].per_channel["T_core"].mean_interval_width_90 for h in h_vals]
    widths_tcool = [unc_report.by_horizon[h].per_channel["T_cool"].mean_interval_width_90 for h in h_vals]
    widths_psys = [unc_report.by_horizon[h].per_channel["P_sys"].mean_interval_width_90 * 10 for h in h_vals]

    plt.plot(h_vals, widths_tcore, "o-", color="#c0392b", label="T_core 90% CI Width (°C)", linewidth=2)
    plt.plot(h_vals, widths_tcool, "s-", color="#2980b9", label="T_cool 90% CI Width (°C)", linewidth=2)
    plt.plot(h_vals, widths_psys, "^-", color="#8e44ad", label="P_sys 90% CI Width (bar x10)", linewidth=2)

    plt.xlabel("Horizon h (steps)", fontsize=12)
    plt.ylabel("Mean 90% Prediction Interval Width", fontsize=12)
    plt.title("Predictive Interval Width Expansion with Horizon", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(plots_path / "uncertainty_vs_horizon.png", bbox_inches="tight")
    plt.close()

    # Plot 4: OOD Error vs Uncertainty Comparison
    plt.figure(figsize=(12, 6), dpi=150)
    reg_names = list(ood_report.regimes.keys())
    clean_names = [r.replace("ood_", "").replace("_", " ").title() for r in reg_names]
    maes_h40 = [ood_report.regimes[r].train_normalized_mae_h40 for r in reg_names]
    uncs_h40 = [ood_report.regimes[r].mean_uncertainty_std_h40 for r in reg_names]

    x = np.arange(len(reg_names))
    w = 0.35
    plt.bar(x - w/2, maes_h40, w, label="h=40 Normalized MAE (Error)", color="#e67e22")
    plt.bar(x + w/2, uncs_h40, w, label="h=40 Predictive Uncertainty (Std)", color="#3498db")

    plt.xticks(x, clean_names, rotation=25, ha="right", fontsize=10)
    plt.ylabel("Standardized Metric", fontsize=12)
    plt.title("In-Distribution vs OOD Regimes: Error and Predictive Uncertainty", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_path / "ood_uncertainty.png", bbox_inches="tight")
    plt.close()

    print(f"\nSaved all metrics and plots to {out_path}")


if __name__ == "__main__":
    run_task_3_3_experiments()
