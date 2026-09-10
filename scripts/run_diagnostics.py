"""Master Diagnostic Runner for Task 3.2A.

Executes comprehensive scientific diagnostics across:
1. Reconstruction vs Transition Prediction on baseline_001 (Deterministic & Stochastic)
2. KL Weight Ablation: beta_KL in {0, 0.001, 0.01}
3. Latent Capacity Ablation: latent_dim in {32, 64, 128}
4. Deterministic vs Stochastic Latent Mode
5. Context Window Length: context_length in {10, 20, 40, 80}
6. Hidden-State Linear Probes (T_amb, W_wear, Q_internal, xi_leak)
7. Action-Sensitivity and Directional Causal Audits
8. Physical & Train-Normalized Aggregate Metric Benchmarks

Strictly preserves artifacts/baseline/ and outputs all diagnostic artifacts to artifacts/diagnosis/.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn
import numpy as np

from prism.dataset.schema import LearnerEpisode, OracleEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.seed import set_seed
from prism.training.normalization import ObservationNormalizer
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader
from prism.training.trainer import WorldModelTrainer
from prism.training.checkpointing import load_checkpoint
from prism.training.metrics import PersistenceBaseline, MLPDynamicsBaseline
from prism.diagnosis.domain_metrics import (
    ComprehensiveEvaluationReport,
    compute_comprehensive_metrics,
    DOMAINS,
    PHYSICAL_UNITS,
)
from prism.diagnosis.reconstruction_vs_prediction import evaluate_reconstruction_vs_prediction
from prism.diagnosis.latent_probes import fit_and_evaluate_latent_probes, LatentProbeReport
from prism.diagnosis.action_sensitivity import evaluate_action_sensitivity, ActionSensitivityReport


def run_diagnostics(
    base_config_path: str = "configs/training_baseline.yaml",
    output_dir: str = "artifacts/diagnosis",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    seed = cfg_dict.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    print("=" * 115)
    print("TASK 3.2A — PRISM WORLD MODEL BASELINE DIAGNOSTIC SUITE")
    print("=" * 115)

    # 1. Load Datasets
    train_dir = Path(cfg_dict["dataset"]["train_dir"])
    val_dir = Path(cfg_dict["dataset"]["val_dir"])
    test_dir = Path(cfg_dict["dataset"]["test_dir"])

    train_eps = [LearnerEpisode.load_npz(f) for f in sorted(train_dir.glob("*.npz"))]
    val_eps = [LearnerEpisode.load_npz(f) for f in sorted(val_dir.glob("*.npz"))]
    test_eps = [LearnerEpisode.load_npz(f) for f in sorted(test_dir.glob("*.npz"))]

    oracle_train_dir = Path("data/pilot/oracle/train")
    oracle_test_dir = Path("data/pilot/oracle/test")
    oracle_train_eps = [OracleEpisode.load_npz(f) for f in sorted(oracle_train_dir.glob("*.npz"))]
    oracle_test_eps = [OracleEpisode.load_npz(f) for f in sorted(oracle_test_dir.glob("*.npz"))]

    # Load / Fit Train-only normalizer
    norm_path = Path("artifacts/baseline/normalization.yaml")
    if norm_path.exists():
        normalizer = ObservationNormalizer.load_yaml(norm_path)
    else:
        normalizer = ObservationNormalizer.fit(train_eps)

    print(f"Loaded: {len(train_eps)} Train, {len(val_eps)} Val, {len(test_eps)} Test learner episodes.")
    print(f"Loaded: {len(oracle_train_eps)} Train, {len(oracle_test_eps)} Test oracle episodes for hidden-state probing.")

    # 2. Evaluate Baseline A: Persistence and Baseline B: MLP Dynamics on Test Loader
    test_ds_std = PrismWindowedDataset(test_eps, context_length=40, prediction_length=1, stride=2, normalizer=normalizer)
    test_loader_std = create_dataloader(test_ds_std, batch_size=32, shuffle=False)

    train_ds_std = PrismWindowedDataset(train_eps, context_length=40, prediction_length=1, stride=2, normalizer=normalizer)
    train_loader_std = create_dataloader(train_ds_std, batch_size=32, shuffle=True)

    # --- Baseline A: Persistence ---
    pers_baseline = PersistenceBaseline()
    all_pers_preds = []
    all_pers_targets = []
    all_pers_masks = []
    with torch.no_grad():
        for batch in test_loader_std:
            raw_obs = batch["raw_observations"]
            raw_mask = batch["raw_mask"]
            pred = pers_baseline.predict_one_step(raw_obs)
            all_pers_preds.append(pred.reshape(-1, 8))
            all_pers_targets.append(raw_obs[:, 1:].reshape(-1, 8))
            all_pers_masks.append(raw_mask[:, 1:].reshape(-1, 8))

    pers_report = compute_comprehensive_metrics(
        predictions_phys=torch.cat(all_pers_preds, dim=0),
        targets_phys=torch.cat(all_pers_targets, dim=0),
        mask=torch.cat(all_pers_masks, dim=0),
        normalizer=normalizer,
        model_name="Persistence",
    )

    # --- Baseline B: MLP Dynamics ---
    mlp = MLPDynamicsBaseline(obs_dim=8, action_dim=4, hidden_dim=64)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    mlp.train()
    for _ in range(25):
        for batch in train_loader_std:
            inputs = batch["inputs"]
            curr_obs = inputs.observations[:, :-1].reshape(-1, 8)
            curr_act = inputs.actions[:, :-1].reshape(-1, 4)
            next_target = inputs.observations[:, 1:].reshape(-1, 8)
            step_mask = inputs.observation_mask[:, 1:].reshape(-1, 8)

            opt_mlp.zero_grad()
            pred_next = mlp(curr_obs, curr_act)
            loss = torch.mean(((pred_next - next_target) ** 2) * step_mask)
            loss.backward()
            opt_mlp.step()

    mlp.eval()
    all_mlp_preds = []
    all_mlp_targets = []
    all_mlp_masks = []
    with torch.no_grad():
        for batch in test_loader_std:
            inputs = batch["inputs"]
            raw_obs = batch["raw_observations"]
            raw_mask = batch["raw_mask"]

            curr_obs = inputs.observations[:, :-1].reshape(-1, 8)
            curr_act = inputs.actions[:, :-1].reshape(-1, 4)
            pred_norm = mlp(curr_obs, curr_act)
            pred_phys = normalizer.denormalize(pred_norm)

            all_mlp_preds.append(pred_phys)
            all_mlp_targets.append(raw_obs[:, 1:].reshape(-1, 8))
            all_mlp_masks.append(raw_mask[:, 1:].reshape(-1, 8))

    mlp_report = compute_comprehensive_metrics(
        predictions_phys=torch.cat(all_mlp_preds, dim=0),
        targets_phys=torch.cat(all_mlp_targets, dim=0),
        mask=torch.cat(all_mlp_masks, dim=0),
        normalizer=normalizer,
        model_name="MLP_Dynamics",
    )

    # 3. Load Frozen baseline_001
    baseline_model = CausalWorldModel(WorldModelConfig.from_yaml("configs/world_model.yaml"))
    load_checkpoint(Path("artifacts/baseline/best.pt"), baseline_model)
    baseline_model.eval()

    print("\n--- Diagnostic 1: Reconstruction vs 1-Step Prediction (baseline_001) ---")
    base_recon_det, base_pred_det = evaluate_reconstruction_vs_prediction(
        baseline_model, test_loader_std, normalizer, deterministic=True
    )
    base_recon_stoch, base_pred_stoch = evaluate_reconstruction_vs_prediction(
        baseline_model, test_loader_std, normalizer, deterministic=False
    )

    print(f"Reconstruction (Deterministic): Train-Norm MAE = {base_recon_det.train_normalized_aggregate_mae:.4f} | Raw MAE = {base_recon_det.raw_overall_mae:.4f}")
    print(f"1-Step Pred    (Deterministic): Train-Norm MAE = {base_pred_det.train_normalized_aggregate_mae:.4f} | Raw MAE = {base_pred_det.raw_overall_mae:.4f}")
    print(f"1-Step Pred    (Stochastic)   : Train-Norm MAE = {base_pred_stoch.train_normalized_aggregate_mae:.4f} | Raw MAE = {base_pred_stoch.raw_overall_mae:.4f}")

    print("\n--- Diagnostic 2: Hidden-State Linear Probes (baseline_001) ---")
    probe_report = fit_and_evaluate_latent_probes(
        baseline_model, oracle_train_eps, oracle_test_eps, normalizer
    )
    for var, pm in probe_report.probe_metrics.items():
        print(f"  Probe Z_t -> {var:<12}: R^2 = {pm.r2_score:>7.3f} | MAE = {pm.mae:>7.3f} (Mean: {pm.target_mean:.2f}, Std: {pm.target_std:.2f})")

    print("\n--- Diagnostic 3: Action-Sensitivity Causal Check (baseline_001) ---")
    act_report = evaluate_action_sensitivity(baseline_model, normalizer)
    print(f"  All directional checks passed: {act_report.all_directional_checks_passed}")
    print(f"  Valve sweep F_cool response: {act_report.valve_sweep.predicted_means['F_cool']}")
    print(f"  Throttle sweep L_cpu response: {act_report.throttle_sweep.predicted_means['L_cpu']}")
    print(f"  Throttle sweep T_core response: {act_report.throttle_sweep.predicted_means['T_core']}")

    # 4. Controlled Ablation Matrix Training
    # Define Ablation Experiments
    ablation_specs = [
        ("diag_001_beta_kl_0", {"beta_kl": 0.0, "latent_dim": 32, "context_length": 40}),
        ("diag_002_beta_kl_0001", {"beta_kl": 0.001, "latent_dim": 32, "context_length": 40}),
        ("diag_003_latent_64", {"beta_kl": 0.01, "latent_dim": 64, "context_length": 40}),
        ("diag_004_latent_128", {"beta_kl": 0.01, "latent_dim": 128, "context_length": 40}),
        ("diag_005_context_10", {"beta_kl": 0.01, "latent_dim": 32, "context_length": 10}),
        ("diag_006_context_20", {"beta_kl": 0.01, "latent_dim": 32, "context_length": 20}),
        ("diag_007_context_80", {"beta_kl": 0.01, "latent_dim": 32, "context_length": 80}),
    ]

    ablation_results: Dict[str, Any] = {}

    for exp_name, params in ablation_specs:
        print(f"\nTraining Ablation [{exp_name}] with params {params}...")
        set_seed(seed)

        wm_cfg = WorldModelConfig.from_yaml("configs/world_model.yaml")
        wm_cfg.model.latent_dim = params["latent_dim"]
        wm_cfg.loss_weights.beta_kl = params["beta_kl"]
        wm_cfg.training.max_epochs = 15
        wm_cfg.training.early_stopping_patience = 5

        ctx_l = params["context_length"]
        tr_ds = PrismWindowedDataset(train_eps, context_length=ctx_l, prediction_length=1, stride=2, normalizer=normalizer, rng_seed=seed)
        v_ds = PrismWindowedDataset(val_eps, context_length=ctx_l, prediction_length=1, stride=2, normalizer=normalizer)
        te_ds = PrismWindowedDataset(test_eps, context_length=ctx_l, prediction_length=1, stride=2, normalizer=normalizer)

        tr_ld = create_dataloader(tr_ds, batch_size=32, shuffle=True)
        v_ld = create_dataloader(v_ds, batch_size=32, shuffle=False)
        te_ld = create_dataloader(te_ds, batch_size=32, shuffle=False)

        abl_model = CausalWorldModel(wm_cfg)
        abl_opt = torch.optim.AdamW(abl_model.parameters(), lr=1e-3, weight_decay=1e-5)

        exp_save_dir = out_path / exp_name
        abl_trainer = WorldModelTrainer(
            model=abl_model,
            optimizer=abl_opt,
            train_loader=tr_ld,
            val_loader=v_ld,
            normalizer=normalizer,
            config=wm_cfg,
            save_dir=exp_save_dir,
        )
        abl_trainer.fit()

        # Evaluate on test set
        load_checkpoint(exp_save_dir / "best.pt", abl_model)
        abl_recon, abl_pred = evaluate_reconstruction_vs_prediction(
            abl_model, te_ld, normalizer, deterministic=True
        )

        # Evaluate latent probe
        abl_probe = fit_and_evaluate_latent_probes(
            abl_model, oracle_train_eps, oracle_test_eps, normalizer
        )

        ablation_results[exp_name] = {
            "params": params,
            "reconstruction": abl_recon.to_dict(),
            "prediction": abl_pred.to_dict(),
            "probes": abl_probe.to_dict(),
        }

        print(f"[{exp_name}] -> 1-Step Pred Train-Norm MAE: {abl_pred.train_normalized_aggregate_mae:.4f} | Raw MAE: {abl_pred.raw_overall_mae:.4f} | T_core: {abl_pred.per_channel['T_core'].mae:.3f}°C | T_cool: {abl_pred.per_channel['T_cool'].mae:.3f}°C")

    # 5. Compile Final Diagnostic Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "baselines": {
            "persistence": pers_report.to_dict(),
            "mlp_dynamics": mlp_report.to_dict(),
        },
        "baseline_001": {
            "reconstruction_deterministic": base_recon_det.to_dict(),
            "reconstruction_stochastic": base_recon_stoch.to_dict(),
            "prediction_deterministic": base_pred_det.to_dict(),
            "prediction_stochastic": base_pred_stoch.to_dict(),
            "latent_probes": probe_report.to_dict(),
            "action_sensitivity": act_report.to_dict(),
        },
        "ablations": ablation_results,
    }

    report_file = out_path / "diagnosis_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n===============================================================================")
    print(f"DIAGNOSTIC COMPLETED: Full structured report saved to {report_file}")
    print(f"===============================================================================")


if __name__ == "__main__":
    run_diagnostics()
