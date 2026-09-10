"""Training and Freezing Pipeline for PRISM Baseline 002.

Executes:
1. Preserves baseline_001 in artifacts/baseline/baseline_001/
2. Sets deterministic seed = 42
3. Fits train-only normalizer and saves to artifacts/baseline_002/normalization.yaml
4. Trains CausalWorldModel with latent_dim = 64, beta_kl = 0.001, context = 40
5. Evaluates on validation split and untouched test split (1-step prediction & reconstruction)
6. Freezes artifacts/baseline_002/ with best.pt, last.pt, config.yaml, normalization.yaml, metrics.json
"""

from __future__ import annotations
import sys
import shutil
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn

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
from prism.diagnosis.domain_metrics import compute_comprehensive_metrics
from prism.diagnosis.reconstruction_vs_prediction import evaluate_reconstruction_vs_prediction
from prism.diagnosis.latent_probes import fit_and_evaluate_latent_probes


def train_and_freeze_baseline_002() -> None:
    config_path = "configs/training_baseline_002.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    print("=" * 115)
    print("TASK 3.2B — PRISM BASELINE 002 TRAINING & FREEZING PIPELINE")
    print("=" * 115)

    # 1. Organize baseline_001
    base_dir = Path("artifacts/baseline")
    b1_dir = base_dir / "baseline_001"
    b1_dir.mkdir(parents=True, exist_ok=True)

    for item in ["best.pt", "last.pt", "metrics.json", "normalization.yaml"]:
        src = base_dir / item
        dst = b1_dir / item
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    print(f"Preserved baseline_001 in {b1_dir}")

    # Create baseline_002 dir
    b2_dir = Path(cfg["checkpointing"]["save_dir"])
    b2_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Learner and Oracle datasets
    train_dir = Path(cfg["dataset"]["train_dir"])
    val_dir = Path(cfg["dataset"]["val_dir"])
    test_dir = Path(cfg["dataset"]["test_dir"])

    train_eps = [LearnerEpisode.load_npz(f) for f in sorted(train_dir.glob("*.npz"))]
    val_eps = [LearnerEpisode.load_npz(f) for f in sorted(val_dir.glob("*.npz"))]
    test_eps = [LearnerEpisode.load_npz(f) for f in sorted(test_dir.glob("*.npz"))]

    oracle_train_eps = [OracleEpisode.load_npz(f) for f in sorted(Path("data/pilot/oracle/train").glob("*.npz"))]
    oracle_test_eps = [OracleEpisode.load_npz(f) for f in sorted(Path("data/pilot/oracle/test").glob("*.npz"))]

    # 3. Fit Normalizer Strictly on Train
    normalizer = ObservationNormalizer.fit(train_eps)
    norm_save_path = b2_dir / "normalization.yaml"
    normalizer.save_yaml(norm_save_path)
    print(f"Computed training-set normalizer -> saved to {norm_save_path}")

    # Copy config.yaml
    shutil.copy2("configs/world_model_002.yaml", b2_dir / "config.yaml")

    # 4. Create Windowed Datasets
    ctx_len = cfg["dataset"]["context_length"]
    pred_len = cfg["dataset"]["prediction_length"]
    stride = cfg["dataset"]["stride"]
    mask_drop = cfg["masking"]["dropout_prob"]
    batch_size = cfg["training"]["batch_size"]

    train_ds = PrismWindowedDataset(train_eps, ctx_len, pred_len, stride, normalizer, mask_dropout_prob=mask_drop, rng_seed=seed)
    val_ds = PrismWindowedDataset(val_eps, ctx_len, pred_len, stride, normalizer, mask_dropout_prob=0.0)
    test_ds = PrismWindowedDataset(test_eps, ctx_len, pred_len, stride, normalizer, mask_dropout_prob=0.0)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset windows: {len(train_ds)} Train, {len(val_ds)} Val, {len(test_ds)} Test.")

    # 5. Initialize Model
    wm_config = WorldModelConfig.from_yaml("configs/world_model_002.yaml")
    model = CausalWorldModel(wm_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=wm_config,
        save_dir=b2_dir,
    )

    # 6. Fit Model
    print(f"Training Baseline 002 (latent_dim=64, beta_kl=0.001, context=40)...")
    train_summary = trainer.fit()
    print(f"Training completed at epoch {train_summary['final_epoch']} with Best Val MAE: {train_summary['best_val_mae']:.4f}")

    # 7. Final Evaluation on Validation and Test Sets
    load_checkpoint(b2_dir / "best.pt", model)

    print("\n--- Evaluating Baseline 002 on Validation Split ---")
    val_loss, val_pred_summary = trainer.evaluate(val_loader)
    val_recon_rep, val_pred_rep = evaluate_reconstruction_vs_prediction(model, val_loader, normalizer, deterministic=True)

    print("\n--- Evaluating Baseline 002 on Untouched Test Split ---")
    test_loss, test_pred_summary = trainer.evaluate(test_loader)
    test_recon_rep, test_pred_rep = evaluate_reconstruction_vs_prediction(model, test_loader, normalizer, deterministic=True)
    test_recon_stoch, test_pred_stoch = evaluate_reconstruction_vs_prediction(model, test_loader, normalizer, deterministic=False)

    # Latent probes
    probe_rep = fit_and_evaluate_latent_probes(model, oracle_train_eps, oracle_test_eps, normalizer)

    # 8. Export metrics.json
    metrics_data = {
        "experiment": "baseline_002",
        "seed": seed,
        "config": {
            "latent_dim": 64,
            "beta_kl": 0.001,
            "context_length": 40,
            "epochs_trained": train_summary["final_epoch"],
        },
        "validation": {
            "loss": val_loss,
            "1step_prediction_deterministic": val_pred_rep.to_dict(),
            "reconstruction_deterministic": val_recon_rep.to_dict(),
        },
        "test": {
            "loss": test_loss,
            "1step_prediction_deterministic": test_pred_rep.to_dict(),
            "1step_prediction_stochastic": test_pred_stoch.to_dict(),
            "reconstruction_deterministic": test_recon_rep.to_dict(),
            "reconstruction_stochastic": test_recon_stoch.to_dict(),
            "latent_probes": probe_rep.to_dict(),
        },
    }

    metrics_file = b2_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nSaved canonical Baseline 002 metrics to {metrics_file}")
    print("\n" + "=" * 115)
    print(f"{'Channel':<15} | {'Val Pred MAE':<15} | {'Test Pred MAE':<15} | {'Test Pred RMSE':<15} | {'Test Gaussian NLL':<18}")
    print("-" * 115)
    for ch, m in test_pred_rep.per_channel.items():
        v_mae = val_pred_rep.per_channel[ch].mae
        print(f"{ch:<15} | {v_mae:>13.3f}   | {m.mae:>13.3f}   | {m.rmse:>13.3f}   | {m.nll:>16.3f}")
    print("=" * 115)
    print(f"{'Train-Norm MAE':<15} | {val_pred_rep.train_normalized_aggregate_mae:>13.4f}   | {test_pred_rep.train_normalized_aggregate_mae:>13.4f}   | {'-':>15} | {test_pred_rep.overall_nll:>16.4f}")
    print(f"{'Raw Overall MAE':<15} | {val_pred_rep.raw_overall_mae:>13.4f}   | {test_pred_rep.raw_overall_mae:>13.4f}   | {test_pred_rep.overall_rmse:>13.4f}   | {'-':>18}")
    print("=" * 115)


if __name__ == "__main__":
    train_and_freeze_baseline_002()
