"""Master Training and Benchmarking Script for Task 3.2.

Executes:
1. Train-only normalization calculation
2. Diagnostic overfit test on tiny batch
3. Baseline A (Persistence) evaluation
4. Baseline B (MLP Dynamics) training & evaluation
5. PRISM Causal World Model training with validation early stopping
6. Final evaluation on TEST split
7. Comparative benchmark reporting and metrics.json export
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.seed import set_seed
from prism.training.normalization import ObservationNormalizer
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader
from prism.training.trainer import WorldModelTrainer
from prism.training.metrics import (
    PersistenceBaseline,
    MLPDynamicsBaseline,
    compute_regression_metrics,
    format_benchmark_table,
    export_metrics_json,
    EvaluationSummary,
)


def evaluate_persistence_on_loader(dataloader, normalizer: ObservationNormalizer) -> EvaluationSummary:
    """Evaluate Persistence Baseline (O_hat_{t+1} = O_t) on a DataLoader."""
    baseline = PersistenceBaseline()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            raw_obs = batch["raw_observations"]  # [B, L, 8]
            raw_mask = batch["raw_mask"]          # [B, L, 8]

            # Persistence prediction: pred[:, 0] = raw_obs[:, 0]
            pred = baseline.predict_one_step(raw_obs)  # [B, L-1, 8]
            target = raw_obs[:, 1:]                   # [B, L-1, 8]
            mask = raw_mask[:, 1:]                    # [B, L-1, 8]

            all_preds.append(pred.reshape(-1, 8))
            all_targets.append(target.reshape(-1, 8))
            all_masks.append(mask.reshape(-1, 8))

    return compute_regression_metrics(
        predictions=torch.cat(all_preds, dim=0),
        targets=torch.cat(all_targets, dim=0),
        mask=torch.cat(all_masks, dim=0),
        model_name="Persistence",
    )


def train_and_evaluate_mlp_baseline(
    train_loader,
    test_loader,
    normalizer: ObservationNormalizer,
    epochs: int = 25,
) -> EvaluationSummary:
    """Train and evaluate the Supervised MLP Dynamics Baseline."""
    mlp = MLPDynamicsBaseline(obs_dim=8, action_dim=4, hidden_dim=64)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    mlp.train()
    for _ in range(epochs):
        for batch in train_loader:
            inputs = batch["inputs"]
            norm_obs = inputs.observations  # [B, L, 8]
            act = inputs.actions            # [B, L, 4]
            mask = inputs.observation_mask  # [B, L, 8]

            # Current and next step in normalized space
            curr_obs = norm_obs[:, :-1].reshape(-1, 8)
            curr_act = act[:, :-1].reshape(-1, 4)
            next_obs_target = norm_obs[:, 1:].reshape(-1, 8)
            step_mask = mask[:, 1:].reshape(-1, 8)

            optimizer.zero_grad()
            next_obs_pred = mlp(curr_obs, curr_act)
            # Masked MSE loss
            loss = torch.mean(((next_obs_pred - next_obs_target) ** 2) * step_mask)
            loss.backward()
            optimizer.step()

    # Evaluate on test loader in physical units
    mlp.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"]
            raw_obs = batch["raw_observations"]
            raw_mask = batch["raw_mask"]

            norm_obs = inputs.observations
            act = inputs.actions

            curr_obs = norm_obs[:, :-1].reshape(-1, 8)
            curr_act = act[:, :-1].reshape(-1, 4)

            pred_norm = mlp(curr_obs, curr_act)
            pred_denorm = normalizer.denormalize(pred_norm)

            target = raw_obs[:, 1:].reshape(-1, 8)
            mask = raw_mask[:, 1:].reshape(-1, 8)

            all_preds.append(pred_denorm)
            all_targets.append(target)
            all_masks.append(mask)

    return compute_regression_metrics(
        predictions=torch.cat(all_preds, dim=0),
        targets=torch.cat(all_targets, dim=0),
        mask=torch.cat(all_masks, dim=0),
        model_name="MLP_Dynamics",
    )


def run_experiment(config_path: str = "configs/training_baseline.yaml") -> None:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    seed = cfg_dict.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    print("=" * 115)
    print("TASK 3.2 — PRISM WORLD MODEL BASELINE TRAINING PIPELINE")
    print("=" * 115)

    # 1. Load Raw Training, Validation, and Test Episodes
    train_dir = Path(cfg_dict["dataset"]["train_dir"])
    val_dir = Path(cfg_dict["dataset"]["val_dir"])
    test_dir = Path(cfg_dict["dataset"]["test_dir"])

    train_eps = [LearnerEpisode.load_npz(f) for f in sorted(train_dir.glob("*.npz"))]
    val_eps = [LearnerEpisode.load_npz(f) for f in sorted(val_dir.glob("*.npz"))]
    test_eps = [LearnerEpisode.load_npz(f) for f in sorted(test_dir.glob("*.npz"))]

    print(f"Loaded {len(train_eps)} Train, {len(val_eps)} Validation, {len(test_eps)} Test episodes.")

    # 2. Fit Normalizer Strictly on Train Split
    normalizer = ObservationNormalizer.fit(train_eps)
    norm_save_path = Path(cfg_dict.get("normalization", {}).get("save_path", "artifacts/baseline/normalization.yaml"))
    normalizer.save_yaml(norm_save_path)
    print(f"Computed training-set-only normalization statistics -> saved to {norm_save_path}")

    # 3. Create Windowed Datasets and DataLoaders
    ctx_len = cfg_dict["dataset"]["context_length"]
    pred_len = cfg_dict["dataset"]["prediction_length"]
    stride = cfg_dict["dataset"]["stride"]
    mask_drop = cfg_dict.get("masking", {}).get("dropout_prob", 0.0)
    batch_size = cfg_dict["training"]["batch_size"]

    train_ds = PrismWindowedDataset(train_eps, ctx_len, pred_len, stride, normalizer, mask_drop, rng_seed=seed)
    val_ds = PrismWindowedDataset(val_eps, ctx_len, pred_len, stride, normalizer, mask_dropout_prob=0.0)
    test_ds = PrismWindowedDataset(test_eps, ctx_len, pred_len, stride, normalizer, mask_dropout_prob=0.0)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Created Windowed Datasets: {len(train_ds)} Train windows, {len(val_ds)} Val windows, {len(test_ds)} Test windows.")

    # 4. Critical Diagnostic: Overfit a Tiny Subset
    print("\n[Diagnostic 3.2.17] Overfitting a tiny 4-episode batch...")
    tiny_eps = train_eps[:4]
    tiny_ds = PrismWindowedDataset(tiny_eps, ctx_len, pred_len, stride=10, normalizer=normalizer)
    tiny_loader = create_dataloader(tiny_ds, batch_size=len(tiny_ds), shuffle=False)

    diag_model = CausalWorldModel(WorldModelConfig.from_yaml("configs/world_model.yaml"))
    diag_opt = torch.optim.AdamW(diag_model.parameters(), lr=3e-3)
    diag_trainer = WorldModelTrainer(diag_model, diag_opt, tiny_loader, normalizer=normalizer)
    
    initial_loss = diag_model.compute_loss(next(iter(tiny_loader))["inputs"]).total_loss.item()
    final_overfit_loss = diag_trainer.overfit_diagnostic(num_epochs=35)
    print(f"Diagnostic Loss: Initial {initial_loss:.4f} -> Final {final_overfit_loss:.4f} (Reduction: {(1.0 - final_overfit_loss/initial_loss):.1%}) [PASS]")

    # 5. Evaluate Baseline A: Persistence
    print("\nEvaluating Baseline A (Persistence)...")
    pers_summary = evaluate_persistence_on_loader(test_loader, normalizer)

    # 6. Train and Evaluate Baseline B: MLP Dynamics
    print("Training Baseline B (Supervised MLP Dynamics)...")
    mlp_summary = train_and_evaluate_mlp_baseline(train_loader, test_loader, normalizer, epochs=25)

    # 7. Train PRISM Causal World Model
    print("\nTraining PRISM Causal World Model...")
    wm_config = WorldModelConfig.from_yaml("configs/world_model.yaml")
    wm_config.training.max_epochs = cfg_dict["training"]["epochs"]
    wm_config.training.early_stopping_patience = cfg_dict["training"]["early_stopping_patience"]

    prism_model = CausalWorldModel(wm_config)
    optimizer = torch.optim.AdamW(
        prism_model.parameters(),
        lr=cfg_dict["training"]["learning_rate"],
        weight_decay=cfg_dict["training"]["weight_decay"],
    )

    save_dir = Path(cfg_dict.get("checkpointing", {}).get("save_dir", "artifacts/baseline"))
    trainer = WorldModelTrainer(
        model=prism_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=wm_config,
        save_dir=save_dir,
    )

    trainer.fit()

    # 8. Final One-Step Evaluation of PRISM on TEST Split
    print("\nExecuting Final Evaluation of PRISM World Model on TEST Split...")
    test_loss, prism_summary = trainer.evaluate(test_loader)

    # 9. Format & Print Benchmark Table
    summaries = [pers_summary, mlp_summary, prism_summary]
    table_str = format_benchmark_table(summaries)
    print("\n" + table_str)

    # 10. Export Machine-Readable Metrics JSON
    metrics_file = Path(cfg_dict.get("checkpointing", {}).get("metrics_file", "artifacts/baseline/metrics.json"))
    export_metrics_json(
        eval_summaries=summaries,
        file_path=metrics_file,
        metadata={
            "experiment": cfg_dict.get("experiment", {}).get("name", "baseline_001"),
            "seed": seed,
            "train_episodes": len(train_eps),
            "test_episodes": len(test_eps),
        },
    )
    print(f"\nSaved benchmark metrics to {metrics_file}")


if __name__ == "__main__":
    run_experiment()
