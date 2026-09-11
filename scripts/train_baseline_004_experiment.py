"""Training and Controlled Experiment Pipeline for PRISM Baseline 004 (Task 5.7C).

Conducts controlled training experiment:
- Excitation training data with balanced pump stages, short holds, zero confounding
- Multi-step Autoregressive Rollout Loss (evaluating K=1, K=5, K=10)
- Preserves exact model architecture (latent_dim=64, encoder/transition/decoder=64, residual=True, learn_variance=True)
- Evaluates Action Sensitivity Gates (h=1, 5, 10, 20, 40)
- Saves best model to artifacts/baseline_004/
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import yaml

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.training.dataset import PrismWindowedDataset
from prism.training.normalization import ObservationNormalizer
from prism.training.trainer import WorldModelTrainer
from prism.training.batching import create_dataloader


def evaluate_pump_sensitivity(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Test pump 1 vs pump 3 action sensitivity from identical inferred state across h=1,5,10,20,40."""
    model.eval()
    
    # Load Scenario 4 learner context to test from exact benchmark evaluation state
    scen4_path = Path("data/decision_benchmark/scenarios/scenario_04_pump/learner.npz")
    if not scen4_path.exists():
        raise FileNotFoundError(f"Scenario 4 learner file not found: {scen4_path}")
    
    from prism.dataset.decision_benchmark import LearnerDecisionScenario
    learner_scen = LearnerDecisionScenario.load_npz(scen4_path)
    obs_raw = learner_scen.historical_observations
    mask_raw = learner_scen.historical_observation_mask
    act_raw = learner_scen.historical_actions

    # Normalize observations for model encoding
    norm_obs = normalizer.normalize(torch.tensor(obs_raw, dtype=torch.float32)).numpy()
    
    inputs_ctx = ModelInputs(
        observations=torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device),
        observation_mask=torch.tensor(mask_raw, dtype=torch.float32).unsqueeze(0).to(device),
        actions=torch.tensor(act_raw, dtype=torch.float32).unsqueeze(0).to(device),
    )

    H = 40
    horizons = [1, 5, 10, 20, 40]

    # Build future action plans for Pump 1 vs Pump 3 (keeping valve=60, throttle=100, flush=0)
    act_p1 = np.zeros((1, H, 4), dtype=np.float32)
    act_p1[:, :, 0] = 60.0
    act_p1[:, :, 1] = 100.0
    act_p1[:, :, 2] = 1.0
    act_p1[:, :, 3] = 0.0

    act_p3 = np.zeros((1, H, 4), dtype=np.float32)
    act_p3[:, :, 0] = 60.0
    act_p3[:, :, 1] = 100.0
    act_p3[:, :, 2] = 3.0
    act_p3[:, :, 3] = 0.0

    with torch.no_grad():
        rollout_p1 = model.forecast(inputs_ctx, torch.tensor(act_p1).to(device), deterministic=True)
        rollout_p3 = model.forecast(inputs_ctx, torch.tensor(act_p3).to(device), deterministic=True)

        pred_p1_norm = rollout_p1.observations.mean.squeeze(0).cpu()  # [H, 8]
        pred_p3_norm = rollout_p3.observations.mean.squeeze(0).cpu()  # [H, 8]

        pred_p1 = normalizer.denormalize(pred_p1_norm).numpy()
        pred_p3 = normalizer.denormalize(pred_p3_norm).numpy()

    # Channel indices: T_core=0, T_cool=1, P_sys=2, F_cool=3, Vib_pump=6
    delta_F = pred_p3[:, 3] - pred_p1[:, 3]
    delta_P = pred_p3[:, 2] - pred_p1[:, 2]
    delta_Tcore = pred_p3[:, 0] - pred_p1[:, 0]
    delta_Tcool = pred_p3[:, 1] - pred_p1[:, 1]
    delta_Vib = pred_p3[:, 6] - pred_p1[:, 6]

    diff_norm = pred_p3_norm.numpy() - pred_p1_norm.numpy()
    traj_divergence = float(np.linalg.norm(diff_norm))

    per_h = {}
    for h in horizons:
        idx = h - 1
        per_h[f"h={h}"] = {
            "delta_F_cool": float(delta_F[idx]),
            "delta_P_sys": float(delta_P[idx]),
            "delta_T_core": float(delta_Tcore[idx]),
            "delta_T_cool": float(delta_Tcool[idx]),
            "delta_Vib_pump": float(delta_Vib[idx]),
            "p1_F": float(pred_p1[idx, 3]),
            "p3_F": float(pred_p3[idx, 3]),
            "p1_T": float(pred_p1[idx, 0]),
            "p3_T": float(pred_p3[idx, 0]),
        }

    return {
        "trajectory_divergence_norm": traj_divergence,
        "per_horizon": per_h,
        "delta_F_mean": float(np.mean(delta_F)),
        "delta_P_mean": float(np.mean(delta_P)),
        "delta_Tcore_mean": float(np.mean(delta_Tcore)),
    }


def train_variant(
    variant_name: str,
    k_rollout: int,
    lambda_rollout: float,
    train_episodes: List[LearnerEpisode],
    val_episodes: List[LearnerEpisode],
    normalizer: ObservationNormalizer,
    epochs: int = 35,
    seed: int = 42,
) -> Tuple[CausalWorldModel, Dict[str, Any], Path]:
    """Train a single controlled model variant."""
    save_dir = Path(f"artifacts/baseline_004_variants/{variant_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Training Variant: {variant_name} (Rollout Horizon K={k_rollout}, Lambda={lambda_rollout})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len = 40
    train_dataset = PrismWindowedDataset(
        episodes=train_episodes,
        context_length=context_len,
        prediction_length=1,
        stride=2,
        normalizer=normalizer,
    )
    val_dataset = PrismWindowedDataset(
        episodes=val_episodes,
        context_length=context_len,
        prediction_length=1,
        stride=4,
        normalizer=normalizer,
    )

    train_loader = create_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=16, shuffle=False)

    config = WorldModelConfig()
    config.model.latent_dim = 64
    config.encoder.hidden_dim = 64
    config.encoder.num_layers = 2
    config.transition.hidden_dim = 64
    config.transition.num_layers = 2
    config.transition.residual = True
    config.decoder.hidden_dim = 64
    config.decoder.num_layers = 2
    config.decoder.learn_variance = True
    config.loss_weights.lambda_obs = 1.0
    config.loss_weights.lambda_trans = 1.0
    config.loss_weights.beta_kl = 0.01
    config.loss_weights.lambda_rollout = lambda_rollout
    config.loss_weights.rollout_horizon = k_rollout
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-5
    config.training.max_epochs = epochs
    config.training.early_stopping_patience = 8

    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    normalizer.save_yaml(save_dir / "normalization.yaml")

    model = CausalWorldModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    trainer = WorldModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=config,
        save_dir=save_dir,
    )

    history = trainer.fit()

    # Load best checkpoint
    best_ckpt = torch.load(save_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Evaluate sensitivity
    device = torch.device("cpu")
    sensitivity = evaluate_pump_sensitivity(model, normalizer, device)
    sensitivity["best_val_mae"] = history["best_val_mae"]
    sensitivity["final_epoch"] = history["final_epoch"]

    print(f"[{variant_name}] Best Val MAE: {history['best_val_mae']:.4f} | Trajectory Divergence: {sensitivity['trajectory_divergence_norm']:.4f}")
    print(f"[{variant_name}] h=40: ΔF={sensitivity['per_horizon']['h=40']['delta_F_cool']:+.2f} L/min, ΔP={sensitivity['per_horizon']['h=40']['delta_P_sys']:+.2f} bar, ΔTcore={sensitivity['per_horizon']['h=40']['delta_T_core']:+.2f}°C")

    return model, sensitivity, save_dir


def main() -> None:
    print("=========================================================================")
    print("      TRAINING PRISM BASELINE 004: MINIMAL CORRECTIVE EXPERIMENT         ")
    print("=========================================================================")

    # 1. Load Training and Validation Datasets
    exc_train_dir = Path("data/excitation_dataset/learner/train")
    exc_val_dir = Path("data/excitation_dataset/learner/validation")

    all_train_files = sorted(list(exc_train_dir.glob("*.npz")))
    all_val_files = sorted(list(exc_val_dir.glob("*.npz")))

    print(f"Loading {len(all_train_files)} training episodes from excitation dataset...")
    print(f"Loading {len(all_val_files)} validation episodes...")

    train_episodes = [LearnerEpisode.load_npz(f) for f in all_train_files]
    val_episodes = [LearnerEpisode.load_npz(f) for f in all_val_files]

    # Fit normalizer strictly on training partition
    normalizer = ObservationNormalizer.fit(train_episodes)

    # 2. Train Controlled Variants
    variants = [
        ("variant_A_k1_control", 1, 0.0),
        ("variant_B_k5_rollout", 5, 1.0),
        ("variant_C_k10_rollout", 10, 1.0),
    ]

    results = {}
    best_variant = None
    best_score = -1.0
    best_save_dir = None

    for var_name, k_roll, lambda_roll in variants:
        model, sens, var_save_dir = train_variant(
            variant_name=var_name,
            k_rollout=k_roll,
            lambda_rollout=lambda_roll,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            normalizer=normalizer,
            epochs=30,
            seed=42,
        )
        results[var_name] = sens
        # Ranking score combines trajectory divergence and physical correctness (ΔF > 0, ΔT < 0)
        phys_score = sens["trajectory_divergence_norm"]
        if sens["delta_F_mean"] > 0 and sens["delta_Tcore_mean"] < 0:
            phys_score += 10.0
        
        if phys_score > best_score:
            best_score = phys_score
            best_variant = var_name
            best_save_dir = var_save_dir

    print("\n=========================================================================")
    print("                     VARIANT COMPARISON SUMMARY                          ")
    print("=========================================================================")
    for vname, s in results.items():
        print(f"Variant: {vname:25s} | Val MAE: {s['best_val_mae']:.4f} | Div: {s['trajectory_divergence_norm']:.4f} | Mean ΔF: {s['delta_F_mean']:+.2f} | Mean ΔT: {s['delta_Tcore_mean']:+.2f}")

    print(f"\n>>> Selected Best Variant for Baseline 004: {best_variant}")

    # Copy best variant to artifacts/baseline_004/
    baseline_004_dir = Path("artifacts/baseline_004")
    baseline_004_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(best_save_dir / "best.pt", baseline_004_dir / "best.pt")
    shutil.copy(best_save_dir / "last.pt", baseline_004_dir / "last.pt")
    shutil.copy(best_save_dir / "config.yaml", baseline_004_dir / "config.yaml")
    shutil.copy(best_save_dir / "normalization.yaml", baseline_004_dir / "normalization.yaml")

    # Save summary results
    with open(baseline_004_dir / "experiment_results.json", "w") as f:
        json.dump({
            "selected_variant": best_variant,
            "variants": results,
        }, f, indent=2)

    print(f"✓ Saved baseline_004 weights, config, and normalization to {baseline_004_dir}")


if __name__ == "__main__":
    main()
