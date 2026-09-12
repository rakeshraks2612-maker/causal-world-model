"""Training Pipeline for Baseline 005 Unified World Model (Task 5.10).

Trains baseline_005 on D_unified incorporating:
- 40% Closed-loop PID, 40% Excitation, 20% Physically-Supported Acute Transients
- Shock-Weighted 8D Reconstruction Loss (5x weight on T >= 94°C or |Delta T| >= 10°C)
- K=5 Multi-Step Autoregressive Rollout Loss (lambda_rollout = 0.5)
- Preserved causal graph architecture (latent_dim=64, residual transition, learned variance)
- Saves artifacts to artifacts/baseline_005/
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import yaml

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.training.dataset import PrismWindowedDataset
from prism.training.normalization import ObservationNormalizer
from prism.training.batching import create_dataloader
from prism.training.metrics import compute_regression_metrics


def evaluate_pump_sensitivity(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    device: torch.device,
) -> dict[str, Any]:
    """Test pump 1 vs pump 3 action sensitivity from identical inferred state across horizons."""
    model.eval()
    scen4_path = Path("data/decision_benchmark/scenarios/scenario_04_pump/learner.npz")
    if not scen4_path.exists():
        raise FileNotFoundError(f"Scenario 4 learner file not found: {scen4_path}")
        
    from prism.dataset.decision_benchmark import LearnerDecisionScenario
    learner_scen = LearnerDecisionScenario.load_npz(scen4_path)
    obs_raw = learner_scen.historical_observations
    mask_raw = learner_scen.historical_observation_mask
    act_raw = learner_scen.historical_actions
    
    norm_obs = normalizer.normalize(torch.tensor(obs_raw, dtype=torch.float32)).numpy()
    inputs_ctx = ModelInputs(
        observations=torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device),
        observation_mask=torch.tensor(mask_raw, dtype=torch.float32).unsqueeze(0).to(device),
        actions=torch.tensor(act_raw, dtype=torch.float32).unsqueeze(0).to(device),
    )
    
    H = 40
    horizons = [1, 5, 10, 20, 40]
    
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
        
        pred_p1_norm = rollout_p1.observations.mean.squeeze(0).cpu()
        pred_p3_norm = rollout_p3.observations.mean.squeeze(0).cpu()
        
        pred_p1 = normalizer.denormalize(pred_p1_norm).numpy()
        pred_p3 = normalizer.denormalize(pred_p3_norm).numpy()
        
    delta_F = pred_p3[:, 3] - pred_p1[:, 3]
    delta_P = pred_p3[:, 2] - pred_p1[:, 2]
    delta_Tcore = pred_p3[:, 0] - pred_p1[:, 0]
    delta_Tcool = pred_p3[:, 1] - pred_p1[:, 1]
    
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
        }
        
    return {
        "trajectory_divergence_norm": traj_divergence,
        "per_horizon": per_h,
    }


def train_baseline_005(
    data_dir: str | Path = "data/unified_dataset/learner",
    save_dir: str | Path = "artifacts/baseline_005",
    epochs: int = 35,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "validation"
    
    train_files = sorted(list(train_dir.glob("*.npz")))
    val_files = sorted(list(val_dir.glob("*.npz")))
    
    print(f"Loading {len(train_files)} train episodes and {len(val_files)} val episodes...")
    train_episodes = [LearnerEpisode.load_npz(f) for f in train_files]
    val_episodes = [LearnerEpisode.load_npz(f) for f in val_files]
    
    normalizer = ObservationNormalizer.fit(train_episodes)
    normalizer.save_yaml(save_path / "normalization.yaml")
    
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
    config.loss_weights.beta_kl = 0.001
    config.loss_weights.lambda_rollout = 0.5
    config.loss_weights.rollout_horizon = 5
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-5
    config.training.max_epochs = epochs
    config.training.early_stopping_patience = 10
    
    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
        
    device = torch.device("cpu")
    model = CausalWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    t_mean = normalizer.stats.means[0]
    t_std = normalizer.stats.stds[0]
    
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    
    print(f"\n--> Training Baseline 005 for {epochs} epochs...")
    print(f"    Settings: beta_kl={config.loss_weights.beta_kl}, K_rollout=5 (lambda=0.5), Shock-Weighting=5x")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_obs_acc = 0.0
        train_trans_acc = 0.0
        train_roll_acc = 0.0
        n_train = 0
        
        for batch in train_loader:
            inputs: ModelInputs = batch["inputs"]
            obs = inputs.observations.to(device)
            mask = inputs.observation_mask.to(device)
            act = inputs.actions.to(device)
            dev_inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)
            
            # Compute physical T_core and shock weights: 5x on T >= 94°C or |Delta T| >= 10°C
            t_core_raw = obs[:, :, 0] * t_std + t_mean
            delta_T = torch.zeros_like(t_core_raw)
            delta_T[:, 1:] = torch.abs(t_core_raw[:, 1:] - t_core_raw[:, :-1])
            
            shock_mask = (t_core_raw >= 94.0) | (delta_T >= 10.0)
            weights = torch.ones_like(t_core_raw)
            weights[shock_mask] = 5.0
            
            optimizer.zero_grad()
            loss_out = model.compute_loss(dev_inputs, timestep_weights=weights)
            loss_out.total_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            train_loss_acc += loss_out.total_loss.item()
            train_obs_acc += loss_out.obs_loss.item()
            train_trans_acc += loss_out.trans_loss.item()
            if loss_out.rollout_loss is not None:
                train_roll_acc += loss_out.rollout_loss.item()
            n_train += 1
            
        # Validation
        model.eval()
        val_errors_all = []
        with torch.no_grad():
            for batch in val_loader:
                v_inputs: ModelInputs = batch["inputs"]
                v_obs = v_inputs.observations.to(device)
                v_mask = v_inputs.observation_mask.to(device)
                v_act = v_inputs.actions.to(device)
                v_dev = ModelInputs(observations=v_obs, observation_mask=v_mask, actions=v_act)
                
                post, _, recon = model(v_dev)
                rec_raw = normalizer.denormalize(recon.mean.cpu()).numpy()
                target_raw = normalizer.denormalize(v_obs.cpu()).numpy()
                
                err = np.abs(rec_raw - target_raw)
                val_errors_all.append(err)
                
        all_errs = np.concatenate(val_errors_all, axis=0)
        val_mae = float(np.mean(all_errs))
        val_t_mae = float(np.mean(all_errs[:, :, 0]))
        
        train_loss_avg = train_loss_acc / max(1, n_train)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_mae": val_mae,
            "val_t_mae": val_t_mae,
        })
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss_avg:.4f} | Val MAE: {val_mae:.4f} | Val T_core MAE: {val_t_mae:.2f}°C")
            
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "config": config.to_dict(),
            }, save_path / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"--> Early stopping at epoch {epoch} (Best Epoch: {best_epoch}, Best Val MAE: {best_val_mae:.4f})")
                break
                
    # Load best checkpoint
    best_ckpt = torch.load(save_path / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    sens = evaluate_pump_sensitivity(model, normalizer, device)
    
    manifest = {
        "model_name": "baseline_005",
        "best_epoch": best_epoch,
        "best_val_mae": float(best_val_mae),
        "pump_sensitivity": sens,
        "history": history,
    }
    
    with open(save_path / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nTraining Baseline 005 Complete! Saved to {save_path}")
    print(f"Best Val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    print(f"Pump Action Sensitivity (Scenario 4 Context):")
    print(f"  h=1:  ΔF={sens['per_horizon']['h=1']['delta_F_cool']:+.2f} L/min, ΔP={sens['per_horizon']['h=1']['delta_P_sys']:+.2f} bar, ΔTcore={sens['per_horizon']['h=1']['delta_T_core']:+.2f}°C")
    print(f"  h=5:  ΔF={sens['per_horizon']['h=5']['delta_F_cool']:+.2f} L/min, ΔP={sens['per_horizon']['h=5']['delta_P_sys']:+.2f} bar, ΔTcore={sens['per_horizon']['h=5']['delta_T_core']:+.2f}°C")
    print(f"  h=10: ΔF={sens['per_horizon']['h=10']['delta_F_cool']:+.2f} L/min, ΔP={sens['per_horizon']['h=10']['delta_P_sys']:+.2f} bar, ΔTcore={sens['per_horizon']['h=10']['delta_T_core']:+.2f}°C")
    print(f"  h=40: ΔF={sens['per_horizon']['h=40']['delta_F_cool']:+.2f} L/min, ΔP={sens['per_horizon']['h=40']['delta_P_sys']:+.2f} bar, ΔTcore={sens['per_horizon']['h=40']['delta_T_core']:+.2f}°C")
    print(f"  Trajectory Divergence Norm: {sens['trajectory_divergence_norm']:.4f}")
    
    return manifest


if __name__ == "__main__":
    train_baseline_005()
