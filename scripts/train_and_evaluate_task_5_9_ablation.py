"""Task 5.9: Safety-Critical State Representation Experiment & Controlled Ablation.

Ablation Matrix:
- Variant 0: Control (baseline_004 frozen)
- Variant 1: Reduced KL Pressure (beta_kl = 1e-5)
- Variant 2: Shock-Aware Reconstruction Weighting (5x weight on T >= 94°C and |Delta T| >= 10°C)
- Variant 3: Temporal Shock Representation (Explicit Delta O_t = O_t - O_{t-1} encoder input)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from prism.world_model.config import WorldModelConfig, LossWeightsConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.world_model.encoder import BaseEncoder, GRUEncoder
from prism.training.normalization import ObservationNormalizer
from prism.dataset.schema import LearnerEpisode


# -----------------------------------------------------------------------------
# 1. Dataset Loader for Training & Diagnostic Evaluation
# -----------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Loads trajectories for training."""
    def __init__(self, data_dir: str | Path, normalizer: ObservationNormalizer) -> None:
        self.files = sorted(list(Path(data_dir).glob("*.npz")))
        self.normalizer = normalizer
        self.data = []
        for f in self.files:
            d = np.load(f)
            obs = d["observations"]
            mask = d["observation_mask"]
            act = d["actions"]
            obs_norm = normalizer.normalize(obs)
            if isinstance(obs_norm, torch.Tensor):
                obs_norm = obs_norm.cpu().numpy()
            self.data.append({
                "obs_raw": np.asarray(obs, dtype=np.float32),
                "obs_norm": np.asarray(obs_norm, dtype=np.float32),
                "mask": np.asarray(mask, dtype=bool),
                "act": np.asarray(act, dtype=np.float32),
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "obs_raw": torch.from_numpy(item["obs_raw"]),
            "obs_norm": torch.from_numpy(item["obs_norm"]),
            "mask": torch.from_numpy(item["mask"]),
            "act": torch.from_numpy(item["act"]),
        }


# -----------------------------------------------------------------------------
# 2. Custom Model & Loss Implementations for Variants
# -----------------------------------------------------------------------------

class ShockWeightedLossCalculator:
    """Variant 2: Weights reconstruction error 5x higher on safety-critical states."""
    def __init__(self, normalizer: ObservationNormalizer, beta_kl: float = 0.001) -> None:
        self.normalizer = normalizer
        self.beta_kl = beta_kl

    def compute_loss(
        self,
        posterior_latents: LatentDistribution,
        prior_transitions: LatentDistribution,
        reconstructed_obs: ObservationDistribution,
        target_obs_norm: torch.Tensor,
        target_obs_raw: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        T_core_raw = target_obs_raw[:, :, 0] # [B, T]
        delta_T = torch.zeros_like(T_core_raw)
        delta_T[:, 1:] = torch.abs(T_core_raw[:, 1:] - T_core_raw[:, :-1])
        
        shock_mask = (T_core_raw >= 94.0) | (delta_T >= 10.0)
        weights = torch.ones_like(T_core_raw)
        weights[shock_mask] = 5.0
        weights = weights.unsqueeze(-1) # [B, T, 1]
        
        diff = (reconstructed_obs.mean - target_obs_norm) ** 2
        obs_loss = (diff * weights * mask.float()).mean()
        
        trans_diff = (prior_transitions.mean - posterior_latents.mean[:, 1:]) ** 2
        trans_loss = trans_diff.mean()
        
        mu = posterior_latents.mean
        logvar = posterior_latents.logvar
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        total_loss = obs_loss + trans_loss + self.beta_kl * kl_loss
        return total_loss, {
            "obs_loss": float(obs_loss.item()),
            "trans_loss": float(trans_loss.item()),
            "kl_loss": float(kl_loss.item()),
        }


class TemporalDifferenceEncoder(nn.Module):
    """Variant 3: Encoder with explicit first-order finite differences Delta O_t."""
    def __init__(self, obs_dim: int = 8, act_dim: int = 4, latent_dim: int = 64) -> None:
        super().__init__()
        in_dim = obs_dim * 2 + act_dim + obs_dim
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, inputs: ModelInputs, hidden_state: Optional[torch.Tensor] = None) -> Tuple[LatentDistribution, Optional[torch.Tensor]]:
        obs = inputs.observations
        mask = inputs.observation_mask.float()
        act = inputs.actions
        
        delta_obs = torch.zeros_like(obs)
        delta_obs[:, 1:] = obs[:, 1:] - obs[:, :-1]
        
        feat = torch.cat([obs, mask, act, delta_obs], dim=-1)
        out, h_n = self.gru(feat, hidden_state)
        
        mean = self.fc_mean(out)
        logvar = torch.clamp(self.fc_logvar(out), min=-10.0, max=2.0)
        return LatentDistribution(mean, logvar), h_n


# -----------------------------------------------------------------------------
# 3. Training Loop for Experimental Variants
# -----------------------------------------------------------------------------

def train_experimental_variant(
    variant_name: str,
    base_model_dir: str | Path = "artifacts/baseline_004",
    train_data_dir: str | Path = "data/excitation_dataset/learner/train",
    save_dir: str | Path = "artifacts/experiment_task_5_9",
    epochs: int = 25,
    force_retrain: bool = False,
) -> Path:
    out_dir = Path(save_dir) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if (out_dir / "best.pt").exists() and not force_retrain:
        print(f"--> {variant_name} already trained at {out_dir}. Skipping training.")
        return out_dir
        
    norm = ObservationNormalizer.load_yaml(Path(base_model_dir) / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(Path(base_model_dir) / "config.yaml")
    
    train_ds = TrajectoryDataset(train_data_dir, norm)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = CausalWorldModel(cfg)
    ckpt = torch.load(Path(base_model_dir) / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    if variant_name == "variant_1_reduced_kl":
        beta_kl = 1e-5
        model.config.loss_weights.beta_kl = beta_kl
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        
    elif variant_name == "variant_2_shock_weighted":
        loss_calc = ShockWeightedLossCalculator(norm, beta_kl=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        
    elif variant_name == "variant_3_temporal_shock":
        temp_encoder = TemporalDifferenceEncoder(obs_dim=8, act_dim=4, latent_dim=cfg.model.latent_dim)
        model.encoder = temp_encoder
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    print(f"--> Training {variant_name} for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            obs_norm = batch["obs_norm"]
            obs_raw = batch["obs_raw"]
            mask = batch["mask"]
            act = batch["act"]
            
            optimizer.zero_grad()
            inputs = ModelInputs(obs_norm, mask, act)
            
            if variant_name == "variant_2_shock_weighted":
                posteriors, _ = model.encoder(inputs)
                z_samples = posteriors.sample(deterministic=False)
                prior_trans = model.transition(z_samples[:, :-1], act[:, :-1])
                reconstructed = model.decoder(z_samples)
                
                loss, _ = loss_calc.compute_loss(
                    posteriors, prior_trans, reconstructed, obs_norm, obs_raw, mask
                )
            else:
                posteriors, prior_trans, reconstructed = model(inputs)
                loss_out = model.loss_calculator.compute_loss(
                    posteriors, prior_trans, reconstructed, obs_norm, mask
                )
                loss = loss_out.total_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
            
    # Save checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, out_dir / "best.pt")
    norm.save_yaml(out_dir / "normalization.yaml")
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.to_dict(), f)
        
    return out_dir


# -----------------------------------------------------------------------------
# 4. Pure PyTorch / NumPy Linear Probe & Evaluation Functions
# -----------------------------------------------------------------------------

def compute_roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC using trapezoidal rule without external dependencies."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def train_linear_probe(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Train a simple linear probe with PyTorch BCE loss and return (ROC-AUC, Accuracy)."""
    t_X = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    n = len(X)
    idx = np.random.permutation(n)
    train_n = int(0.8 * n)
    
    X_tr, y_tr = t_X[idx[:train_n]], t_y[idx[:train_n]]
    X_te, y_te = t_X[idx[train_n:]], t_y[idx[train_n:]]
    
    probe = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for _ in range(150):
        opt.zero_grad()
        out = probe(X_tr)
        loss = loss_fn(out, y_tr)
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        logits_te = probe(X_te).squeeze(1)
        probs_te = torch.sigmoid(logits_te).numpy()
        preds_te = (probs_te >= 0.5).astype(int)
        y_te_np = y_te.squeeze(1).numpy().astype(int)
        
    auc = compute_roc_auc_np(y_te_np, probs_te)
    acc = float(np.mean(preds_te == y_te_np))
    return auc, acc


# -----------------------------------------------------------------------------
# 5. Diagnostic Evaluation Across Classes A-E
# -----------------------------------------------------------------------------

def evaluate_variant_on_diagnostic_dataset(
    variant_name: str,
    model_dir: str | Path,
    diag_dir: str | Path = "data/diagnostic_representation_dataset",
    is_temporal_variant: bool = False,
) -> Dict[str, Any]:
    mpath = Path(model_dir)
    norm = ObservationNormalizer.load_yaml(mpath / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(mpath / "config.yaml")
    model = CausalWorldModel(cfg)
    
    if is_temporal_variant:
        model.encoder = TemporalDifferenceEncoder(obs_dim=8, act_dim=4, latent_dim=cfg.model.latent_dim)
        
    ckpt = torch.load(mpath / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    diag_path = Path(diag_dir)
    classes = ["class_a_nominal", "class_b_high_safe", "class_c_boundary", "class_d_runaway", "class_e_acute_shocks"]
    
    results = {
        "variant": variant_name,
        "classes": {},
        "summary": {},
    }
    
    all_latents = []
    all_labels = []
    
    class_latents = {c: [] for c in classes}
    class_t_core_true = {c: [] for c in classes}
    class_t_core_rec = {c: [] for c in classes}
    class_shock_deltas = []
    
    with torch.no_grad():
        for cls_name in classes:
            c_dir = diag_path / cls_name
            npz_files = sorted(list(c_dir.glob("*.npz")))
            
            t_errors_all = []
            t_peak_errors = []
            mse_all = []
            
            for f in npz_files:
                d = np.load(f)
                obs_raw = np.asarray(d["observations"], dtype=np.float32) # (60, 8)
                act_raw = np.asarray(d["actions"], dtype=np.float32)      # (60, 4)
                mask = np.asarray(d["observation_mask"], dtype=bool)
                
                obs_norm = norm.normalize(obs_raw)
                if not isinstance(obs_norm, torch.Tensor):
                    obs_norm = torch.from_numpy(obs_norm)
                t_obs = obs_norm.float().unsqueeze(0)
                t_mask = torch.from_numpy(mask).bool().unsqueeze(0)
                t_act = torch.from_numpy(act_raw).float().unsqueeze(0)
                
                inputs = ModelInputs(t_obs, t_mask, t_act)
                lat_dist, _ = model.encode(inputs)
                z_seq = lat_dist.mean.squeeze(0).cpu().numpy() # (60, 64)
                
                rec_dist = model.decode(lat_dist.mean)
                rec_norm = rec_dist.mean.squeeze(0)
                rec_raw_t = norm.denormalize(rec_norm)
                rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
                
                # Metrics
                err_t = np.abs(obs_raw[:, 0] - rec_raw[:, 0])
                t_errors_all.append(err_t)
                
                peak_true = float(np.max(obs_raw[:, 0]))
                peak_rec = float(np.max(rec_raw[:, 0]))
                t_peak_errors.append(abs(peak_true - peak_rec))
                
                mse = float(np.mean((obs_raw - rec_raw)**2))
                mse_all.append(mse)
                
                class_latents[cls_name].append(z_seq[-1])
                class_t_core_true[cls_name].append(float(obs_raw[-1, 0]))
                class_t_core_rec[cls_name].append(float(rec_raw[-1, 0]))
                
                for t in range(60):
                    all_latents.append(z_seq[t])
                    all_labels.append(1 if obs_raw[t, 0] > 95.0 else 0)
                    
                if cls_name == "class_e_acute_shocks":
                    d_t_phys = obs_raw[1:, 0] - obs_raw[:-1, 0]
                    shock_idx = int(np.argmax(d_t_phys))
                    z_shock_jump = float(np.linalg.norm(z_seq[shock_idx+1] - z_seq[shock_idx]))
                    class_shock_deltas.append({
                        "delta_T": float(d_t_phys[shock_idx]),
                        "delta_z": float(z_shock_jump),
                    })
                    
            t_mae = float(np.mean(np.concatenate(t_errors_all)))
            t_peak_err = float(np.mean(t_peak_errors))
            tot_mse = float(np.mean(mse_all))
            
            results["classes"][cls_name] = {
                "t_core_mae": t_mae,
                "t_peak_error": t_peak_err,
                "total_mse": tot_mse,
                "mean_true_endpoint_T": float(np.mean(class_t_core_true[cls_name])),
                "mean_rec_endpoint_T": float(np.mean(class_t_core_rec[cls_name])),
            }
            
    # Centroid separation
    centroid_a = np.mean(class_latents["class_a_nominal"], axis=0)
    latent_separations = {}
    for cls_name in classes:
        cls_z = np.array(class_latents[cls_name])
        dist = np.mean([np.linalg.norm(z - centroid_a) for z in cls_z])
        latent_separations[cls_name] = float(dist)
    results["latent_separations_from_a"] = latent_separations
    
    # Shock sensitivity
    d_Ts = [x["delta_T"] for x in class_shock_deltas]
    d_zs = [x["delta_z"] for x in class_shock_deltas]
    corr = float(np.corrcoef(d_Ts, d_zs)[0, 1]) if len(d_Ts) > 1 else 0.0
    mean_shock_z = float(np.mean(d_zs))
    results["shock_sensitivity"] = {
        "mean_latent_shock_displacement": mean_shock_z,
        "correlation_deltaT_deltaZ": corr,
    }
    
    # Linear probe
    X = np.array(all_latents)
    y = np.array(all_labels)
    auc, acc = train_linear_probe(X, y)
    results["safety_probe"] = {
        "roc_auc": auc,
        "accuracy": acc,
    }
    
    return results


def main() -> None:
    print("=========================================================================")
    print("      TASK 5.9: SAFETY-CRITICAL STATE REPRESENTATION EXPERIMENT          ")
    print("=========================================================================")
    
    exp_dir = Path("artifacts/experiment_task_5_9")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train Experimental Variants
    v1_dir = train_experimental_variant("variant_1_reduced_kl", epochs=25)
    v2_dir = train_experimental_variant("variant_2_shock_weighted", epochs=25)
    v3_dir = train_experimental_variant("variant_3_temporal_shock", epochs=25)
    
    # 2. Evaluate All Variants
    print("\n--> Evaluating all 4 variants on 250-episode Diagnostic Dataset...")
    res_v0 = evaluate_variant_on_diagnostic_dataset("Variant 0 (Control: B004)", "artifacts/baseline_004")
    res_v1 = evaluate_variant_on_diagnostic_dataset("Variant 1 (Reduced KL: beta=1e-5)", v1_dir)
    res_v2 = evaluate_variant_on_diagnostic_dataset("Variant 2 (Shock-Weighted Recon)", v2_dir)
    res_v3 = evaluate_variant_on_diagnostic_dataset("Variant 3 (Temporal Shock Diff)", v3_dir, is_temporal_variant=True)
    
    master_results = {
        "variant_0_control": res_v0,
        "variant_1_reduced_kl": res_v1,
        "variant_2_shock_weighted": res_v2,
        "variant_3_temporal_shock": res_v3,
    }
    
    with open(exp_dir / "task_5_9_ablation_results.json", "w") as f:
        json.dump(master_results, f, indent=2)
        
    print(f"\nSaved complete ablation results to {exp_dir / 'task_5_9_ablation_results.json'}")
    
    # Print summary tables
    print("\n=== METRIC A: RECONSTRUCTION ERROR (T_core MAE °C) ===")
    print(f"{'Variant':<32} | {'Class A (Nom)':<13} | {'Class B (Safe)':<13} | {'Class C (Bound)':<14} | {'Class D (Runaway)':<16} | {'Class E (Shocks)':<15}")
    print("-" * 115)
    for v_key, r in master_results.items():
        vname = r["variant"]
        ca = r["classes"]["class_a_nominal"]["t_core_mae"]
        cb = r["classes"]["class_b_high_safe"]["t_core_mae"]
        cc = r["classes"]["class_c_boundary"]["t_core_mae"]
        cd = r["classes"]["class_d_runaway"]["t_core_mae"]
        ce = r["classes"]["class_e_acute_shocks"]["t_core_mae"]
        print(f"{vname:<32} | {ca:10.2f}°C  | {cb:10.2f}°C  | {cc:11.2f}°C  | {cd:13.2f}°C  | {ce:12.2f}°C")
        
    print("\n=== METRIC B: LATENT SEPARATION FROM CLASS A (Euclidean Distance) ===")
    print(f"{'Variant':<32} | {'Class B (Safe)':<13} | {'Class C (Bound)':<14} | {'Class D (Runaway)':<16} | {'Separation Ratio (D/B)':<20}")
    print("-" * 105)
    for v_key, r in master_results.items():
        vname = r["variant"]
        db = r["latent_separations_from_a"]["class_b_high_safe"]
        dc = r["latent_separations_from_a"]["class_c_boundary"]
        dd = r["latent_separations_from_a"]["class_d_runaway"]
        ratio = dd / max(db, 1e-4)
        print(f"{vname:<32} | {db:10.2f}    | {dc:11.2f}    | {dd:13.2f}    | {ratio:18.2f}x")

    print("\n=== METRIC C: TEMPORAL SHOCK SENSITIVITY (Class E) ===")
    print(f"{'Variant':<32} | {'Mean Latent Shock Displacement ||Delta z||':<40} | {'Corr(Delta T, Delta z)':<22}")
    print("-" * 100)
    for v_key, r in master_results.items():
        vname = r["variant"]
        disp = r["shock_sensitivity"]["mean_latent_shock_displacement"]
        corr = r["shock_sensitivity"]["correlation_deltaT_deltaZ"]
        print(f"{vname:<32} | {disp:38.4f} | {corr:20.4f}")

    print("\n=== METRIC D: SAFETY BOUNDARY LINEAR PROBE SEPARABILITY ===")
    print(f"{'Variant':<32} | {'ROC-AUC Score':<16} | {'Classification Accuracy':<24}")
    print("-" * 80)
    for v_key, r in master_results.items():
        vname = r["variant"]
        auc = r["safety_probe"]["roc_auc"]
        acc = r["safety_probe"]["accuracy"]
        print(f"{vname:<32} | {auc:14.4f} | {acc*100:21.2f}%")


if __name__ == "__main__":
    main()
