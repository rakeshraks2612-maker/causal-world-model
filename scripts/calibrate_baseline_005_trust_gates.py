"""Calibration Pipeline for Baseline 005 Trust & Anomaly Gates (Task 5.10E).

Calibrates:
1. Training reference distribution (mu_ID, Sigma_ID) across all 18,000 steps of D_unified
2. Residual thresholds (R*_T_P98, R*_8D_P98) on validation partition to guarantee FPR <= 2.0%
3. Novelty threshold (tau_novelty = 15.0)
Saves calibrated parameters to artifacts/baseline_005/trust_calibration.json
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch

from prism.dataset.schema import LearnerEpisode
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.training.normalization import ObservationNormalizer


def compute_training_latent_reference(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    train_dir: str | Path = "data/unified_dataset/learner/train",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all training latents to compute ID reference mean and regularized inverse covariance."""
    files = sorted(list(Path(train_dir).glob("*.npz")))
    train_latents = []
    
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            obs = d["observations"]
            mask = d["observation_mask"]
            act = d["actions"]
            
            obs_norm = normalizer.normalize(obs)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
                
            inputs = ModelInputs(
                obs_norm.float().unsqueeze(0),
                torch.from_numpy(mask).bool().unsqueeze(0),
                torch.from_numpy(act).float().unsqueeze(0),
            )
            lat_dist, _ = model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0).cpu().numpy()
            train_latents.append(z_seq)
            
    z_all = np.concatenate(train_latents, axis=0) # [N, 64]
    mu_id = np.mean(z_all, axis=0)
    cov_id = np.cov(z_all, rowvar=False)
    reg_cov = cov_id + 1e-4 * np.eye(z_all.shape[1])
    inv_cov_id = np.linalg.inv(reg_cov)
    
    return mu_id, inv_cov_id


def calibrate_residual_thresholds(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    val_dir: str | Path = "data/unified_dataset/learner/validation",
    target_fpr: float = 0.02,
) -> dict[str, Any]:
    """Calculate P98 / P95 residual percentiles on validation split to establish calibrated thresholds."""
    files = sorted(list(Path(val_dir).glob("*.npz")))
    r_T_all = []
    r_8D_all = []
    
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            obs_raw = d["observations"]
            mask = d["observation_mask"]
            act = d["actions"]
            
            obs_norm = normalizer.normalize(obs_raw)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
                
            inputs = ModelInputs(
                obs_norm.float().unsqueeze(0),
                torch.from_numpy(mask).bool().unsqueeze(0),
                torch.from_numpy(act).float().unsqueeze(0),
            )
            lat_dist, _ = model.encode(inputs)
            rec_dist = model.decode(lat_dist.mean)
            
            rec_norm = rec_dist.mean.squeeze(0)
            rec_raw_t = normalizer.denormalize(rec_norm)
            rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
            
            r_T = np.abs(obs_raw[:, 0] - rec_raw[:, 0])
            r_8D = np.linalg.norm(obs_norm.numpy() - rec_norm.cpu().numpy(), axis=-1)
            
            r_T_all.extend(r_T.tolist())
            r_8D_all.extend(r_8D.tolist())
            
    p98_T = float(np.percentile(r_T_all, 100.0 * (1.0 - target_fpr)))
    p95_T = float(np.percentile(r_T_all, 95.0))
    p98_8D = float(np.percentile(r_8D_all, 100.0 * (1.0 - target_fpr)))
    p95_8D = float(np.percentile(r_8D_all, 95.0))
    
    return {
        "tau_residual_t_p98": p98_T,
        "tau_residual_t_p95": p95_T,
        "tau_residual_8d_p98": p98_8D,
        "tau_residual_8d_p95": p95_8D,
        "validation_samples_count": len(r_T_all),
        "mean_val_residual_T": float(np.mean(r_T_all)),
        "max_val_residual_T": float(np.max(r_T_all)),
    }


def main() -> None:
    print("=" * 80)
    print("      CALIBRATING BASELINE 005 TRUST & ANOMALY GATES (TASK 5.10E)        ")
    print("=" * 80)
    
    model_dir = Path("artifacts/baseline_005")
    if not (model_dir / "best.pt").exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_dir / 'best.pt'}")
        
    norm = ObservationNormalizer.load_yaml(model_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("--> Computing training latent reference (mu_ID, Sigma_ID)...")
    mu_id, inv_cov_id = compute_training_latent_reference(model, norm)
    
    print("--> Calibrating validation residual thresholds (FPR <= 2.0%)...")
    res_calib = calibrate_residual_thresholds(model, norm)
    
    calibration_manifest = {
        "model_name": "baseline_005",
        "tau_novelty_mahalanobis": 15.0,
        "tau_residual_t_core": res_calib["tau_residual_t_p98"],
        "tau_residual_8d_norm": res_calib["tau_residual_8d_p98"],
        "calibration_details": res_calib,
        "mu_id": mu_id.tolist(),
        "inv_cov_id": inv_cov_id.tolist(),
    }
    
    out_file = model_dir / "trust_calibration.json"
    with open(out_file, "w") as f:
        json.dump(calibration_manifest, f, indent=2)
        
    print(f"\nSaved trust calibration parameters to {out_file}")
    print(f"Calibrated Thresholds:")
    print(f"  - Core Temp Residual Threshold (P98): {res_calib['tau_residual_t_p98']:.2f}°C")
    print(f"  - 8D Normalized Residual Threshold (P98): {res_calib['tau_residual_8d_p98']:.4f}")
    print(f"  - Latent Novelty Threshold (Mahalanobis): 15.00")
    print(f"  - Val Mean Residual T: {res_calib['mean_val_residual_T']:.2f}°C (Max: {res_calib['max_val_residual_T']:.2f}°C)")


if __name__ == "__main__":
    main()
