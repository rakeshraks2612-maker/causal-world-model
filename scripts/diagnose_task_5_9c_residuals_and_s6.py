"""Task 5.9C Diagnostic Script: Online Reconstruction Residual & S6-Configuration Diagnostic.

Experiments:
1. Build Residual Distribution across Classes A-E and S1-S6 for B004 and Variant 2.
2. Calibration and Separability of Online Reconstruction Residual (ROC-AUC, Precision, Recall, FPR, FNR).
3. Progressive Reconstruction of S6 Multivariate Configuration (A through F).
4. Counterfactual Context Test (S6 terminal observation across 6 different histories).
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.world_model.latent_state import LatentDistribution
from prism.training.normalization import ObservationNormalizer
from prism.dataset.decision_benchmark import LearnerDecisionScenario


def load_model(model_dir: str | Path) -> Tuple[CausalWorldModel, ObservationNormalizer, WorldModelConfig]:
    mpath = Path(model_dir)
    norm = ObservationNormalizer.load_yaml(mpath / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(mpath / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(mpath / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, norm, cfg


def compute_training_latent_reference(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    train_data_dir: str | Path = "data/excitation_dataset/learner/train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training set latents to compute ID reference mean and regularized inverse covariance."""
    files = sorted(list(Path(train_data_dir).glob("*.npz")))
    train_latents = []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            obs = d["observations"] # (T, 8)
            mask = d["observation_mask"] # (T, 8)
            act = d["actions"] # (T, 4)
            obs_norm = normalizer.normalize(obs)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            t_obs = obs_norm.float().unsqueeze(0)
            t_mask = torch.from_numpy(mask).bool().unsqueeze(0)
            t_act = torch.from_numpy(act).float().unsqueeze(0)
            inputs = ModelInputs(t_obs, t_mask, t_act)
            lat_dist, _ = model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0).cpu().numpy() # (T, 64)
            train_latents.append(z_seq)
            
    z_all = np.concatenate(train_latents, axis=0) # [N, 64]
    mu_id = np.mean(z_all, axis=0)
    cov_id = np.cov(z_all, rowvar=False)
    reg_cov = cov_id + 1e-4 * np.eye(z_all.shape[1])
    inv_cov_id = np.linalg.inv(reg_cov)
    return mu_id, inv_cov_id


def mahalanobis_dist(z: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    diff = z - mu
    d2 = float(diff @ inv_cov @ diff)
    return float(np.sqrt(max(0.0, d2)))


def compute_stats(arr: List[float] | np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


# =============================================================================
# EXPERIMENT 1: Build Residual Distribution
# =============================================================================
def run_experiment_1(
    model_name: str,
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    diag_dir: str = "data/diagnostic_representation_dataset",
    bench_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    classes = ["class_a_nominal", "class_b_high_safe", "class_c_boundary", "class_d_runaway", "class_e_acute_shocks"]
    diag_path = Path(diag_dir)
    bpath = Path(bench_dir)
    
    exp1_res = {
        "model": model_name,
        "classes": {},
        "scenarios": {},
    }
    
    with torch.no_grad():
        for cls_name in classes:
            c_dir = diag_path / cls_name
            npz_files = sorted(list(c_dir.glob("*.npz")))
            
            t_core_residuals = []
            obs_8d_norm_residuals = []
            
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
                rec_dist = model.decode(lat_dist.mean)
                
                rec_norm = rec_dist.mean.squeeze(0) # (60, 8)
                rec_raw_t = norm.denormalize(rec_norm)
                rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
                
                # Residuals over all timesteps
                r_T = np.abs(obs_raw[:, 0] - rec_raw[:, 0]) # (60,)
                r_8d = np.linalg.norm(obs_norm.numpy() - rec_norm.cpu().numpy(), axis=-1) # (60,)
                
                t_core_residuals.extend(r_T.tolist())
                obs_8d_norm_residuals.extend(r_8d.tolist())
                
            exp1_res["classes"][cls_name] = {
                "t_core_residual": compute_stats(t_core_residuals),
                "obs_8d_norm_residual": compute_stats(obs_8d_norm_residuals),
            }
            
        # Scenarios S1 to S6 at t*
        for s_idx in range(1, 7):
            s_id = f"scenario_0{s_idx}_{['do_nothing', 'valve', 'throttle', 'pump', 'combined', 'all_unsafe'][s_idx-1]}"
            s_file = bpath / "scenarios" / s_id / "learner.npz"
            learner = LearnerDecisionScenario.load_npz(s_file)
            
            obs_raw = learner.historical_observations
            act_raw = learner.historical_actions
            mask = learner.historical_observation_mask
            t_star = learner.intervention_time
            
            obs_norm = norm.normalize(obs_raw)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            t_obs = obs_norm.float().unsqueeze(0)
            t_mask = torch.from_numpy(mask).bool().unsqueeze(0)
            t_act = torch.from_numpy(act_raw).float().unsqueeze(0)
            
            inputs = ModelInputs(t_obs, t_mask, t_act)
            lat_dist, _ = model.encode(inputs)
            rec_dist = model.decode(lat_dist.mean)
            
            rec_norm = rec_dist.mean.squeeze(0)
            rec_raw_t = norm.denormalize(rec_norm)
            rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
            
            r_T_star = float(np.abs(obs_raw[t_star, 0] - rec_raw[t_star, 0]))
            r_8d_star = float(np.linalg.norm(obs_norm.numpy()[t_star] - rec_norm.cpu().numpy()[t_star]))
            
            exp1_res["scenarios"][f"S{s_idx}"] = {
                "scenario_id": s_id,
                "t_star": t_star,
                "true_t_core": float(obs_raw[t_star, 0]),
                "rec_t_core": float(rec_raw[t_star, 0]),
                "t_core_residual": r_T_star,
                "obs_8d_norm_residual": r_8d_star,
            }
            
    return exp1_res


# =============================================================================
# EXPERIMENT 2: Separability & Calibration of Residual Threshold
# =============================================================================
def run_experiment_2(
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    diag_dir: str = "data/diagnostic_representation_dataset",
    bench_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    diag_path = Path(diag_dir)
    classes = ["class_a_nominal", "class_b_high_safe", "class_c_boundary", "class_d_runaway", "class_e_acute_shocks"]
    
    samples = [] # (residual_T, residual_8d, is_unsafe, cls_name)
    
    with torch.no_grad():
        for cls_name in classes:
            c_dir = diag_path / cls_name
            npz_files = sorted(list(c_dir.glob("*.npz")))
            
            for f_idx, f in enumerate(npz_files):
                d = np.load(f)
                obs_raw = np.asarray(d["observations"], dtype=np.float32)
                act_raw = np.asarray(d["actions"], dtype=np.float32)
                mask = np.asarray(d["observation_mask"], dtype=bool)
                
                obs_norm = norm.normalize(obs_raw)
                if not isinstance(obs_norm, torch.Tensor):
                    obs_norm = torch.from_numpy(obs_norm)
                t_obs = obs_norm.float().unsqueeze(0)
                t_mask = torch.from_numpy(mask).bool().unsqueeze(0)
                t_act = torch.from_numpy(act_raw).float().unsqueeze(0)
                
                inputs = ModelInputs(t_obs, t_mask, t_act)
                lat_dist, _ = model.encode(inputs)
                rec_dist = model.decode(lat_dist.mean)
                
                rec_norm = rec_dist.mean.squeeze(0)
                rec_raw_t = norm.denormalize(rec_norm)
                rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
                
                # Check terminal step and full episode
                for t in range(60):
                    r_T = float(np.abs(obs_raw[t, 0] - rec_raw[t, 0]))
                    r_8d = float(np.linalg.norm(obs_norm.numpy()[t] - rec_norm.cpu().numpy()[t]))
                    # Positive class: unsafe / runaway / shock (T > 95°C or Class D/E shock timestep)
                    is_unsafe = 1 if (obs_raw[t, 0] > 95.0 or (cls_name in ["class_d_runaway", "class_e_acute_shocks"] and obs_raw[t, 0] > 94.0)) else 0
                    samples.append({
                        "file_idx": f_idx,
                        "class": cls_name,
                        "t": t,
                        "r_T": r_T,
                        "r_8d": r_8d,
                        "true_T": float(obs_raw[t, 0]),
                        "is_unsafe": is_unsafe,
                    })

    # Split files into calibration (50%) and evaluation (50%)
    calib_samples = [s for s in samples if s["file_idx"] < 25]
    eval_samples = [s for s in samples if s["file_idx"] >= 25]
    
    # Compute ROC-AUC on eval set for r_T
    y_eval = np.array([s["is_unsafe"] for s in eval_samples])
    scores_eval_T = np.array([s["r_T"] for s in eval_samples])
    scores_eval_8d = np.array([s["r_8d"] for s in eval_samples])
    
    def calc_auc(y_true, scores):
        desc_indices = np.argsort(scores, kind="mergesort")[::-1]
        y_true = y_true[desc_indices]
        scores = scores[desc_indices]
        distinct_value_indices = np.where(np.diff(scores))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
        
    auc_T = calc_auc(y_eval, scores_eval_T)
    auc_8d = calc_auc(y_eval, scores_eval_8d)
    
    # Pick threshold R* on calibration set targeting FPR <= 0.02 (2% false alarm on safe states)
    calib_safe_r_T = [s["r_T"] for s in calib_samples if s["is_unsafe"] == 0]
    thresh_T_98 = float(np.percentile(calib_safe_r_T, 98.0))
    thresh_T_95 = float(np.percentile(calib_safe_r_T, 95.0))
    
    calib_safe_r_8d = [s["r_8d"] for s in calib_samples if s["is_unsafe"] == 0]
    thresh_8d_98 = float(np.percentile(calib_safe_r_8d, 98.0))
    
    # Evaluate calibrated thresholds on eval set
    def eval_threshold(thresh, scores, y_true):
        preds = (scores >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        return {
            "threshold": thresh,
            "precision": float(prec),
            "recall": float(rec),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        
    res_thresh_98 = eval_threshold(thresh_T_98, scores_eval_T, y_eval)
    res_thresh_95 = eval_threshold(thresh_T_95, scores_eval_T, y_eval)
    res_8d_98 = eval_threshold(thresh_8d_98, scores_eval_8d, y_eval)
    
    # Test on Benchmark Scenarios
    bpath = Path(bench_dir)
    scen_results = {}
    for s_idx in range(1, 7):
        s_id = f"scenario_0{s_idx}_{['do_nothing', 'valve', 'throttle', 'pump', 'combined', 'all_unsafe'][s_idx-1]}"
        s_file = bpath / "scenarios" / s_id / "learner.npz"
        learner = LearnerDecisionScenario.load_npz(s_file)
        
        obs_raw = learner.historical_observations
        act_raw = learner.historical_actions
        mask = learner.historical_observation_mask
        t_star = learner.intervention_time
        
        obs_norm = norm.normalize(obs_raw)
        if not isinstance(obs_norm, torch.Tensor):
            obs_norm = torch.from_numpy(obs_norm)
        t_obs = obs_norm.float().unsqueeze(0)
        t_mask = torch.from_numpy(mask).bool().unsqueeze(0)
        t_act = torch.from_numpy(act_raw).float().unsqueeze(0)
        
        inputs = ModelInputs(t_obs, t_mask, t_act)
        with torch.no_grad():
            lat_dist, _ = model.encode(inputs)
            rec_dist = model.decode(lat_dist.mean)
        
        rec_norm = rec_dist.mean.squeeze(0)
        rec_raw_t = norm.denormalize(rec_norm)
        rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
        
        r_T_star = float(np.abs(obs_raw[t_star, 0] - rec_raw[t_star, 0]))
        r_8d_star = float(np.linalg.norm(obs_norm.numpy()[t_star] - rec_norm.cpu().numpy()[t_star]))
        
        flagged_T_98 = bool(r_T_star >= thresh_T_98)
        flagged_8d_98 = bool(r_8d_star >= thresh_8d_98)
        
        scen_results[f"S{s_idx}"] = {
            "scenario_id": s_id,
            "true_T": float(obs_raw[t_star, 0]),
            "rec_T": float(rec_raw[t_star, 0]),
            "r_T": r_T_star,
            "r_8d": r_8d_star,
            "flagged_unsafe_T_98": flagged_T_98,
            "flagged_unsafe_8d_98": flagged_8d_98,
        }
        
    return {
        "roc_auc_t_core": auc_T,
        "roc_auc_8d_norm": auc_8d,
        "calibration_results": {
            "thresh_T_p98": res_thresh_98,
            "thresh_T_p95": res_thresh_95,
            "thresh_8d_p98": res_8d_98,
        },
        "benchmark_scenarios_evaluation": scen_results,
    }


# =============================================================================
# EXPERIMENT 3: Progressive Reconstruction of S6 Multivariate Configuration
# =============================================================================
def run_experiment_3(
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    mu_id: np.ndarray,
    inv_cov_id: np.ndarray,
    bench_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    """Reconstruct S6 progressively from Config A to Config F."""
    # Load exact S6 learner episode
    s6_path = Path(bench_dir) / "scenarios" / "scenario_06_all_unsafe" / "learner.npz"
    s6_data = LearnerDecisionScenario.load_npz(s6_path)
    s6_obs = np.array(s6_data.historical_observations, copy=True) # (31, 8)
    s6_act = np.array(s6_data.historical_actions, copy=True)      # (31, 4)
    
    # Equilibrium nominal state: 89°C baseline across all 31 timesteps
    # Extract nominal vector from t=28 of S6 (before shock)
    nom_obs_vec = s6_obs[28].copy()
    nom_act_vec = s6_act[28].copy()
    
    configs = {}
    
    # Config A: 89°C baseline (all channels nominal steady state)
    obs_A = np.tile(nom_obs_vec, (31, 1))
    act_A = np.tile(nom_act_vec, (31, 1))
    configs["Config_A_89C_Baseline"] = (obs_A, act_A, "89°C nominal equilibrium across all channels")
    
    # Config B: 89°C baseline + 120°C thermal shock at terminal step t=30
    obs_B = obs_A.copy()
    obs_B[30, 0] = 120.37
    configs["Config_B_89C_Plus_120C_Shock"] = (obs_B, act_A, "89°C baseline + 120°C single-channel thermal shock at t=30")
    
    # Config C: 89°C baseline + extreme valve closure (valve closed 0%, flow drops to minimum)
    obs_C = obs_A.copy()
    act_C = act_A.copy()
    obs_C[:, 5] = 0.0 # Valve position = 0%
    obs_C[:, 3] = 2.0 # Coolant flow drops to near 0
    act_C[:, 0] = 0.0 # Valve command = 0%
    configs["Config_C_89C_Plus_Valve_Closed"] = (obs_C, act_C, "89°C baseline + valve 0% closed / flow choked")
    
    # Config D: 89°C baseline + extreme valve closure + low load
    obs_D = obs_C.copy()
    act_D = act_C.copy()
    obs_D[:, 4] = 15.0 # CPU load = 15%
    obs_D[:, 7] = 0.8  # Power draw drops to 0.8 kW
    act_D[:, 1] = 15.0 # Throttle command = 15%
    configs["Config_D_89C_Plus_Valve_Closed_Low_Load"] = (obs_D, act_D, "89°C baseline + valve closed + low workload")
    
    # Config E: 89°C baseline + extreme valve closure + low load + 120°C shock
    obs_E = obs_D.copy()
    obs_E[30, 0] = 120.37
    configs["Config_E_Valve_Closed_Low_Load_Plus_120C_Shock"] = (obs_E, act_D, "Valve closed + low load + 120°C shock at t=30")
    
    # Config F: Exact Full S6 observation/action history
    configs["Config_F_Full_S6_History"] = (s6_obs, s6_act, "Exact Full S6 observation and action history")
    
    results = {}
    
    # Get reference nominal latent z_nom
    with torch.no_grad():
        obs_norm_A = norm.normalize(obs_A)
        if not isinstance(obs_norm_A, torch.Tensor):
            obs_norm_A = torch.from_numpy(obs_norm_A)
        inputs_A = ModelInputs(obs_norm_A.float().unsqueeze(0), torch.ones_like(obs_norm_A).bool().unsqueeze(0), torch.from_numpy(act_A).float().unsqueeze(0))
        lat_dist_A, _ = model.encode(inputs_A)
        z_nom_t_star = lat_dist_A.mean.squeeze(0)[30].cpu().numpy()
        
    for cfg_key, (obs_cfg, act_cfg, desc) in configs.items():
        with torch.no_grad():
            obs_norm = norm.normalize(obs_cfg)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            t_obs = obs_norm.float().unsqueeze(0)
            t_mask = torch.ones_like(t_obs).bool()
            t_act = torch.from_numpy(act_cfg).float().unsqueeze(0)
            
            inputs = ModelInputs(t_obs, t_mask, t_act)
            lat_dist, _ = model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0) # (31, 64)
            z_logvar_seq = lat_dist.logvar.squeeze(0) # (31, 64)
            
            rec_dist = model.decode(lat_dist.mean)
            rec_norm = rec_dist.mean.squeeze(0)
            rec_raw_t = norm.denormalize(rec_norm)
            rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
            
            # Transition forward from z_30
            z_30 = z_seq[30].unsqueeze(0) # [1, 64]
            act_30 = t_act[:, 30]          # [1, 4]
            trans_dist = model.transition(z_30, act_30)
            pred_rec = model.decode(trans_dist.mean)
            pred_raw_t = norm.denormalize(pred_rec.mean.squeeze(0))
            pred_raw = pred_raw_t.cpu().numpy() if isinstance(pred_raw_t, torch.Tensor) else np.asarray(pred_raw_t)
            
            z_30_np = z_seq[30].cpu().numpy()
            z_29_np = z_seq[29].cpu().numpy()
            
            # Latent metrics
            z_norm = float(np.linalg.norm(z_30_np))
            z_disp_step = float(np.linalg.norm(z_30_np - z_29_np))
            z_disp_nom = float(np.linalg.norm(z_30_np - z_nom_t_star))
            d_mah = mahalanobis_dist(z_30_np, mu_id, inv_cov_id)
            
            # Uncertainty metrics
            aleatoric_std = float(torch.exp(0.5 * z_logvar_seq[30]).mean().item())
            
            # Reconstructions
            true_T = float(obs_cfg[30, 0])
            rec_T = float(rec_raw[30, 0])
            r_T = abs(true_T - rec_T)
            r_8d = float(np.linalg.norm(obs_norm.numpy()[30] - rec_norm.cpu().numpy()[30]))
            
            results[cfg_key] = {
                "description": desc,
                "true_T_core": true_T,
                "rec_T_core": rec_T,
                "pred_forward_T_core": float(pred_raw[0]) if pred_raw.ndim == 1 else float(pred_raw[0, 0]),
                "residual_T_core": r_T,
                "residual_8d_norm": r_8d,
                "latent_norm": z_norm,
                "latent_mahalanobis_d": d_mah,
                "latent_step_displacement": z_disp_step,
                "latent_displacement_from_nom": z_disp_nom,
                "aleatoric_sigma_latent": aleatoric_std,
            }
            
    return results


# =============================================================================
# EXPERIMENT 4: Counterfactual Context Test
# =============================================================================
def run_experiment_4(
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    mu_id: np.ndarray,
    inv_cov_id: np.ndarray,
    bench_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    """Encode the exact terminal S6 observation (t=30) after 6 distinct 30-step historical contexts."""
    # Load exact S6 learner episode
    s6_path = Path(bench_dir) / "scenarios" / "scenario_06_all_unsafe" / "learner.npz"
    s6_data = LearnerDecisionScenario.load_npz(s6_path)
    s6_obs = np.array(s6_data.historical_observations, copy=True) # (31, 8)
    s6_act = np.array(s6_data.historical_actions, copy=True)      # (31, 4)
    
    # Exact terminal observation vector of S6 (t=30, 120.37°C)
    s6_terminal_obs = s6_obs[30].copy()
    s6_terminal_act = s6_act[30].copy()
    
    contexts = {}
    
    # 1. Normal History (all channels at nominal ~75-80°C)
    nom_vec = np.array([75.0, 50.0, 3.0, 30.0, 60.0, 60.0, 8.0, 2.5], dtype=np.float32)
    act_nom = np.array([60.0, 100.0, 2.0, 0.0], dtype=np.float32)
    obs_1 = np.tile(nom_vec, (31, 1))
    act_1 = np.tile(act_nom, (31, 1))
    obs_1[30] = s6_terminal_obs
    act_1[30] = s6_terminal_act
    contexts["1_Normal_History"] = (obs_1, act_1, "Nominal steady-state (T_core ≈ 75°C, load 60%, valve 60%)")
    
    # 2. High-Temperature History (running warm at high-safe ~92°C)
    high_t_vec = np.array([92.0, 65.0, 3.2, 45.0, 80.0, 90.0, 10.0, 2.8], dtype=np.float32)
    obs_2 = np.tile(high_t_vec, (31, 1))
    act_2 = np.tile(act_nom, (31, 1))
    obs_2[30] = s6_terminal_obs
    act_2[30] = s6_terminal_act
    contexts["2_High_Temperature_History"] = (obs_2, act_2, "High-safe thermal regime (T_core ≈ 92°C, high load 80%)")
    
    # 3. Low-Load History (workload low, valve nominal)
    low_load_vec = np.array([65.0, 40.0, 3.0, 25.0, 20.0, 60.0, 6.0, 1.2], dtype=np.float32)
    act_low_load = np.array([60.0, 20.0, 2.0, 0.0], dtype=np.float32)
    obs_3 = np.tile(low_load_vec, (31, 1))
    act_3 = np.tile(act_low_load, (31, 1))
    obs_3[30] = s6_terminal_obs
    act_3[30] = s6_terminal_act
    contexts["3_Low_Load_History"] = (obs_3, act_3, "Low-load regime (T_core ≈ 65°C, load 20%)")
    
    # 4. Valve-Closed History (valve closed, load nominal)
    valve_closed_vec = np.array([85.0, 60.0, 1.5, 5.0, 60.0, 5.0, 8.0, 2.5], dtype=np.float32)
    act_vc = np.array([5.0, 100.0, 2.0, 0.0], dtype=np.float32)
    obs_4 = np.tile(valve_closed_vec, (31, 1))
    act_4 = np.tile(act_vc, (31, 1))
    obs_4[30] = s6_terminal_obs
    act_4[30] = s6_terminal_act
    contexts["4_Valve_Closed_History"] = (obs_4, act_4, "Valve-closed bottleneck (valve 5%, flow 5 L/min)")
    
    # 5. Valve-Closed + Low-Load History
    vc_ll_vec = np.array([78.0, 55.0, 1.5, 5.0, 20.0, 5.0, 6.0, 1.2], dtype=np.float32)
    act_vc_ll = np.array([5.0, 20.0, 2.0, 0.0], dtype=np.float32)
    obs_5 = np.tile(vc_ll_vec, (31, 1))
    act_5 = np.tile(act_vc_ll, (31, 1))
    obs_5[30] = s6_terminal_obs
    act_5[30] = s6_terminal_act
    contexts["5_Valve_Closed_Plus_Low_Load_History"] = (obs_5, act_5, "Valve closed (5%) + low load (20%)")
    
    # 6. Exact S6 History
    obs_6 = s6_obs.copy()
    act_6 = s6_act.copy()
    contexts["6_Exact_S6_History"] = (obs_6, act_6, "Exact 30-step historical trajectory from Scenario 6")
    
    results = {}
    
    for ctx_key, (obs_ctx, act_ctx, desc) in contexts.items():
        with torch.no_grad():
            obs_norm = norm.normalize(obs_ctx)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            t_obs = obs_norm.float().unsqueeze(0)
            t_mask = torch.ones_like(t_obs).bool()
            t_act = torch.from_numpy(act_ctx).float().unsqueeze(0)
            
            inputs = ModelInputs(t_obs, t_mask, t_act)
            lat_dist, _ = model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0)
            
            rec_dist = model.decode(lat_dist.mean)
            rec_norm = rec_dist.mean.squeeze(0)
            rec_raw_t = norm.denormalize(rec_norm)
            rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
            
            z_30_np = z_seq[30].cpu().numpy()
            z_29_np = z_seq[29].cpu().numpy()
            
            z_norm = float(np.linalg.norm(z_30_np))
            z_disp_step = float(np.linalg.norm(z_30_np - z_29_np))
            d_mah = mahalanobis_dist(z_30_np, mu_id, inv_cov_id)
            
            rec_T = float(rec_raw[30, 0])
            r_T = abs(120.37 - rec_T)
            r_8d = float(np.linalg.norm(obs_norm.numpy()[30] - rec_norm.cpu().numpy()[30]))
            
            results[ctx_key] = {
                "context_description": desc,
                "history_T_before_shock": float(obs_ctx[29, 0]),
                "true_T_shock": 120.37,
                "rec_T_shock": rec_T,
                "residual_T_core": r_T,
                "residual_8d_norm": r_8d,
                "latent_norm": z_norm,
                "latent_mahalanobis_d": d_mah,
                "latent_step_displacement": z_disp_step,
            }
            
    return results


def main() -> None:
    print("=" * 100)
    print("   TASK 5.9C: ONLINE RECONSTRUCTION RESIDUAL & S6-CONFIGURATION DIAGNOSTIC")
    print("=" * 100)
    
    out_dir = Path("artifacts/diagnosis/task_5_9c")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load B004 (Control) and Variant 2 (Candidate)
    print("\n--> Loading frozen baseline_004 and Variant 2 models...")
    model_b004, norm_b004, cfg_b004 = load_model("artifacts/baseline_004")
    model_v2, norm_v2, cfg_v2 = load_model("artifacts/experiment_task_5_9/variant_2_shock_weighted")
    
    print("--> Computing training latent reference distributions (mu_ID, Sigma_ID)...")
    mu_b004, inv_cov_b004 = compute_training_latent_reference(model_b004, norm_b004)
    mu_v2, inv_cov_v2 = compute_training_latent_reference(model_v2, norm_v2)
    
    # -------------------------------------------------------------------------
    # Experiment 1: Residual Distribution
    # -------------------------------------------------------------------------
    print("\n[Experiment 1] Building Residual Distributions across Classes A-E and Scenarios S1-S6...")
    exp1_b004 = run_experiment_1("B004 (Control)", model_b004, norm_b004)
    exp1_v2 = run_experiment_1("Variant 2 (Shock-Weighted)", model_v2, norm_v2)
    
    # -------------------------------------------------------------------------
    # Experiment 2: Separability & Threshold Calibration
    # -------------------------------------------------------------------------
    print("\n[Experiment 2] Running Separability & Calibration on Held-out Set...")
    exp2_b004 = run_experiment_2(model_b004, norm_b004)
    exp2_v2 = run_experiment_2(model_v2, norm_v2)
    
    # -------------------------------------------------------------------------
    # Experiment 3: Progressive S6 Configuration (A through F)
    # -------------------------------------------------------------------------
    print("\n[Experiment 3] Running Progressive S6 Multivariate Configuration Reconstruction...")
    exp3_b004 = run_experiment_3(model_b004, norm_b004, mu_b004, inv_cov_b004)
    exp3_v2 = run_experiment_3(model_v2, norm_v2, mu_v2, inv_cov_v2)
    
    # -------------------------------------------------------------------------
    # Experiment 4: Counterfactual Context Test
    # -------------------------------------------------------------------------
    print("\n[Experiment 4] Running Counterfactual Context Test on Terminal S6 Observation...")
    exp4_b004 = run_experiment_4(model_b004, norm_b004, mu_b004, inv_cov_b004)
    exp4_v2 = run_experiment_4(model_v2, norm_v2, mu_v2, inv_cov_v2)
    
    master_report = {
        "experiment_1_residual_distributions": {
            "b004_control": exp1_b004,
            "variant_2_shock_weighted": exp1_v2,
        },
        "experiment_2_separability_and_calibration": {
            "b004_control": exp2_b004,
            "variant_2_shock_weighted": exp2_v2,
        },
        "experiment_3_progressive_s6_reconstruction": {
            "b004_control": exp3_b004,
            "variant_2_shock_weighted": exp3_v2,
        },
        "experiment_4_counterfactual_context_test": {
            "b004_control": exp4_b004,
            "variant_2_shock_weighted": exp4_v2,
        },
    }
    
    json_path = out_dir / "task_5_9c_diagnostic_results.json"
    with open(json_path, "w") as f:
        json.dump(master_report, f, indent=2)
    print(f"\nSaved complete diagnostic results to {json_path}")
    
    # Print formatted tables
    print("\n" + "=" * 105)
    print("EXPERIMENT 1: RESIDUAL DISTRIBUTION SUMMARY (T_core Residual |T - T_hat| °C)")
    print("=" * 105)
    print(f"{'Regime / Scenario':<25} | {'B004 Mean':<11} | {'B004 P95':<11} | {'B004 Max':<11} | {'V2 Mean':<11} | {'V2 P95':<11} | {'V2 Max':<11}")
    print("-" * 105)
    for cls_name in ["class_a_nominal", "class_b_high_safe", "class_c_boundary", "class_d_runaway", "class_e_acute_shocks"]:
        b_s = exp1_b004["classes"][cls_name]["t_core_residual"]
        v_s = exp1_v2["classes"][cls_name]["t_core_residual"]
        c_label = cls_name.replace("class_", "Class ").replace("_", " ").title()
        print(f"{c_label:<25} | {b_s['mean']:9.2f}°C | {b_s['p95']:9.2f}°C | {b_s['max']:9.2f}°C | {v_s['mean']:9.2f}°C | {v_s['p95']:9.2f}°C | {v_s['max']:9.2f}°C")
    print("-" * 105)
    for s_id in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        b_s = exp1_b004["scenarios"][s_id]
        v_s = exp1_v2["scenarios"][s_id]
        print(f"Scenario {s_id} (t*)           | {b_s['t_core_residual']:9.2f}°C | {'-':>11} | {b_s['t_core_residual']:9.2f}°C | {v_s['t_core_residual']:9.2f}°C | {'-':>11} | {v_s['t_core_residual']:9.2f}°C")
        
    print("\n" + "=" * 105)
    print("EXPERIMENT 3: PROGRESSIVE S6 CONFIGURATION (B004 Control vs Variant 2)")
    print("=" * 105)
    print(f"{'Configuration':<45} | {'True T':<8} | {'B004 Rec':<10} | {'B004 Res':<10} | {'B004 d_M':<10} | {'V2 Rec':<10} | {'V2 Res':<10} | {'V2 d_M':<10}")
    print("-" * 125)
    for cfg_key in exp3_b004.keys():
        b_c = exp3_b004[cfg_key]
        v_c = exp3_v2[cfg_key]
        lbl = cfg_key.replace("Config_", "").replace("_", " ")
        print(f"{lbl:<45} | {b_c['true_T_core']:6.2f}°C | {b_c['rec_T_core']:8.2f}°C | {b_c['residual_T_core']:8.2f}°C | {b_c['latent_mahalanobis_d']:8.2f} | {v_c['rec_T_core']:8.2f}°C | {v_c['residual_T_core']:8.2f}°C | {v_c['latent_mahalanobis_d']:8.2f}")

    print("\n" + "=" * 105)
    print("EXPERIMENT 4: COUNTERFACTUAL CONTEXT TEST (S6 Terminal Observation = 120.37°C)")
    print("=" * 105)
    print(f"{'Context Preceding 120.37°C Shock':<45} | {'B004 Rec T':<12} | {'B004 Res':<10} | {'B004 ||Δz||':<12} | {'V2 Rec T':<12} | {'V2 Res':<10} | {'V2 ||Δz||':<12}")
    print("-" * 125)
    for ctx_key in exp4_b004.keys():
        b_ctx = exp4_b004[ctx_key]
        v_ctx = exp4_v2[ctx_key]
        lbl = ctx_key.replace("_", " ")
        print(f"{lbl:<45} | {b_ctx['rec_T_shock']:10.2f}°C | {b_ctx['residual_T_core']:8.2f}°C | {b_ctx['latent_step_displacement']:10.4f} | {v_ctx['rec_T_shock']:10.2f}°C | {v_ctx['residual_T_core']:8.2f}°C | {v_ctx['latent_step_displacement']:10.4f}")


if __name__ == "__main__":
    main()
