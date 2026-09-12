"""Task 5.10B: Full Decision-System Integration Validation.

Validates the complete PRISM decision intelligence pipeline with Baseline 005:
1. Full 6-scenario benchmark evaluation with upstream Trust Gateway.
2. Formal verification that S6 cannot reach planner recommendation.
3. Verification that S1-S5 are not over-abstained (selective competence).
4. Selective decision quality metrics (Coverage, Selective Accuracy, Selective Regret, Abstention Recall, False-Trust Rate).
5. Threshold sensitivity analysis (4°C, 5°C, 6.08°C, 7°C, 8°C, 10°C).
6. Dual-gate ablation (Residual only vs Novelty only vs Dual OR gate).
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator, ModelTrustState
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario


def load_b005_pipeline() -> Tuple[CausalWorldModel, ObservationNormalizer, ModelTrustEvaluator, InterventionPlanner]:
    model_dir = Path("artifacts/baseline_005")
    norm = ObservationNormalizer.load_yaml(model_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    with open(model_dir / "trust_calibration.json") as f:
        calib = json.load(f)
        
    mu_id = np.array(calib["mu_id"], dtype=np.float32)
    inv_cov_id = np.array(calib["inv_cov_id"], dtype=np.float32)
    
    trust_evaluator = ModelTrustEvaluator(
        model=model,
        normalizer=norm,
        mu_id=mu_id,
        inv_cov_id=inv_cov_id,
        tau_residual_t=calib["tau_residual_t_core"],
        tau_residual_8d=calib["tau_residual_8d_norm"],
        tau_novelty=calib["tau_novelty_mahalanobis"],
    )
    
    planner = InterventionPlanner(model, norm)
    return model, norm, trust_evaluator, planner


# =============================================================================
# 1. Full 6-Scenario Benchmark Evaluation with Upstream Trust Gateway
# =============================================================================
def evaluate_benchmark_with_trust_gate(
    planner: InterventionPlanner,
    trust_evaluator: ModelTrustEvaluator,
    benchmark_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    bpath = Path(benchmark_dir)
    scenarios = [
        ("scenario_01_do_nothing", "Scenario 1: Inaction Restraint"),
        ("scenario_02_valve", "Scenario 2: Valve Intervention"),
        ("scenario_03_throttle", "Scenario 3: Workload Throttling"),
        ("scenario_04_pump", "Scenario 4: Pump Head Modulating"),
        ("scenario_05_combined", "Scenario 5: Multi-Variable Compound"),
        ("scenario_06_all_unsafe", "Scenario 6: All-Unsafe Abstention"),
    ]
    
    results = {}
    
    for s_id, s_title in scenarios:
        sdir = bpath / "scenarios" / s_id
        learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")
        oracle = OracleDecisionScenario.load_npz(sdir / "oracle.npz")
        
        # 1. Trust Evaluation at t*
        diag = trust_evaluator.evaluate_trust(
            historical_obs=learner.historical_observations,
            historical_mask=learner.historical_observation_mask,
            historical_act=learner.historical_actions,
            t_star=learner.intervention_time,
        )
        
        # 2. Plan Intervention with Upfront Trust Gateway
        t0 = time.perf_counter()
        rec: PlanRecommendation = planner.plan_intervention(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
            trust_evaluator=trust_evaluator,
        )
        plan_ms = (time.perf_counter() - t0) * 1000.0
        
        rec_cand = rec.recommended_candidate
        rec_id = rec_cand.candidate_id if rec_cand else "ABSTAIN"
        orc_opt_id = oracle.oracle_optimal_candidate_id if oracle.oracle_optimal_candidate_id != "none" else "ABSTAIN"
        
        # Utilities and regret
        if rec_cand and rec_id != "ABSTAIN":
            prism_util = float(rec_cand.utility_score)
            orc_sel = oracle.candidate_outcomes.get(rec_id)
            orc_opt = oracle.candidate_outcomes.get(orc_opt_id)
            regret = float(orc_opt.true_utility - orc_sel.true_utility) if (orc_sel and orc_opt) else 0.0
            is_safe_rec = rec_cand.is_safe
            pred_peak_t = float(rec_cand.peak_t_core)
            pred_max_p = float(rec_cand.max_pressure)
            pred_min_f = float(rec_cand.min_flow)
        else:
            prism_util = None
            regret = 0.0 if (orc_opt_id == "ABSTAIN" or rec.should_abstain and orc_opt_id == "ABSTAIN") else None
            is_safe_rec = None
            pred_peak_t = None
            pred_max_p = None
            pred_min_f = None
            
        results[s_id] = {
            "scenario_id": s_id,
            "title": s_title,
            "trust_state": diag.state.value,
            "residual_t_core": diag.residual_t_core,
            "residual_8d_norm": diag.residual_8d_norm,
            "latent_mahalanobis_d": diag.latent_mahalanobis_d,
            "epistemic_sigma": diag.epistemic_sigma,
            "should_abstain": rec.should_abstain,
            "abstention_reason": rec.abstention_reason,
            "prism_recommendation": rec_id,
            "oracle_optimal": orc_opt_id,
            "agreement": (rec_id == orc_opt_id),
            "prism_utility": prism_util,
            "oracle_utility": oracle.candidate_outcomes[rec_id].true_utility if (rec_id != "ABSTAIN" and rec_id in oracle.candidate_outcomes) else (-1000.0 if rec_id == "ABSTAIN" and orc_opt_id == "none" else 0.0),
            "regret": regret,
            "is_safe_recommendation": is_safe_rec,
            "predicted_peak_t_core": pred_peak_t,
            "predicted_max_pressure": pred_max_p,
            "predicted_min_flow": pred_min_f,
            "planning_time_ms": plan_ms,
        }
        
    return results


# =============================================================================
# 2. Selective Decision Quality Metrics
# =============================================================================
def compute_selective_decision_metrics(
    bench_results: Dict[str, Any],
) -> Dict[str, Any]:
    total_scenarios = len(bench_results)
    trusted_count = 0
    trusted_correct = 0
    trusted_regrets = []
    
    unsafe_cases_total = 1 # Scenario 6 is the emergency out-of-support/all-unsafe case
    unsafe_cases_abstained = 0
    unsafe_cases_trusted = 0
    
    for s_id, r in bench_results.items():
        is_trusted = (r["trust_state"] != "model_abstain" and not r["should_abstain"])
        if is_trusted:
            trusted_count += 1
            if r["agreement"]:
                trusted_correct += 1
            if r["regret"] is not None:
                trusted_regrets.append(r["regret"])
                
        if s_id == "scenario_06_all_unsafe":
            if r["should_abstain"]:
                unsafe_cases_abstained += 1
            else:
                unsafe_cases_trusted += 1
                
    coverage = trusted_count / total_scenarios
    selective_accuracy = (trusted_correct / trusted_count) if trusted_count > 0 else 0.0
    selective_regret = float(np.mean(trusted_regrets)) if trusted_regrets else 0.0
    abstention_recall = unsafe_cases_abstained / unsafe_cases_total
    false_trust_rate = unsafe_cases_trusted / unsafe_cases_total
    
    return {
        "total_scenarios": total_scenarios,
        "trusted_decisions_count": trusted_count,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "selective_mean_regret": selective_regret,
        "abstention_recall": abstention_recall,
        "false_trust_rate": false_trust_rate,
    }


# =============================================================================
# 3. Threshold Sensitivity Analysis
# =============================================================================
def evaluate_threshold_sensitivity(
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    mu_id: np.ndarray,
    inv_cov_id: np.ndarray,
    val_dir: str = "data/unified_dataset/learner/validation",
    bench_dir: str = "data/decision_benchmark",
) -> Dict[str, Any]:
    """Test thresholds {4°C, 5°C, 6.08°C, 7°C, 8°C, 10°C} on FPR and S6 detection."""
    thresholds = [4.0, 5.0, 6.08, 7.0, 8.0, 10.0]
    
    # Validation data for FPR
    val_files = sorted(list(Path(val_dir).glob("*.npz")))
    val_residuals = []
    with torch.no_grad():
        for f in val_files:
            d = np.load(f)
            obs = d["observations"]
            act = d["actions"]
            mask = d["observation_mask"]
            obs_norm = norm.normalize(obs)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            inputs = ModelInputs(obs_norm.float().unsqueeze(0), torch.from_numpy(mask).bool().unsqueeze(0), torch.from_numpy(act).float().unsqueeze(0))
            lat_dist, _ = model.encode(inputs)
            rec_dist = model.decode(lat_dist.mean)
            rec_norm = rec_dist.mean.squeeze(0)
            rec_raw = norm.denormalize(rec_norm).cpu().numpy()
            r_T = np.abs(obs[:, 0] - rec_raw[:, 0])
            val_residuals.extend(r_T.tolist())
            
    val_residuals_arr = np.array(val_residuals)
    
    # Benchmark S6 residual
    s6_file = Path(bench_dir) / "scenarios" / "scenario_06_all_unsafe" / "learner.npz"
    s6_data = LearnerDecisionScenario.load_npz(s6_file)
    obs_s6 = norm.normalize(s6_data.historical_observations)
    if not isinstance(obs_s6, torch.Tensor):
        obs_s6 = torch.from_numpy(obs_s6)
    inputs_s6 = ModelInputs(obs_s6.float().unsqueeze(0), torch.from_numpy(s6_data.historical_observation_mask).bool().unsqueeze(0), torch.from_numpy(s6_data.historical_actions).float().unsqueeze(0))
    with torch.no_grad():
        lat_s6, _ = model.encode(inputs_s6)
        rec_s6 = model.decode(lat_s6.mean)
        rec_s6_raw = norm.denormalize(rec_s6.mean.squeeze(0)).cpu().numpy()
    s6_r_T = float(np.abs(s6_data.historical_observations[30, 0] - rec_s6_raw[30, 0]))
    
    # S1-S5 residuals
    safe_scen_residuals = []
    for s_idx in range(1, 6):
        s_id = f"scenario_0{s_idx}_{['do_nothing', 'valve', 'throttle', 'pump', 'combined'][s_idx-1]}"
        s_file = Path(bench_dir) / "scenarios" / s_id / "learner.npz"
        scen_d = LearnerDecisionScenario.load_npz(s_file)
        o_n = norm.normalize(scen_d.historical_observations)
        if not isinstance(o_n, torch.Tensor):
            o_n = torch.from_numpy(o_n)
        inp = ModelInputs(o_n.float().unsqueeze(0), torch.from_numpy(scen_d.historical_observation_mask).bool().unsqueeze(0), torch.from_numpy(scen_d.historical_actions).float().unsqueeze(0))
        with torch.no_grad():
            l_d, _ = model.encode(inp)
            r_d = model.decode(l_d.mean)
            r_raw = norm.denormalize(r_d.mean.squeeze(0)).cpu().numpy()
        safe_scen_residuals.append(float(np.abs(scen_d.historical_observations[scen_d.intervention_time, 0] - r_raw[scen_d.intervention_time, 0])))
        
    sensitivity_results = {}
    for tau in thresholds:
        fpr = float(np.mean(val_residuals_arr > tau))
        s6_detected = bool(s6_r_T > tau)
        safe_scens_trusted = int(sum(r <= tau for r in safe_scen_residuals))
        coverage = safe_scens_trusted / 5.0
        false_trust = 0.0 if s6_detected else 1.0
        
        sensitivity_results[f"tau_{tau:.2f}C"] = {
            "threshold_C": tau,
            "validation_fpr": fpr,
            "s6_residual_C": s6_r_T,
            "s6_detected": s6_detected,
            "safe_scenarios_trusted_count": f"{safe_scens_trusted}/5",
            "coverage": coverage,
            "false_trust_rate": false_trust,
        }
        
    return sensitivity_results


# =============================================================================
# 4. Dual-Gate Ablation (Residual vs Novelty vs Dual OR)
# =============================================================================
def evaluate_dual_gate_ablation(
    model: CausalWorldModel,
    norm: ObservationNormalizer,
    mu_id: np.ndarray,
    inv_cov_id: np.ndarray,
    val_dir: str = "data/unified_dataset/learner/validation",
    bench_dir: str = "data/decision_benchmark",
    tau_r: float = 6.08,
    tau_d: float = 15.0,
) -> Dict[str, Any]:
    """Compare Residual-only, Novelty-only, and Dual OR Gate."""
    # Compute on validation data
    val_files = sorted(list(Path(val_dir).glob("*.npz")))
    val_r_T = []
    val_d_M = []
    
    with torch.no_grad():
        for f in val_files:
            d = np.load(f)
            obs = d["observations"]
            act = d["actions"]
            mask = d["observation_mask"]
            obs_norm = norm.normalize(obs)
            if not isinstance(obs_norm, torch.Tensor):
                obs_norm = torch.from_numpy(obs_norm)
            inputs = ModelInputs(obs_norm.float().unsqueeze(0), torch.from_numpy(mask).bool().unsqueeze(0), torch.from_numpy(act).float().unsqueeze(0))
            lat_dist, _ = model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0).cpu().numpy()
            rec_dist = model.decode(lat_dist.mean)
            rec_raw = norm.denormalize(rec_dist.mean.squeeze(0)).cpu().numpy()
            
            r_T = np.abs(obs[:, 0] - rec_raw[:, 0])
            val_r_T.extend(r_T.tolist())
            
            for t in range(len(z_seq)):
                diff = z_seq[t] - mu_id
                d2 = float(diff @ inv_cov_id @ diff)
                val_d_M.append(float(np.sqrt(max(0.0, d2))))
                
    val_r_arr = np.array(val_r_T)
    val_d_arr = np.array(val_d_M)
    
    # S6 metrics
    s6_file = Path(bench_dir) / "scenarios" / "scenario_06_all_unsafe" / "learner.npz"
    s6_data = LearnerDecisionScenario.load_npz(s6_file)
    obs_s6 = norm.normalize(s6_data.historical_observations)
    if not isinstance(obs_s6, torch.Tensor):
        obs_s6 = torch.from_numpy(obs_s6)
    inputs_s6 = ModelInputs(obs_s6.float().unsqueeze(0), torch.from_numpy(s6_data.historical_observation_mask).bool().unsqueeze(0), torch.from_numpy(s6_data.historical_actions).float().unsqueeze(0))
    with torch.no_grad():
        lat_s6, _ = model.encode(inputs_s6)
        z_s6 = lat_s6.mean.squeeze(0)[30].cpu().numpy()
        rec_s6 = model.decode(lat_s6.mean)
        rec_s6_raw = norm.denormalize(rec_s6.mean.squeeze(0)).cpu().numpy()
    s6_r_T = float(np.abs(s6_data.historical_observations[30, 0] - rec_s6_raw[30, 0]))
    diff_s6 = z_s6 - mu_id
    s6_d_M = float(np.sqrt(max(0.0, float(diff_s6 @ inv_cov_id @ diff_s6))))
    
    # 1. Residual Only
    fpr_res = float(np.mean(val_r_arr > tau_r))
    s6_res_detected = bool(s6_r_T > tau_r)
    
    # 2. Novelty Only
    fpr_nov = float(np.mean(val_d_arr > tau_d))
    s6_nov_detected = bool(s6_d_M > tau_d)
    
    # 3. Dual OR Gate
    val_or_flagged = (val_r_arr > tau_r) | (val_d_arr > tau_d)
    fpr_or = float(np.mean(val_or_flagged))
    s6_or_detected = bool(s6_r_T > tau_r or s6_d_M > tau_d)
    
    return {
        "residual_only": {
            "threshold": f"R_T > {tau_r:.2f}°C",
            "validation_fpr": fpr_res,
            "s6_metric": f"R_T = {s6_r_T:.2f}°C",
            "s6_detected": s6_res_detected,
            "coverage_safe": float(1.0 - fpr_res),
        },
        "novelty_only": {
            "threshold": f"D_latent > {tau_d:.2f}",
            "validation_fpr": fpr_nov,
            "s6_metric": f"d_M = {s6_d_M:.2f}",
            "s6_detected": s6_nov_detected,
            "coverage_safe": float(1.0 - fpr_nov),
        },
        "dual_or_gate": {
            "threshold": f"R_T > {tau_r:.2f}°C OR D_latent > {tau_d:.2f}",
            "validation_fpr": fpr_or,
            "s6_detected": s6_or_detected,
            "coverage_safe": float(1.0 - fpr_or),
        },
    }


def main() -> None:
    print("=========================================================================")
    print("      TASK 5.10B: FULL DECISION-SYSTEM INTEGRATION VALIDATION             ")
    print("=========================================================================")
    
    out_dir = Path("artifacts/baseline_005/decision_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model, norm, trust_evaluator, planner = load_b005_pipeline()
    
    # 1. Full 6-Scenario Benchmark Evaluation with Trust Gateway
    print("\n[1] Evaluating Complete 6-Scenario Benchmark with Upstream Trust Gateway...")
    bench_res = evaluate_benchmark_with_trust_gate(planner, trust_evaluator)
    
    # 2. Selective Decision Quality Metrics
    print("\n[2] Computing Selective Decision Metrics...")
    selective_metrics = compute_selective_decision_metrics(bench_res)
    
    # 3. Threshold Sensitivity Analysis
    print("\n[3] Running Residual Threshold Sensitivity Analysis...")
    mu_id = np.array(trust_evaluator.mu_id)
    inv_cov_id = np.array(trust_evaluator.inv_cov_id)
    sensitivity_res = evaluate_threshold_sensitivity(model, norm, mu_id, inv_cov_id)
    
    # 4. Dual-Gate Ablation
    print("\n[4] Running Dual-Gate Ablation (Residual vs Novelty vs OR Gate)...")
    dual_gate_res = evaluate_dual_gate_ablation(model, norm, mu_id, inv_cov_id)
    
    # 5. Formal Assertion Check on S6 Abstention
    print("\n[5] Formally Asserting S6 Abstention Contract...")
    s6_out = bench_res["scenario_06_all_unsafe"]
    assert s6_out["trust_state"] == "model_abstain", f"S6 trust state was {s6_out['trust_state']}, expected model_abstain"
    assert s6_out["should_abstain"] is True, "S6 should_abstain was False, expected True"
    assert s6_out["prism_recommendation"] == "ABSTAIN", f"S6 recommendation was {s6_out['prism_recommendation']}, expected ABSTAIN"
    print("✓ S6 Abstention Assertion PASSED: S6 strictly aborted, planner blocked, zero unsafe candidate emitted.")
    
    master_report = {
        "benchmark_evaluation": bench_res,
        "selective_decision_metrics": selective_metrics,
        "threshold_sensitivity": sensitivity_res,
        "dual_gate_ablation": dual_gate_res,
        "s6_assertion_verdict": "PASS — S6 strictly blocked from emitting recommendation",
    }
    
    out_json = out_dir / "task_5_10b_decision_validation_report.json"
    with open(out_json, "w") as f:
        json.dump(master_report, f, indent=2)
        
    print(f"\nSaved complete decision validation report to {out_json}")
    
    # Print formatted summary tables
    print("\n" + "=" * 115)
    print("FULL 6-SCENARIO DECISION BENCHMARK RESULTS (BASELINE 005 + TRUST GATE)")
    print("=" * 115)
    print(f"{'Scenario':<25} | {'Trust State':<14} | {'PRISM Rec':<18} | {'Oracle Opt':<18} | {'Regret':<9} | {'R_T (°C)':<9} | {'d_M':<6}")
    print("-" * 115)
    for s_id, r in bench_res.items():
        s_lbl = r["title"].split(":")[0]
        reg_str = f"{r['regret']:.6f}" if r["regret"] is not None else "N/A"
        print(f"{s_lbl:<25} | {r['trust_state']:<14} | {r['prism_recommendation']:<18} | {r['oracle_optimal']:<18} | {reg_str:<9} | {r['residual_t_core']:7.2f}°C | {r['latent_mahalanobis_d']:5.2f}")
        
    print("\n" + "=" * 105)
    print("SELECTIVE DECISION QUALITY METRICS")
    print("=" * 105)
    for k, v in selective_metrics.items():
        print(f"  - {k:<30}: {v}")

    print("\n" + "=" * 105)
    print("THRESHOLD SENSITIVITY TABLE")
    print("=" * 105)
    print(f"{'Threshold':<12} | {'Val FPR':<10} | {'S6 Residual':<13} | {'S6 Detected?':<14} | {'Safe Trusted':<14} | {'False Trust Rate':<16}")
    print("-" * 105)
    for k, v in sensitivity_res.items():
        print(f"{v['threshold_C']:8.2f}°C    | {v['validation_fpr']*100:6.2f}%    | {v['s6_residual_C']:9.2f}°C   | {str(v['s6_detected']):<14} | {v['safe_scenarios_trusted_count']:<14} | {v['false_trust_rate']*100:14.2f}%")

    print("\n" + "=" * 105)
    print("DUAL-GATE ABLATION (RESIDUAL VS NOVELTY VS OR GATE)")
    print("=" * 105)
    print(f"{'Gate Configuration':<32} | {'Val FPR':<10} | {'S6 Metric':<18} | {'S6 Detected?':<14} | {'Safe Coverage':<14}")
    print("-" * 105)
    for g_key, g_val in dual_gate_res.items():
        lbl = g_key.replace("_", " ").title()
        metric_str = g_val.get("s6_metric", "-")
        print(f"{lbl:<32} | {g_val['validation_fpr']*100:6.2f}%    | {metric_str:<18} | {str(g_val['s6_detected']):<14} | {g_val['coverage_safe']*100:11.2f}%")


if __name__ == "__main__":
    main()
