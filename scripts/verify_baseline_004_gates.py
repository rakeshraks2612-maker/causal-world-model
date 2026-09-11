"""Comprehensive Verification Suite for Task 5.7C Gates (Baseline 004).

Evaluates:
- Gate A/B: Action Sensitivity (Pump 1 vs Pump 3 at h=1, 5, 10, 20, 40)
- Gate C: Causal Directionality (Pump ^ -> Flow ^ -> Pressure ^ -> Core Temp v)
- Gate D: Action Discrimination (Monotonicity across stages 1, 2, 3, 4)
- Gate E: Scenario 4 Decision Evaluation (PRISM vs Oracle, candidate table, explanation)
- Gate F: Regression & Generalization (Scenario 1, 2, 3 re-evaluation under baseline_004)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

from prism.dataset.decision_benchmark import (
    LearnerDecisionScenario,
    OracleDecisionScenario,
    DecisionClass,
)
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.training.normalization import ObservationNormalizer
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from scripts.evaluate_decision_scenario_01 import evaluate_scenario_01
from scripts.evaluate_decision_scenario_02 import evaluate_scenario_02
from scripts.evaluate_decision_scenario_03 import evaluate_scenario_03
from scripts.evaluate_decision_scenario_04 import evaluate_scenario_04


def run_gate_checks(model_dir: str | Path = "artifacts/baseline_004") -> Dict[str, Any]:
    model_path = Path(model_dir)
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cpu")

    # Load Scenario 4 historical context (t=0..40)
    scen4_path = Path("data/decision_benchmark/scenarios/scenario_04_pump/learner.npz")
    learner_scen = LearnerDecisionScenario.load_npz(scen4_path)

    obs_raw = learner_scen.historical_observations
    mask_raw = learner_scen.historical_observation_mask
    act_raw = learner_scen.historical_actions
    norm_obs = norm.normalize(torch.tensor(obs_raw, dtype=torch.float32)).numpy()

    inputs_ctx = ModelInputs(
        observations=torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device),
        observation_mask=torch.tensor(mask_raw, dtype=torch.float32).unsqueeze(0).to(device),
        actions=torch.tensor(act_raw, dtype=torch.float32).unsqueeze(0).to(device),
    )

    H = 40
    horizons = [1, 5, 10, 20, 40]

    # 1. Action Sensitivity & Discrimination across Stages 1, 2, 3, 4
    stage_preds = {}
    stage_preds_norm = {}

    for stage in [1, 2, 3, 4]:
        act = np.zeros((1, H, 4), dtype=np.float32)
        act[:, :, 0] = 60.0    # Fixed valve
        act[:, :, 1] = 100.0   # Fixed throttle
        act[:, :, 2] = float(stage)
        act[:, :, 3] = 0.0

        with torch.no_grad():
            rollout = model.forecast(inputs_ctx, torch.tensor(act).to(device), deterministic=True)
            p_norm = rollout.observations.mean.squeeze(0).cpu()
            p_phys = norm.denormalize(p_norm).numpy()
            stage_preds_norm[stage] = p_norm.numpy()
            stage_preds[stage] = p_phys

    # Gate A & B: Pump 1 vs Pump 3 Sensitivity Table
    p1 = stage_preds[1]
    p3 = stage_preds[3]
    diff_p3_p1_norm = stage_preds_norm[3] - stage_preds_norm[1]
    traj_div_1_3 = float(np.linalg.norm(diff_p3_p1_norm))

    sensitivity_table = []
    for h in horizons:
        idx = h - 1
        sensitivity_table.append({
            "horizon": h,
            "P1_Flow": float(p1[idx, 3]),
            "P3_Flow": float(p3[idx, 3]),
            "delta_Flow": float(p3[idx, 3] - p1[idx, 3]),
            "P1_Press": float(p1[idx, 2]),
            "P3_Press": float(p3[idx, 2]),
            "delta_Press": float(p3[idx, 2] - p1[idx, 2]),
            "P1_Tcore": float(p1[idx, 0]),
            "P3_Tcore": float(p3[idx, 0]),
            "delta_Tcore": float(p3[idx, 0] - p1[idx, 0]),
            "delta_Tcool": float(p3[idx, 1] - p1[idx, 1]),
            "delta_Vib": float(p3[idx, 6] - p1[idx, 6]),
        })

    # Gate C: Causal Directionality Check
    mean_delta_F = float(np.mean(p3[:, 3] - p1[:, 3]))
    mean_delta_P = float(np.mean(p3[:, 2] - p1[:, 2]))
    mean_delta_Tcore = float(np.mean(p3[:, 0] - p1[:, 0]))

    causal_direction_pass = (mean_delta_F > 0.0) and (mean_delta_P > 0.0) and (mean_delta_Tcore < 0.0)

    # Gate D: Action Discrimination (Stages 1..4 at h=40)
    h40_flows = [float(stage_preds[s][39, 3]) for s in [1, 2, 3, 4]]
    h40_pressures = [float(stage_preds[s][39, 2]) for s in [1, 2, 3, 4]]
    h40_tcores = [float(stage_preds[s][39, 0]) for s in [1, 2, 3, 4]]

    monotonic_flow = (h40_flows[0] <= h40_flows[1] <= h40_flows[2] <= h40_flows[3])
    discrimination_pass = monotonic_flow and (h40_tcores[2] < h40_tcores[0])

    # 2. Gate E: Evaluate Scenario 4
    print("\n>>> Running Full Decision Benchmark for Scenario 4 with Baseline 004...")
    scen4_eval = evaluate_scenario_04(model_dir=model_dir, verbose=True)

    # 3. Gate F: Generalization Check (Scenarios 1, 2, 3)
    print("\n>>> Running Generalization Regression across Scenarios 1, 2, 3 with Baseline 004...")
    scen1_eval = evaluate_scenario_01(model_dir=model_dir, verbose=False)
    scen2_eval = evaluate_scenario_02(model_dir=model_dir, verbose=False)
    scen3_eval = evaluate_scenario_03(model_dir=model_dir, verbose=False)

    return {
        "gate_sensitivity": {
            "trajectory_divergence_1_3": traj_div_1_3,
            "sensitivity_table": sensitivity_table,
            "mean_delta_F": mean_delta_F,
            "mean_delta_P": mean_delta_P,
            "mean_delta_Tcore": mean_delta_Tcore,
        },
        "gate_causal_direction": {
            "pass": causal_direction_pass,
            "flow_increasing": mean_delta_F > 0.0,
            "pressure_increasing": mean_delta_P > 0.0,
            "core_temp_decreasing": mean_delta_Tcore < 0.0,
        },
        "gate_action_discrimination": {
            "pass": discrimination_pass,
            "h40_flows_p1_p4": h40_flows,
            "h40_pressures_p1_p4": h40_pressures,
            "h40_tcores_p1_p4": h40_tcores,
        },
        "scenario_04_results": {
            "prism_selected": scen4_eval["prism_selected"],
            "oracle_optimal": scen4_eval["oracle_optimal"],
            "agreement_pass": scen4_eval["action_agreement"],
            "decision_class": scen4_eval["decision_class"],
            "utility_regret": scen4_eval["utility_regret"],
            "candidates": scen4_eval["candidates"],
        },
        "generalization_regression": {
            "scenario_01": {
                "selected": scen1_eval["prism_selected"],
                "oracle": scen1_eval["oracle_optimal"],
                "pass": scen1_eval["action_agreement"],
            },
            "scenario_02": {
                "selected": scen2_eval["prism_selected"],
                "oracle": scen2_eval["oracle_optimal"],
                "pass": scen2_eval["action_agreement"],
            },
            "scenario_03": {
                "selected": scen3_eval["prism_selected"],
                "oracle": scen3_eval["oracle_optimal"],
                "pass": scen3_eval["action_agreement"],
            },
        },
    }


def main() -> None:
    print("=========================================================================")
    print("         VERIFYING BASELINE 004 GATES FOR TASK 5.7C                      ")
    print("=========================================================================")

    results = run_gate_checks("artifacts/baseline_004")

    print("\n=========================================================================")
    print("                          GATE RESULTS SUMMARY                           ")
    print("=========================================================================")
    
    sens = results["gate_sensitivity"]
    print(f"Gate A (Sensitivity): Trajectory Divergence (P1 vs P3) = {sens['trajectory_divergence_1_3']:.4f} (baseline_003 was 0.2994)")
    print("\nPump 1 vs Pump 3 Sensitivity Table Across Horizons:")
    print(f"{'Horizon':<8} | {'ΔFlow (L/min)':<14} | {'ΔPress (bar)':<14} | {'ΔTcore (°C)':<14} | {'ΔVib':<10}")
    print("-" * 70)
    for row in sens["sensitivity_table"]:
        print(f"h={row['horizon']:<6} | {row['delta_Flow']:+14.2f} | {row['delta_Press']:+14.2f} | {row['delta_Tcore']:+14.2f} | {row['delta_Vib']:+10.2f}")

    cdir = results["gate_causal_direction"]
    print(f"\nGate B (Causal Direction): {'PASS ✅' if cdir['pass'] else 'FAIL ❌'}")
    print(f"  Flow Increases:     {cdir['flow_increasing']} (Mean ΔF = {sens['mean_delta_F']:+.2f} L/min)")
    print(f"  Pressure Increases: {cdir['pressure_increasing']} (Mean ΔP = {sens['mean_delta_P']:+.2f} bar)")
    print(f"  Core Temp Cools:    {cdir['core_temp_decreasing']} (Mean ΔT = {sens['mean_delta_Tcore']:+.2f}°C)")

    disc = results["gate_action_discrimination"]
    print(f"\nGate C (Discrimination across Stages 1-4 at h=40): {'PASS ✅' if disc['pass'] else 'FAIL ❌'}")
    print(f"  Flows (P1..P4):     {[f'{f:.2f}' for f in disc['h40_flows_p1_p4']]} L/min")
    print(f"  Pressures (P1..P4): {[f'{p:.2f}' for p in disc['h40_pressures_p1_p4']]} bar")
    print(f"  Tcore (P1..P4):     {[f'{t:.2f}' for t in disc['h40_tcores_p1_p4']]} °C")

    s4 = results["scenario_04_results"]
    print(f"\nGate D (Scenario 4 Decision Agreement): {'PASS ✅' if s4['agreement_pass'] else 'FAIL ❌'}")
    print(f"  PRISM Selected: {s4['prism_selected']}")
    print(f"  Oracle Optimal: {s4['oracle_optimal']}")
    print(f"  Utility Regret: {s4['utility_regret']:.6f}")

    gen = results["generalization_regression"]
    print(f"\nGate E (Generalization across Scenarios 1, 2, 3):")
    print(f"  Scenario 1 (Do Nothing): {'PASS ✅' if gen['scenario_01']['pass'] else 'FAIL ❌'} (Selected: {gen['scenario_01']['selected']})")
    print(f"  Scenario 2 (Valve):      {'PASS ✅' if gen['scenario_02']['pass'] else 'FAIL ❌'} (Selected: {gen['scenario_02']['selected']})")
    print(f"  Scenario 3 (Throttle):   {'PASS ✅' if gen['scenario_03']['pass'] else 'FAIL ❌'} (Selected: {gen['scenario_03']['selected']})")

    # Save verification artifact
    out_file = Path("artifacts/baseline_004/gate_verification_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved gate verification results to {out_file}")


if __name__ == "__main__":
    main()
