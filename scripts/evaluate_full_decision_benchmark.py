"""Full Decision Quality Benchmark Evaluation Harness (Task 5.8).

Evaluates PRISM World Model across all 6 frozen decision scenarios:
1. scenario_01_do_nothing (Inaction / Restraint)
2. scenario_02_valve (Cooling Valve Intervention)
3. scenario_03_throttle (Compute Throttling)
4. scenario_04_pump (Hydraulic Pump Bottleneck)
5. scenario_05_combined (Multi-Variable Compound Intervention)
6. scenario_06_all_unsafe (Emergency Regime / Rigorous Abstention)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

from prism.dataset.decision_benchmark import (
    LearnerDecisionScenario,
    OracleDecisionScenario,
)
from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.planning.planner import InterventionPlanner, PlanRecommendation


def evaluate_model_on_benchmark(model_dir: str | Path, benchmark_dir: str | Path = "data/decision_benchmark") -> Dict[str, Any]:
    """Execute complete 6-scenario decision quality evaluation for a given world model."""
    mpath = Path(model_dir)
    bpath = Path(benchmark_dir)
    
    norm = ObservationNormalizer.load_yaml(mpath / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(mpath / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(mpath / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    planner = InterventionPlanner(model, norm)
    
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
        
        t0 = time.perf_counter()
        rec: PlanRecommendation = planner.plan_intervention(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
        )
        plan_time_ms = (time.perf_counter() - t0) * 1000.0
        
        is_abstain = rec.should_abstain
        rec_cand = rec.recommended_candidate
        rec_id = rec_cand.candidate_id if rec_cand else "ABSTAIN"
        orc_opt_id = oracle.oracle_optimal_candidate_id if oracle.oracle_optimal_candidate_id != "none" else "ABSTAIN"
        agreement = (rec_id == orc_opt_id)
        
        # Calculate regret and margins
        prism_util = float(rec_cand.utility_score) if rec_cand else None
        orc_sel_outcome = oracle.candidate_outcomes.get(rec_id) if rec_id != "ABSTAIN" else None
        orc_opt_outcome = oracle.candidate_outcomes.get(orc_opt_id) if orc_opt_id != "ABSTAIN" else None
        
        if orc_sel_outcome and orc_opt_outcome:
            regret = float(orc_opt_outcome.true_utility - orc_sel_outcome.true_utility)
        elif rec_id == "ABSTAIN" and orc_opt_id == "ABSTAIN":
            regret = 0.0
        else:
            regret = None
            
        # Margin vs 2nd best safe candidate
        sorted_safe = sorted([c for c in rec.all_evaluated_candidates if c.is_safe], key=lambda c: c.utility_score, reverse=True)
        if len(sorted_safe) > 1:
            margin_2nd = float(sorted_safe[0].utility_score - sorted_safe[1].utility_score)
        else:
            margin_2nd = 0.0
            
        cands_data = []
        for c in rec.all_evaluated_candidates:
            orc_c = oracle.candidate_outcomes.get(c.candidate_id)
            cands_data.append({
                "candidate_id": c.candidate_id,
                "is_safe_prism": c.is_safe,
                "is_safe_oracle": orc_c.is_safe if orc_c else False,
                "peak_t_core": c.peak_t_core,
                "max_pressure": c.max_pressure,
                "min_flow": c.min_flow,
                "latent_novelty": c.latent_novelty,
                "utility_prism": c.utility_score,
                "utility_oracle": orc_c.true_utility if orc_c else None,
                "violations": c.safety_violations,
            })
            
        results[s_id] = {
            "scenario_id": s_id,
            "title": s_title,
            "prism_selected": rec_id,
            "oracle_optimal": orc_opt_id,
            "agreement": agreement,
            "should_abstain": is_abstain,
            "prism_utility": prism_util,
            "oracle_selected_utility": orc_sel_outcome.true_utility if orc_sel_outcome else None,
            "oracle_optimal_utility": orc_opt_outcome.true_utility if orc_opt_outcome else None,
            "utility_regret": regret,
            "margin_vs_second_best": margin_2nd,
            "planning_time_ms": plan_time_ms,
            "explanation": rec.explanation.to_dict() if rec.explanation else None,
            "candidates": cands_data,
        }
        
    return results


def main() -> None:
    print("=========================================================================")
    print("      PRISM FULL DECISION QUALITY BENCHMARK (TASK 5.8)                   ")
    print("=========================================================================")
    
    res_b4 = evaluate_model_on_benchmark("artifacts/baseline_004")
    res_b3 = evaluate_model_on_benchmark("artifacts/baseline_003")
    
    out_dir = Path("artifacts/baseline_004")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task_5_8_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_004": res_b4,
            "baseline_003": res_b3,
        }, f, indent=2)
        
    print(f"Saved master benchmark results to {out_path}")


if __name__ == "__main__":
    main()
