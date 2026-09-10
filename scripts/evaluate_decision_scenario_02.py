"""Evaluation Harness for Scenario 2 — Valve Intervention Is Optimal (Task 5.5).

Demonstrates and verifies:
1. Strict Learner/Oracle Firewall: Planner consumes only learner.npz.
2. Complete Candidate Simulation and Safety Evaluation.
3. Action Cost Model Objective Scoring and Causal Mechanism Propagation.
4. Oracle Ground-Truth Comparison (Post-Decision).
5. Comprehensive Candidate Table, Causal Delta Comparison, and Explanation.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.decision_benchmark import (
    LearnerDecisionScenario,
    OracleDecisionScenario,
    DecisionClass,
)
from prism.planning.planner import InterventionPlanner, PlanRecommendation


def evaluate_scenario_02(
    model_dir: str | Path = "artifacts/baseline_003",
    benchmark_dir: str | Path = "data/decision_benchmark",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete decision evaluation for Scenario 2."""
    model_path = Path(model_dir)
    bench_path = Path(benchmark_dir)
    scen_dir = bench_path / "scenarios" / "scenario_02_valve"

    # 1. Load trained world model and normalizer
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    planner = InterventionPlanner(model, norm)

    # 2. FIREWALL: Load learner record only for planning
    learner_scen = LearnerDecisionScenario.load_npz(scen_dir / "learner.npz")

    t_start = time.perf_counter()
    plan_rec: PlanRecommendation = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )
    plan_duration_ms = (time.perf_counter() - t_start) * 1000.0

    # 3. Post-Decision: Load Oracle for evaluation
    oracle_scen = OracleDecisionScenario.load_npz(scen_dir / "oracle.npz")

    selected_cand = plan_rec.recommended_candidate
    selected_id = selected_cand.candidate_id if selected_cand else "none"
    oracle_opt_id = oracle_scen.oracle_optimal_candidate_id
    action_agreement = (selected_id == oracle_opt_id)

    # Decision class: RECOMMEND if safe candidate chosen, ABSTAIN if should_abstain
    decision_class = "ABSTAIN" if plan_rec.should_abstain else "RECOMMEND"
    unnecessary_intervention = (selected_id == "cand_do_nothing")

    # Utility scores & regrets
    prism_utility = float(selected_cand.utility_score) if selected_cand else -1000.0
    oracle_opt_outcome = oracle_scen.candidate_outcomes.get(oracle_opt_id)
    oracle_opt_utility = float(oracle_opt_outcome.true_utility) if oracle_opt_outcome else 0.0

    selected_oracle_outcome = oracle_scen.candidate_outcomes.get(selected_id)
    selected_oracle_utility = float(selected_oracle_outcome.true_utility) if selected_oracle_outcome else -1000.0
    utility_regret = float(oracle_opt_utility - selected_oracle_utility)

    # Decision margin: Utility(selected) - Utility(second_best)
    sorted_cands = sorted(plan_rec.all_evaluated_candidates, key=lambda c: c.utility_score, reverse=True)
    if len(sorted_cands) > 1:
        decision_margin = float(sorted_cands[0].utility_score - sorted_cands[1].utility_score)
    else:
        decision_margin = 0.0

    # Margin against do-nothing
    do_nothing_cand = next((c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_do_nothing"), None)
    do_nothing_utility = float(do_nothing_cand.utility_score) if do_nothing_cand else -1000.0
    margin_vs_do_nothing = float(prism_utility - do_nothing_utility)

    is_safe = bool(selected_cand.is_safe) if selected_cand else False

    # 4. Compute Causal Mechanism Deltas (PRISM vs Oracle for selected candidate)
    # PRISM Deltas (simulated intervention vs baseline)
    prism_sim = selected_cand.simulation_result
    if prism_sim is not None:
        prism_base_obs = prism_sim.baseline_observations
        prism_int_obs = prism_sim.intervened_observations

        # Channels: 0:T_core, 1:T_cool, 2:P_sys, 3:F_cool, 5:V_pos
        delta_prism_v_pos = float(np.mean(prism_int_obs[:, 5] - prism_base_obs[:, 5]))
        delta_prism_f_cool = float(np.mean(prism_int_obs[:, 3] - prism_base_obs[:, 3]))
        delta_prism_p_sys = float(np.mean(prism_int_obs[:, 2] - prism_base_obs[:, 2]))
        delta_prism_t_cool = float(np.mean(prism_int_obs[:, 1] - prism_base_obs[:, 1]))
        delta_prism_t_core = float(np.mean(prism_int_obs[:, 0] - prism_base_obs[:, 0]))
    else:
        delta_prism_v_pos = delta_prism_f_cool = delta_prism_p_sys = delta_prism_t_cool = delta_prism_t_core = 0.0

    # Oracle Deltas (selected candidate vs baseline do-nothing)
    oracle_base_outcome = oracle_scen.candidate_outcomes.get("cand_do_nothing")
    if selected_oracle_outcome and oracle_base_outcome:
        orc_int_obs = selected_oracle_outcome.ground_truth_observations
        orc_base_obs = oracle_base_outcome.ground_truth_observations
        delta_orc_v_pos = float(np.mean(orc_int_obs[:, 5] - orc_base_obs[:, 5]))
        delta_orc_f_cool = float(np.mean(orc_int_obs[:, 3] - orc_base_obs[:, 3]))
        delta_orc_p_sys = float(np.mean(orc_int_obs[:, 2] - orc_base_obs[:, 2]))
        delta_orc_t_cool = float(np.mean(orc_int_obs[:, 1] - orc_base_obs[:, 1]))
        delta_orc_t_core = float(np.mean(orc_int_obs[:, 0] - orc_base_obs[:, 0]))
    else:
        delta_orc_v_pos = delta_orc_f_cool = delta_orc_p_sys = delta_orc_t_cool = delta_orc_t_core = 0.0

    causal_direction_concordance = (
        (delta_prism_v_pos > 0 and delta_orc_v_pos > 0) and
        (delta_prism_f_cool > 0 and delta_orc_f_cool > 0) and
        (delta_prism_p_sys > 0 and delta_orc_p_sys > 0) and
        (delta_prism_t_core < 0 and delta_orc_t_core < 0)
    )

    result = {
        "scenario_id": "scenario_02_valve",
        "scenario_name": learner_scen.scenario_name,
        "oracle_optimal": oracle_opt_id,
        "prism_selected": selected_id,
        "action_agreement": action_agreement,
        "decision_class": decision_class,
        "expected_decision_class": oracle_scen.expected_decision_class.value,
        "is_safe": is_safe,
        "unnecessary_intervention": unnecessary_intervention,
        "prism_utility": prism_utility,
        "oracle_utility": selected_oracle_utility,
        "oracle_optimal_utility": oracle_opt_utility,
        "utility_regret": utility_regret,
        "decision_margin": decision_margin,
        "margin_vs_do_nothing": margin_vs_do_nothing,
        "causal_deltas": {
            "PRISM": {
                "delta_V_pos": delta_prism_v_pos,
                "delta_F_cool": delta_prism_f_cool,
                "delta_P_sys": delta_prism_p_sys,
                "delta_T_cool": delta_prism_t_cool,
                "delta_T_core": delta_prism_t_core,
            },
            "Oracle": {
                "delta_V_pos": delta_orc_v_pos,
                "delta_F_cool": delta_orc_f_cool,
                "delta_P_sys": delta_orc_p_sys,
                "delta_T_cool": delta_orc_t_cool,
                "delta_T_core": delta_orc_t_core,
            },
            "concordance": causal_direction_concordance,
        },
        "planning_time_ms": plan_duration_ms,
        "candidates": [c.to_dict() for c in plan_rec.all_evaluated_candidates],
        "explanation": plan_rec.explanation.to_dict(),
    }

    if verbose:
        print("=========================================================================")
        print("      PRISM DECISION BENCHMARK — SCENARIO 2 EVALUATION REPORT            ")
        print("=========================================================================")
        print(f"Scenario: {learner_scen.scenario_name}")
        print(f"Description: {learner_scen.description}\n")

        print("### Candidate Evaluation Table (PRISM vs Oracle)")
        print("| Candidate ID | Safe (PRISM) | Peak T (°C) | Max P (bar) | Min F (L/min) | PRISM Util | Novelty | Safe (Orc) | Orc Util | Result |")
        print("| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :--- |")
        for c in sorted_cands:
            tag = "🏆 SELECTED" if c.candidate_id == selected_id else "REJECTED"
            orc_out = oracle_scen.candidate_outcomes.get(c.candidate_id)
            orc_safe = "🟢 Yes" if (orc_out and orc_out.is_safe) else "🔴 No"
            orc_util = f"{orc_out.true_utility:+.4f}" if orc_out else "N/A"
            print(f"| `{c.candidate_id}` | {'🟢 Yes' if c.is_safe else '🔴 No'} | {c.peak_t_core:.2f} | {c.max_pressure:.2f} | {c.min_flow:.2f} | {c.utility_score:+.4f} | {c.latent_novelty:.2f} | {orc_safe} | {orc_util} | {tag} |")

        print("\n### Cost Breakdown Table (PRISM)")
        print("| Candidate ID | Perf Benefit | Thermal Pen | Pressure Pen | Pump Cost | Actuation Cost | Throttle Cost | Total Cost | Net Utility |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for c in sorted_cands:
            cb = c.cost_breakdown
            print(f"| `{c.candidate_id}` | {cb.performance_benefit:.4f} | {cb.thermal_penalty:.4f} | {cb.pressure_penalty:.4f} | {cb.pump_cost:.4f} | {cb.actuation_cost:.4f} | {cb.throttle_cost:.4f} | {cb.total_cost:.4f} | {cb.net_utility:+.4f} |")

        print("\n### Causal Mechanism Propagation (Intervention Deltas)")
        print("| Physical Variable | PRISM Predicted Δ | Oracle Ground-Truth Δ | Causal Concordance | Expected Direction |")
        print("| :--- | ---: | ---: | :---: | :---: |")
        print(f"| `ΔV_pos` (Valve Position) | {delta_prism_v_pos:+.2f} % | {delta_orc_v_pos:+.2f} % | {'✅ PASS' if delta_prism_v_pos > 0 and delta_orc_v_pos > 0 else '❌ FAIL'} | `ΔV_pos > 0` |")
        print(f"| `ΔF_cool` (Coolant Flow)   | {delta_prism_f_cool:+.2f} L/min | {delta_orc_f_cool:+.2f} L/min | {'✅ PASS' if delta_prism_f_cool > 0 and delta_orc_f_cool > 0 else '❌ FAIL'} | `ΔF_cool > 0` |")
        print(f"| `ΔP_sys` (System Pressure) | {delta_prism_p_sys:+.2f} bar | {delta_orc_p_sys:+.2f} bar | {'✅ PASS' if (delta_prism_p_sys > 0) == (delta_orc_p_sys > 0) else '❌ FAIL'} | `Hydraulic response` |")
        print(f"| `ΔT_cool` (Coolant Temp)  | {delta_prism_t_cool:+.2f} °C | {delta_orc_t_cool:+.2f} °C | {'✅ PASS' if delta_prism_t_cool < 0 and delta_orc_t_cool < 0 else '❌ FAIL'} | `ΔT_cool < 0` |")
        print(f"| `ΔT_core` (Core Temp)     | {delta_prism_t_core:+.2f} °C | {delta_orc_t_core:+.2f} °C | {'✅ PASS' if delta_prism_t_core < 0 and delta_orc_t_core < 0 else '❌ FAIL'} | `ΔT_core < 0` |")

        print("\n### Decision Intelligence Summary")
        print(f"Oracle Optimal:             {oracle_opt_id}")
        print(f"PRISM Selected:             {selected_id}")
        print(f"Action Agreement:           {'PASS' if action_agreement else 'FAIL'}")
        print(f"Decision Class:             {decision_class} (Expected: {oracle_scen.expected_decision_class.value})")
        print(f"PRISM Utility:              {prism_utility:+.4f}")
        print(f"Oracle Utility:             {selected_oracle_utility:+.4f}")
        print(f"Utility Regret:             {utility_regret:.6f}")
        print(f"Safety Verdict:             {'PASS (Safe)' if is_safe else 'FAIL (Unsafe)'}")
        print(f"Decision Margin (vs 2nd):   {decision_margin:+.4f}")
        print(f"Decision Margin (vs Inaction): {margin_vs_do_nothing:+.4f}")
        print(f"Causal Direction Concord:   {'PASS' if causal_direction_concordance else 'FAIL'}")
        print(f"Planning Latency:           {plan_duration_ms:.2f} ms")

        print("\n### Generated Causal Explanation")
        print(plan_rec.explanation.format_markdown())
        print("=========================================================================")

    return result


if __name__ == "__main__":
    evaluate_scenario_02()
