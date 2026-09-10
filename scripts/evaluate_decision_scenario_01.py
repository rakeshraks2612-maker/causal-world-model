"""Evaluation Harness for Scenario 1 — Do Nothing Is Optimal (Task 5.4).

Demonstrates and verifies:
1. Strict Learner/Oracle Firewall: Planner consumes only learner.npz.
2. Complete Candidate Simulation and Safety Evaluation.
3. Action Cost Model Objective Scoring.
4. Oracle Ground-Truth Comparison (Post-Decision).
5. Comprehensive Candidate Table and Causal Explanation.
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


def evaluate_scenario_01(
    model_dir: str | Path = "artifacts/baseline_003",
    benchmark_dir: str | Path = "data/decision_benchmark",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete decision evaluation for Scenario 1."""
    model_path = Path(model_dir)
    bench_path = Path(benchmark_dir)
    scen_dir = bench_path / "scenarios" / "scenario_01_do_nothing"

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
    unnecessary_intervention = (selected_id != "cand_do_nothing")

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

    is_safe = bool(selected_cand.is_safe) if selected_cand else False

    result = {
        "scenario_id": "scenario_01_do_nothing",
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
        "planning_time_ms": plan_duration_ms,
        "candidates": [c.to_dict() for c in plan_rec.all_evaluated_candidates],
        "explanation": plan_rec.explanation.to_dict(),
    }

    if verbose:
        print("=========================================================================")
        print("      PRISM DECISION BENCHMARK — SCENARIO 1 EVALUATION REPORT            ")
        print("=========================================================================")
        print(f"Scenario: {learner_scen.scenario_name}")
        print(f"Description: {learner_scen.description}\n")

        print("### Candidate Evaluation Table")
        print("| Candidate ID | Safe | Peak T (°C) | Max P (bar) | Min F (L/min) | Net Utility | Latent Novelty | Result |")
        print("| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :--- |")
        for c in sorted_cands:
            tag = "🏆 SELECTED" if c.candidate_id == selected_id else "REJECTED"
            print(f"| `{c.candidate_id}` | {'🟢 Yes' if c.is_safe else '🔴 No'} | {c.peak_t_core:.2f} | {c.max_pressure:.2f} | {c.min_flow:.2f} | {c.utility_score:+.4f} | {c.latent_novelty:.2f} | {tag} |")

        print("\n### Cost Breakdown Table")
        print("| Candidate ID | Perf Benefit | Thermal Pen | Pressure Pen | Pump Cost | Actuation Cost | Throttle Cost | Total Cost | Net Utility |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for c in sorted_cands:
            cb = c.cost_breakdown
            print(f"| `{c.candidate_id}` | {cb.performance_benefit:.4f} | {cb.thermal_penalty:.4f} | {cb.pressure_penalty:.4f} | {cb.pump_cost:.4f} | {cb.actuation_cost:.4f} | {cb.throttle_cost:.4f} | {cb.total_cost:.4f} | {cb.net_utility:+.4f} |")

        print("\n### Decision Intelligence Summary")
        print(f"Oracle Optimal:             {oracle_opt_id}")
        print(f"PRISM Selected:             {selected_id}")
        print(f"Action Agreement:           {'PASS' if action_agreement else 'FAIL'}")
        print(f"Decision Class:             {decision_class} (Expected: {oracle_scen.expected_decision_class.value})")
        print(f"PRISM Utility:              {prism_utility:+.4f}")
        print(f"Oracle Utility:             {selected_oracle_utility:+.4f}")
        print(f"Utility Regret:             {utility_regret:.4f}")
        print(f"Safety Verdict:             {'PASS (Safe)' if is_safe else 'FAIL (Unsafe)'}")
        print(f"Unnecessary Intervention:   {'NO (Optimal Inaction)' if not unnecessary_intervention else 'YES (Sub-optimal Over-actuation)'}")
        print(f"Decision Margin:            {decision_margin:+.4f}")
        print(f"Planning Latency:           {plan_duration_ms:.2f} ms")

        print("\n### Generated Causal Explanation")
        print(plan_rec.explanation.format_markdown())
        print("=========================================================================")

    return result


if __name__ == "__main__":
    evaluate_scenario_01()
