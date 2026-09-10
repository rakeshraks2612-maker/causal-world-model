"""Scenario 1 Regression Tests: Do Nothing Is Optimal (Task 5.4).

Verifies:
1. Learner/Oracle isolation and firewall enforcement.
2. Candidate set integrity (4 candidates present).
3. Do-nothing simulation identity with factual baseline.
4. Candidate safety gating prior to utility ranking.
5. Utility scoring and fine-grained cost breakdown integrity.
6. Planner returns valid PlanRecommendation.
7. Decision class matches expected RECOMMEND.
8. PRISM selection matches Oracle optimal ('cand_do_nothing').
9. Selected recommendation is strictly safe.
10. Unnecessary intervention is False (avoids spurious actuation).
11. Generated explanation provides valid causal rationale.
12. Deterministic reproducibility across repeated runs.
"""

from __future__ import annotations
import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.decision_benchmark import (
    LearnerDecisionScenario,
    OracleDecisionScenario,
    DecisionClass,
)
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.schema import FORBIDDEN_LEARNER_KEYS
from scripts.evaluate_decision_scenario_01 import evaluate_scenario_01


@pytest.fixture(scope="module")
def scenario_system():
    """Load model, normalizer, and scenario 1 records."""
    model_path = Path("artifacts/baseline_003")
    assert model_path.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scen_dir = Path("data/decision_benchmark/scenarios/scenario_01_do_nothing")
    learner_scen = LearnerDecisionScenario.load_npz(scen_dir / "learner.npz")
    oracle_scen = OracleDecisionScenario.load_npz(scen_dir / "oracle.npz")

    planner = InterventionPlanner(model, norm)
    return planner, learner_scen, oracle_scen


def test_scenario_01_firewall_and_latent_isolation(scenario_system):
    """1. Verify that learner record has zero unobserved latent states or oracle artifacts."""
    _, learner_scen, _ = scenario_system
    
    # Check support metadata
    for k in learner_scen.support_metadata:
        assert k not in FORBIDDEN_LEARNER_KEYS

    # Raw npz archive check
    raw_data = np.load("data/decision_benchmark/scenarios/scenario_01_do_nothing/learner.npz", allow_pickle=True)
    for forbidden in FORBIDDEN_LEARNER_KEYS:
        assert forbidden not in raw_data.files


def test_scenario_01_candidate_set_integrity(scenario_system):
    """2. Verify scenario 1 contains the 4 canonical candidates."""
    _, learner_scen, _ = scenario_system
    cand_ids = [c.candidate_id for c in learner_scen.candidate_actions]
    assert len(cand_ids) == 4
    assert "cand_do_nothing" in cand_ids
    assert "cand_valve_85" in cand_ids
    assert "cand_pump_4" in cand_ids
    assert "cand_throttle_30" in cand_ids


def test_scenario_01_do_nothing_baseline_identity(scenario_system):
    """3. Verify that cand_do_nothing produces a rollout identical to the unintervened baseline."""
    planner, learner_scen, _ = scenario_system

    # Direct baseline simulation
    baseline_sim = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=None,
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    do_nothing_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_do_nothing")
    assert do_nothing_eval.simulation_result is not None

    np.testing.assert_allclose(
        do_nothing_eval.simulation_result.intervened_observations,
        baseline_sim.baseline_observations,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Do-nothing rollout must match baseline rollout identically",
    )


def test_scenario_01_safety_and_utility_evaluation(scenario_system):
    """4 & 5. Verify all candidates have safety results and fine-grained cost breakdowns."""
    planner, learner_scen, _ = scenario_system
    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    for c in plan_rec.all_evaluated_candidates:
        assert c.safety_result is not None, f"Candidate {c.candidate_id} missing safety_result"
        assert c.cost_breakdown is not None, f"Candidate {c.candidate_id} missing cost_breakdown"
        assert isinstance(c.utility_score, float)
        assert c.is_safe is True, f"In Scenario 1, candidate {c.candidate_id} should be physically safe"


def test_scenario_01_decision_recommendation_and_agreement(scenario_system):
    """6, 7, 8, 9, 10. Verify planner chooses cand_do_nothing with RECOMMEND and zero regret."""
    planner, learner_scen, oracle_scen = scenario_system
    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    assert plan_rec.should_abstain is False
    assert plan_rec.recommended_candidate is not None
    
    selected = plan_rec.recommended_candidate
    assert selected.candidate_id == "cand_do_nothing"
    assert selected.candidate_id == oracle_scen.oracle_optimal_candidate_id
    assert selected.is_safe is True

    # Decision margin: do_nothing utility must strictly exceed alternatives
    sorted_cands = sorted(plan_rec.all_evaluated_candidates, key=lambda c: c.utility_score, reverse=True)
    assert sorted_cands[0].candidate_id == "cand_do_nothing"
    decision_margin = sorted_cands[0].utility_score - sorted_cands[1].utility_score
    assert decision_margin > 0.0, "Do-nothing must have strictly positive decision margin"


def test_scenario_01_causal_explanation_validity(scenario_system):
    """11. Verify explanation contains structured, non-empty diagnosis and mechanisms."""
    planner, learner_scen, _ = scenario_system
    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    exp = plan_rec.explanation
    assert exp.summary != ""
    assert exp.diagnosis != ""
    assert len(exp.rejected_alternatives) >= 3
    assert "SAFE" in exp.safety_verdict


def test_scenario_01_determinism(scenario_system):
    """12. Verify repeated planning calls return strictly identical candidate scores."""
    planner, learner_scen, _ = scenario_system

    res1 = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    res2 = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    assert res1.recommended_candidate.candidate_id == res2.recommended_candidate.candidate_id
    assert res1.recommended_candidate.utility_score == res2.recommended_candidate.utility_score


def test_scenario_01_harness_integration():
    """Verify scripts/evaluate_decision_scenario_01.py runs end-to-end with PASS."""
    res = evaluate_scenario_01(verbose=False)
    assert res["action_agreement"] is True
    assert res["prism_selected"] == "cand_do_nothing"
    assert res["unnecessary_intervention"] is False
    assert res["is_safe"] is True
    assert res["utility_regret"] == 0.0
