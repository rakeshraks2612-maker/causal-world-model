"""Unit tests for Task 6.2: PRISM Causal Explanation Engine."""

import json
from pathlib import Path
import pytest
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.decision_benchmark import LearnerDecisionScenario
from prism.explanation.evidence import (
    DecisionEvidence,
    build_decision_evidence,
)
from prism.explanation.causal_explanation import (
    CausalExplanation,
    build_causal_explanation,
)


@pytest.fixture(scope="module")
def b005_pipeline():
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


def get_evidence_for_scenario(scenario_id: str, b005_pipeline, timestamp_utc: str = "2026-09-11T12:00:00Z") -> DecisionEvidence:
    model, norm, trust_evaluator, planner = b005_pipeline
    sdir = Path(f"data/decision_benchmark/scenarios/{scenario_id}")
    learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        learner.historical_observations,
        learner.historical_observation_mask,
        learner.historical_actions,
        learner.intervention_time,
    )

    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    return build_decision_evidence(
        recommendation=rec,
        trust_diagnostic=diag,
        scenario_id=scenario_id,
        timestamp_utc=timestamp_utc,
    )


# -----------------------------------------------------------------------------
# Test 1: Pump Pathway
# -----------------------------------------------------------------------------
def test_pump_pathway(b005_pipeline):
    """Test 1: S4 pump input produces exact A_pump -> F_cool -> T_cool -> T_core pathway."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert explanation.intervention.target == "A_pump"
    assert explanation.intervention.intervention_type == "ACTION"
    assert explanation.intervention.value == 3.0

    node_vars = [n.variable for n in explanation.causal_chain.nodes]
    assert node_vars == ["A_pump", "F_cool", "T_cool", "T_core"]

    assert explanation.causal_chain.nodes[0].relationship == "ACTUATOR_INPUT"
    assert explanation.causal_chain.nodes[1].relationship == "DIRECT_DESCENDANT"
    assert explanation.causal_chain.nodes[2].relationship == "MEDIATED_DESCENDANT"
    assert explanation.causal_chain.nodes[3].relationship == "FINAL_OUTCOME"

    assert "F_cool" in explanation.causal_chain.all_affected_descendants
    assert "T_core" in explanation.causal_chain.all_affected_descendants
    assert "P_sys" in explanation.causal_chain.all_affected_descendants


# -----------------------------------------------------------------------------
# Test 2: Throttle Pathway
# -----------------------------------------------------------------------------
def test_throttle_pathway(b005_pipeline):
    """Test 2: S3 throttle input produces exact A_throttle -> L_cpu -> P_elec -> T_core pathway."""
    evidence = get_evidence_for_scenario("scenario_03_throttle", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert explanation.intervention.target in ["A_throttle", "L_cpu"]
    assert explanation.intervention.value == 50.0

    node_vars = [n.variable for n in explanation.causal_chain.nodes]
    assert node_vars == ["A_throttle", "L_cpu", "P_elec", "T_core"]

    assert explanation.summary.effect_direction == "REDUCE_CORE_TEMPERATURE"
    assert "Joule heating" in explanation.summary.mechanism


# -----------------------------------------------------------------------------
# Test 3: Valve Pathway
# -----------------------------------------------------------------------------
def test_valve_pathway(b005_pipeline):
    """Test 3: S2 valve scenario derives valve DAG pathway."""
    evidence = get_evidence_for_scenario("scenario_02_valve", b005_pipeline)
    # Even if S2 recommends pump_4, test direct valve pathway derivation
    # by constructing a valve evidence object
    evidence.causal_evidence.intervention_target = "A_valve"
    evidence.causal_evidence.intervention_value = 85.0
    explanation = build_causal_explanation(evidence)

    node_vars = [n.variable for n in explanation.causal_chain.nodes]
    assert node_vars == ["A_valve", "V_pos", "F_cool", "P_sys", "T_core"]


# -----------------------------------------------------------------------------
# Test 4: Numerical Delta Integrity
# -----------------------------------------------------------------------------
def test_numerical_delta_integrity(b005_pipeline):
    """Test 4: Verify delta_t_core == recommended_peak_t_core - baseline_peak_t_core."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    base_t = explanation.predicted_effects.baseline_peak_t_core
    rec_t = explanation.predicted_effects.intervened_peak_t_core
    delta_t = explanation.predicted_effects.delta_t_core

    assert base_t is not None and rec_t is not None and delta_t is not None
    assert pytest.approx(delta_t, 1e-4) == rec_t - base_t

    delta_f = explanation.predicted_effects.delta_flow
    assert delta_f is not None
    assert pytest.approx(delta_f, 1e-4) == evidence.causal_evidence.counterfactual_comparison["delta_flow"]


# -----------------------------------------------------------------------------
# Test 5: No Fabricated Intermediate Values
# -----------------------------------------------------------------------------
def test_no_fabricated_intermediate_values(b005_pipeline):
    """Test 5: Intermediate variables without numerical telemetry have predicted_delta=None."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    t_cool_node = next(n for n in explanation.causal_chain.nodes if n.variable == "T_cool")
    assert t_cool_node.predicted_delta is None
    assert t_cool_node.evidence_available is False
    assert t_cool_node.predicted_direction == "DECREASE"


# -----------------------------------------------------------------------------
# Test 6: Alternative Rejection Rationale
# -----------------------------------------------------------------------------
def test_alternative_rejection(b005_pipeline):
    """Test 6: S4 cand_pump_4 rejected due to LATENT_NOVELTY (15.22 > 15.00)."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    pump_4_alt = next((alt for alt in explanation.rejected_alternatives if alt.candidate_id == "cand_pump_4"), None)
    assert pump_4_alt is not None
    assert pump_4_alt.reason_code == "LATENT_NOVELTY"
    assert "15.22" in pump_4_alt.reason_explanation or "latent support" in pump_4_alt.reason_explanation

    do_nothing_alt = next((alt for alt in explanation.rejected_alternatives if alt.candidate_id == "cand_do_nothing"), None)
    assert do_nothing_alt is not None
    assert do_nothing_alt.reason_code == "LOWER_UTILITY"


# -----------------------------------------------------------------------------
# Test 7: Decision Margin Calculation
# -----------------------------------------------------------------------------
def test_decision_margin(b005_pipeline):
    """Test 7: Verify decision margin equals best utility minus second-best utility."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    margin_info = explanation.decision_margin
    assert margin_info.best_candidate == "cand_pump_3"
    assert margin_info.second_best_candidate == "cand_do_nothing"
    assert margin_info.decision_strength == "POSITIVE_MARGIN"

    expected_margin = margin_info.best_utility - margin_info.second_best_utility
    assert pytest.approx(margin_info.decision_margin, 1e-4) == expected_margin
    assert margin_info.decision_margin > 0.0


# -----------------------------------------------------------------------------
# Test 8: Trusted State Qualification
# -----------------------------------------------------------------------------
def test_trusted_state(b005_pipeline):
    """Test 8: S4 is MODEL_TRUSTED and permits causal recommendation."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert "MODEL_TRUSTED" in explanation.trust_qualification
    assert explanation.abstention_details is None
    assert len(explanation.causal_chain.nodes) > 0


# -----------------------------------------------------------------------------
# Test 9: Abstention Contract on S6
# -----------------------------------------------------------------------------
def test_abstention_s6(b005_pipeline):
    """Test 9: S6 triggers MODEL_ABSTAIN, PLANNING_BLOCKED, and explicit violation difference."""
    evidence = get_evidence_for_scenario("scenario_06_all_unsafe", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert "MODEL_ABSTAIN" in explanation.trust_qualification
    assert explanation.abstention_details is not None
    assert explanation.abstention_details.trigger == "RECONSTRUCTION_INCONSISTENCY"
    assert explanation.abstention_details.action_taken == "PLANNING_BLOCKED"

    obs = explanation.abstention_details.observed_value
    thresh = explanation.abstention_details.threshold
    delta = explanation.abstention_details.violation_delta

    assert pytest.approx(obs, 1e-2) == 33.88
    assert pytest.approx(thresh, 1e-2) == 6.08
    assert pytest.approx(delta, 1e-2) == obs - thresh  # ~27.80°C


# -----------------------------------------------------------------------------
# Test 10: No Simulation Under Abstention
# -----------------------------------------------------------------------------
def test_no_simulation_under_abstention(b005_pipeline):
    """Test 10: S6 has no candidate nodes, no rejected alternatives, and blocked counterfactual."""
    evidence = get_evidence_for_scenario("scenario_06_all_unsafe", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert len(explanation.causal_chain.nodes) == 0
    assert len(explanation.rejected_alternatives) == 0
    assert "Counterfactual simulation withheld" in explanation.predicted_effects.counterfactual_statement
    assert explanation.summary.effect_direction == "NO_EFFECT_UNTRUSTED"


# -----------------------------------------------------------------------------
# Test 11: Provenance and Hash Propagation
# -----------------------------------------------------------------------------
def test_provenance_propagation(b005_pipeline):
    """Test 11: Source evidence hash and version strings are faithfully propagated."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert explanation.provenance.source_evidence_hash == evidence.provenance.decision_hash
    assert explanation.provenance.model_version == evidence.provenance.model_version
    assert explanation.provenance.dataset_version == evidence.provenance.dataset_version
    assert explanation.provenance.planner_version == evidence.provenance.planner_version
    assert explanation.provenance.benchmark_version == evidence.provenance.benchmark_version
    assert explanation.provenance.scenario_id == "scenario_04_pump"


# -----------------------------------------------------------------------------
# Test 12: Determinism
# -----------------------------------------------------------------------------
def test_determinism(b005_pipeline):
    """Test 12: Generating the explanation multiple times produces identical output."""
    evidence = get_evidence_for_scenario("scenario_04_pump", b005_pipeline)
    exp_1 = build_causal_explanation(evidence, timestamp_utc="2026-09-11T12:00:00Z")
    exp_2 = build_causal_explanation(evidence, timestamp_utc="2026-09-11T12:00:00Z")

    assert exp_1.to_dict() == exp_2.to_dict()
    assert exp_1.to_json() == exp_2.to_json()


# -----------------------------------------------------------------------------
# Test 13: S1 Inaction Restraint Explanation
# -----------------------------------------------------------------------------
def test_s1_inaction_explanation(b005_pipeline):
    """Test 13: S1 generates clean Inaction Restraint explanation."""
    evidence = get_evidence_for_scenario("scenario_01_do_nothing", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert explanation.intervention.intervention_type == "NONE"
    assert "DO NOTHING" in explanation.summary.headline
    assert explanation.summary.effect_direction == "NOMINAL_EQUILIBRIUM"
    assert len(explanation.causal_chain.nodes) == 1
    assert explanation.causal_chain.nodes[0].variable == "None"


# -----------------------------------------------------------------------------
# Test 14: S5 Compound Intervention Explanation
# -----------------------------------------------------------------------------
def test_s5_compound_explanation(b005_pipeline):
    """Test 14: S5 generates clean compound intervention explanation."""
    evidence = get_evidence_for_scenario("scenario_05_combined", b005_pipeline)
    explanation = build_causal_explanation(evidence)

    assert explanation.intervention.target in ["A_pump", "pump", "compound"]
    assert explanation.summary.effect_direction == "REDUCE_CORE_TEMPERATURE"
    assert explanation.predicted_effects.delta_t_core is not None
    assert len(explanation.causal_chain.nodes) > 0
