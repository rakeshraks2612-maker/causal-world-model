"""Unit tests for Task 6.1: PRISM Decision Evidence Contract."""

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
    get_causal_dag_path,
    compute_deterministic_evidence_hash,
    SAFETY_CONSTRAINTS,
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


def test_causal_dag_path_mappings():
    """Verify deterministic causal DAG paths match physical SCM specification."""
    pump_path, pump_vars, pump_dir = get_causal_dag_path("A_pump")
    assert pump_path == ["A_pump", "F_cool", "T_cool", "T_core"]
    assert "F_cool" in pump_vars and "T_core" in pump_vars
    assert "COOLANT" in pump_dir

    throt_path, throt_vars, throt_dir = get_causal_dag_path("A_throttle")
    assert throt_path == ["A_throttle", "L_cpu", "P_elec", "T_core"]
    assert "L_cpu" in throt_vars and "P_elec" in throt_vars
    assert "JOULE" in throt_dir

    valve_path, valve_vars, valve_dir = get_causal_dag_path("A_valve")
    assert valve_path == ["A_valve", "V_pos", "F_cool", "P_sys", "T_core"]
    assert "V_pos" in valve_vars

    none_path, none_vars, none_dir = get_causal_dag_path("none")
    assert none_path == ["None"]
    assert len(none_vars) == 0


def test_decision_evidence_s4_pump(b005_pipeline):
    """Verify complete DecisionEvidence object for Scenario 4 (Pump modulation)."""
    model, norm, trust_evaluator, planner = b005_pipeline
    sdir = Path("data/decision_benchmark/scenarios/scenario_04_pump")
    learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")

    # Trust evaluation
    diag = trust_evaluator.evaluate_trust(
        learner.historical_observations,
        learner.historical_observation_mask,
        learner.historical_actions,
        learner.intervention_time,
    )

    # Planning
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    evidence: DecisionEvidence = build_decision_evidence(
        recommendation=rec,
        trust_diagnostic=diag,
        scenario_id="scenario_04_pump",
        timestamp_utc="2026-09-11T12:00:00Z",
    )

    # 1. Decision assertions
    assert evidence.decision.recommendation == "cand_pump_3"
    assert evidence.decision.decision_status == "RECOMMENDED"
    assert evidence.decision.abstention_reason is None

    # 2. Trust assertions
    assert evidence.trust.trust_state == "MODEL_TRUSTED"
    assert evidence.trust.reconstruction_residual_t_core < 6.08
    assert evidence.trust.latent_novelty_d < 15.0

    # 3. Causal evidence
    assert evidence.causal_evidence.intervention_target == "A_pump"
    assert evidence.causal_evidence.intervention_value == 3.0
    assert evidence.causal_evidence.causal_path == ["A_pump", "F_cool", "T_cool", "T_core"]
    assert evidence.causal_evidence.counterfactual_comparison["delta_flow"] > 0

    # 4. Predicted outcome & Exact safety margins
    peak_t = evidence.predicted_outcome.peak_t_core
    max_p = evidence.predicted_outcome.max_pressure
    min_f = evidence.predicted_outcome.min_flow
    assert peak_t is not None and peak_t < 95.0
    assert max_p is not None and max_p < 5.5
    assert min_f is not None and min_f > 8.0

    assert pytest.approx(evidence.safety_evidence.thermal_margin_c, 1e-4) == 95.0 - peak_t
    assert pytest.approx(evidence.safety_evidence.pressure_margin_bar, 1e-4) == 5.5 - max_p
    assert pytest.approx(evidence.safety_evidence.flow_margin_l_min, 1e-4) == min_f - 8.0
    assert evidence.safety_evidence.is_safe is True
    assert evidence.safety_evidence.safety_state == "SAFE"

    # 5. Decision quality
    assert evidence.decision_quality.utility_score is not None
    assert evidence.decision_quality.second_best_candidate is not None
    assert evidence.decision_quality.decision_margin > 0.0
    assert len(evidence.decision_quality.alternatives_rejected) > 0

    # 6. Provenance & Hash
    assert evidence.provenance.model_version == "baseline_005"
    assert evidence.provenance.scenario_id == "scenario_04_pump"
    assert len(evidence.provenance.decision_hash) == 64  # SHA-256

    # JSON serialization and report formatting
    j_str = evidence.to_json()
    assert "cand_pump_3" in j_str
    report_text = evidence.format_ascii_report()
    assert "cand_pump_3" in report_text
    assert "Thermal Margin" in report_text


def test_decision_evidence_s6_abstention(b005_pipeline):
    """Verify DecisionEvidence contract on Scenario 6 (Acute shock abstention)."""
    model, norm, trust_evaluator, planner = b005_pipeline
    sdir = Path("data/decision_benchmark/scenarios/scenario_06_all_unsafe")
    learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")

    # Trust evaluation
    diag = trust_evaluator.evaluate_trust(
        learner.historical_observations,
        learner.historical_observation_mask,
        learner.historical_actions,
        learner.intervention_time,
    )

    # Planning with trust gate
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    evidence: DecisionEvidence = build_decision_evidence(
        recommendation=rec,
        trust_diagnostic=diag,
        scenario_id="scenario_06_all_unsafe",
        timestamp_utc="2026-09-11T12:00:00Z",
    )

    # 1. Decision assertions
    assert evidence.decision.recommendation is None
    assert evidence.decision.decision_status == "BLOCKED"
    assert "Reconstruction inconsistency" in evidence.decision.abstention_reason

    # 2. Trust assertions
    assert evidence.trust.trust_state == "MODEL_ABSTAIN"
    assert evidence.trust.reconstruction_residual_t_core > 30.0  # ~33.88°C
    assert evidence.trust.tau_residual_t == pytest.approx(6.08, 0.05)

    # 3. Safety assertions
    assert evidence.safety_evidence.safety_state == "ABSTAIN_REQUIRED"
    assert evidence.safety_evidence.is_safe is False
    assert evidence.safety_evidence.thermal_margin_c is None  # Blocked

    # 4. Predicted outcome
    assert evidence.predicted_outcome.peak_t_core is None

    # 5. Decision quality
    assert evidence.decision_quality.utility_score is None
    assert len(evidence.decision_quality.alternatives_rejected) == 0

    # 6. Provenance
    assert len(evidence.provenance.decision_hash) == 64
    assert evidence.provenance.scenario_id == "scenario_06_all_unsafe"


def test_decision_evidence_s1_inaction(b005_pipeline):
    """Verify DecisionEvidence contract on Scenario 1 (Inaction restraint)."""
    model, norm, trust_evaluator, planner = b005_pipeline
    sdir = Path("data/decision_benchmark/scenarios/scenario_01_do_nothing")
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

    evidence: DecisionEvidence = build_decision_evidence(
        recommendation=rec,
        trust_diagnostic=diag,
        scenario_id="scenario_01_do_nothing",
        timestamp_utc="2026-09-11T12:00:00Z",
    )

    assert evidence.decision.recommendation == "cand_do_nothing"
    assert evidence.decision.decision_status == "NO_ACTION_REQUIRED"
    assert evidence.trust.trust_state == "MODEL_TRUSTED"
    assert evidence.causal_evidence.causal_path == ["None"]
    assert evidence.safety_evidence.safety_state == "SAFE"
    assert evidence.safety_evidence.is_safe is True
    assert evidence.safety_evidence.thermal_margin_c > 0.0


def test_decision_evidence_s3_throttle(b005_pipeline):
    """Verify DecisionEvidence contract on Scenario 3 (Workload throttling)."""
    model, norm, trust_evaluator, planner = b005_pipeline
    sdir = Path("data/decision_benchmark/scenarios/scenario_03_throttle")
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

    evidence: DecisionEvidence = build_decision_evidence(
        recommendation=rec,
        trust_diagnostic=diag,
        scenario_id="scenario_03_throttle",
        timestamp_utc="2026-09-11T12:00:00Z",
    )

    assert evidence.decision.recommendation == "cand_throttle_50"
    assert evidence.decision.decision_status == "RECOMMENDED"
    assert evidence.trust.trust_state == "MODEL_TRUSTED"
    assert evidence.causal_evidence.causal_path == ["A_throttle", "L_cpu", "P_elec", "T_core"]
    assert "JOULE" in evidence.causal_evidence.direction_of_effect
    assert evidence.safety_evidence.safety_state == "SAFE"
    assert evidence.safety_evidence.thermal_margin_c > 0.0
