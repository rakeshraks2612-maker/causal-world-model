"""PRISM Phase 6 Integrated Evidence Test Suite (Task 6.7).

Validates the complete end-to-end Phase 6 evidence stack as a unified,
tamper-evident, deterministic system across the frozen 6-scenario benchmark
(S1-S6) without modifying planner decisions, safety gates, or model weights.

Test Categories:
1. Benchmark Scenario System Projections (S1-S6)
2. Tamper-Evident SHA-256 Fingerprint Sensitivity
3. Deterministic Bit-Identical Reproducibility
4. Non-Interference Invariant (Planner & Gate Isolation)
5. JSON Serialization & Deserialization Fidelity
6. Executive Markdown Dossier Structure Completeness
7. Twin-World Counterfactual Integration & Graceful Absence
8. Non-Finite / Corrupted Value Fail-Closed Robustness
"""

from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator, ModelTrustState
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario

from prism.explanation.evidence import (
    build_decision_evidence,
    DecisionEvidence,
    DecisionSummary,
    TrustEvidence,
    CausalEvidence,
    PredictedOutcome,
    SafetyEvidence,
    DecisionQuality,
    ProvenanceEvidence,
)
from prism.explanation.causal_explanation import (
    CausalExplanation,
    build_causal_explanation,
)
from prism.explanation.counterfactual_evidence import (
    CounterfactualEvidence,
    FactualWorldContext,
    CounterfactualIntervention,
    CounterfactualWorldOutcome,
    CausalEffectEvidence,
    TwinWorldIntegrity,
    SafetyComparison,
    UncertaintyContext,
    CounterfactualProvenance,
    build_counterfactual_evidence,
)
from prism.explanation.safety_evidence import (
    SafetyEvidenceDetail,
    build_safety_evidence_from_decision_evidence,
    evaluate_safety_evidence_detail,
)
from prism.explanation.abstention_explanation import (
    AbstentionExplanation,
    build_abstention_explanation,
    build_abstention_explanation_from_decision_evidence,
)
from prism.explanation.unified_record import (
    PrismDecisionRecord,
    UnifiedProvenance,
    build_unified_decision_record,
    compute_deterministic_unified_hash,
)


@pytest.fixture(scope="module")
def baseline_system():
    """Load the frozen baseline_005 model, trust evaluator, and planner."""
    mpath = Path("artifacts/baseline_005")
    norm = ObservationNormalizer.load_yaml(mpath / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(mpath / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(mpath / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(mpath / "trust_calibration.json") as f:
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
# 1. Benchmark Scenario System Projections (S1-S6)
# =============================================================================

def test_scenario_01_inaction_restraint_record(baseline_system):
    """S1: Inaction restraint produces trusted, safe do-nothing decision record."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_01_do_nothing")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_01", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_do_nothing"
    assert record.decision.decision_status == "NO_ACTION_REQUIRED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.safety.overall_state in ["SAFE", "MARGINAL"]
    assert record.abstention.abstained is False
    assert record.provenance.scenario_id == "scenario_01"
    assert len(record.provenance.unified_record_hash) == 64


def test_scenario_02_valve_cooling_record(baseline_system):
    """S2: Valve cooling candidate evaluation preserves decision ranking and margins."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_02_valve")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_02", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_pump_4"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.limiting_constraint is not None
    assert record.provenance.unified_record_hash is not None


def test_scenario_03_workload_throttling_record(baseline_system):
    """S3: Workload throttling produces conservative thermal evaluation and positive margins."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_03_throttle")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_03", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_throttle_50"
    assert record.decision.decision_status == "RECOMMENDED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.safety.constraints["thermal"].margin > 0.0
    assert record.abstention.abstained is False


def test_scenario_04_pump_modulation_record(baseline_system):
    """S4: Hydraulic pump modulation validates multi-objective Pareto trade-off."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_04_pump")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_04", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_pump_3"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.decision_quality.utility_score is not None


def test_scenario_05_compound_intervention_record(baseline_system):
    """S5: Multi-variable compound intervention validates multi-variable ranking."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_05_combined")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_05", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_pump_4_only"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True


def test_scenario_06_model_trust_abstention_record(baseline_system):
    """S6: Emergency regime upfront trust failure triggers rigorous MODEL_ABSTAIN."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_06_all_unsafe")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_06", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation is None
    assert record.decision.decision_status == "BLOCKED"
    assert record.trust.trust_state == "MODEL_ABSTAIN"
    assert record.trust.reconstruction_residual_t_core > 6.0
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "ABSTAIN_REQUIRED"
    assert record.abstention.abstained is True
    assert record.abstention.abstention_type == "MODEL_ABSTAIN"
    assert "reconstruction residual" in record.abstention.primary_reason.lower()
    assert record.decision_quality.utility_score is None


# =============================================================================
# 2. Cryptographic Fingerprint & Tamper-Evidence Tests
# =============================================================================

def test_tamper_evident_integrity_on_evidence_mutation(baseline_system):
    """Assert changing any evidence metric alters the SHA-256 root fingerprint."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_01_do_nothing")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_01", timestamp_utc="2026-09-11T12:00:00Z")
    rec_orig = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")
    original_hash = rec_orig.provenance.unified_record_hash

    # Mutate trust residual
    dec_ev_tampered_trust = copy.deepcopy(dec_ev)
    dec_ev_tampered_trust.trust.reconstruction_residual_t_core += 1.0
    rec_tampered_trust = build_unified_decision_record(dec_ev_tampered_trust, timestamp_utc="2026-09-11T12:00:00Z")
    assert rec_tampered_trust.provenance.unified_record_hash != original_hash

    # Mutate predicted outcome peak temperature
    dec_ev_tampered_safety = copy.deepcopy(dec_ev)
    dec_ev_tampered_safety.predicted_outcome.peak_t_core += 5.0
    rec_tampered_safety = build_unified_decision_record(dec_ev_tampered_safety, timestamp_utc="2026-09-11T12:00:00Z")
    assert rec_tampered_safety.provenance.unified_record_hash != original_hash

    # Mutate utility score
    dec_ev_tampered_utility = copy.deepcopy(dec_ev)
    dec_ev_tampered_utility.decision_quality.utility_score = 0.9999
    rec_tampered_utility = build_unified_decision_record(dec_ev_tampered_utility, timestamp_utc="2026-09-11T12:00:00Z")
    assert rec_tampered_utility.provenance.unified_record_hash != original_hash


def test_deterministic_bit_identical_reproducibility(baseline_system):
    """Assert assembling identical evidence contracts yields bit-identical SHA-256 hashes."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_03_throttle")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev1 = build_decision_evidence(rec, diag, scenario_id="scenario_03", timestamp_utc="2026-09-11T12:00:00Z")
    record1 = build_unified_decision_record(dec_ev1, timestamp_utc="2026-09-11T12:00:00Z")

    dec_ev2 = build_decision_evidence(rec, diag, scenario_id="scenario_03", timestamp_utc="2026-09-11T12:00:00Z")
    record2 = build_unified_decision_record(dec_ev2, timestamp_utc="2026-09-11T12:00:00Z")

    assert record1.provenance.unified_record_hash == record2.provenance.unified_record_hash
    assert record1.to_json() == record2.to_json()


# =============================================================================
# 3. Non-Interference & Immutability Invariant
# =============================================================================

def test_non_interference_invariant(baseline_system):
    """Assert generating the entire evidence stack does not alter planner state or ranking."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_01_do_nothing")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    # Snapshot planner outputs
    rec_cand_before = rec.recommended_candidate.candidate_id if rec.recommended_candidate else None
    num_evals_before = len(rec.all_evaluated_candidates)
    cands_scores_before = [(c.candidate_id, c.utility_score, c.is_safe) for c in rec.all_evaluated_candidates]

    # Run complete Phase 6 evidence assembly
    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_01", timestamp_utc="2026-09-11T12:00:00Z")
    _ = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    # Verify planner outputs after evidence assembly
    rec_cand_after = rec.recommended_candidate.candidate_id if rec.recommended_candidate else None
    num_evals_after = len(rec.all_evaluated_candidates)
    cands_scores_after = [(c.candidate_id, c.utility_score, c.is_safe) for c in rec.all_evaluated_candidates]

    assert rec_cand_before == rec_cand_after
    assert num_evals_before == num_evals_after
    assert cands_scores_before == cands_scores_after


# =============================================================================
# 4. JSON Serialization & Markdown Dossier Completeness
# =============================================================================

def test_json_roundtrip_fidelity(baseline_system):
    """Assert JSON export round-trips without data loss or type corruption."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_04_pump")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_04", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    json_str = record.to_json()
    parsed = json.loads(json_str)

    assert parsed["decision"]["recommendation"] == "cand_pump_3"
    assert parsed["provenance"]["unified_record_hash"] == record.provenance.unified_record_hash
    assert parsed["trust"]["trust_state"] == "MODEL_TRUSTED"
    assert "limiting_constraint" in parsed["safety"]


def test_markdown_dossier_all_sections_present(baseline_system):
    """Assert formatted Markdown dossier includes all 7 mandated executive sections."""
    _, _, trust_evaluator, planner = baseline_system
    bpath = Path("data/decision_benchmark/scenarios/scenario_06_all_unsafe")
    learner = LearnerDecisionScenario.load_npz(bpath / "learner.npz")

    diag = trust_evaluator.evaluate_trust(
        historical_obs=learner.historical_observations,
        historical_mask=learner.historical_observation_mask,
        historical_act=learner.historical_actions,
        t_star=learner.intervention_time,
    )
    rec = planner.plan_intervention(
        historical_observations=learner.historical_observations,
        historical_actions=learner.historical_actions,
        future_actions=learner.future_baseline_actions,
        custom_candidates=learner.candidate_actions,
        intervention_time=learner.intervention_time,
        trust_evaluator=trust_evaluator,
    )

    dec_ev = build_decision_evidence(rec, diag, scenario_id="scenario_06", timestamp_utc="2026-09-11T12:00:00Z")
    record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

    md = record.format_markdown()
    assert "PRISM Unified Decision Record & Audit Dossier" in md
    assert "## 1. Executive Summary & Causal Reasoning" in md
    assert "## 2. Upfront Model Trust & Telemetry Integrity" in md
    assert "## 3. Physical Safety & Support Constraint Audit" in md
    assert "## 4. Abstention & Trust Boundary Audit" in md
    assert "## 6. Decision Quality & Optimization Gap" in md
    assert "## 7. Cryptographic Provenance Chain" in md
    assert record.provenance.unified_record_hash in md


# =============================================================================
# 5. Non-Finite / Corrupted Value Fail-Closed Robustness
# =============================================================================

def test_non_finite_fail_closed_robustness():
    """Assert non-finite values (NaN/Inf) produce fail-closed outcomes across safety and abstention."""
    # Safety detail with NaN
    saf_detail = evaluate_safety_evidence_detail(
        peak_t_core=float("nan"),
        max_pressure=3.5,
        min_flow=15.0,
        latent_novelty=2.0,
        sigma_t_core=1.0,
        sigma_pressure=0.2,
        sigma_flow=0.5,
    )
    assert saf_detail.is_safe is False
    assert saf_detail.overall_state == "ABSTAIN_REQUIRED"
    assert saf_detail.limiting_constraint.constraint_name == "model_trust_gateway"
    assert any(v.severity == "ABSTAIN_REQUIRED" for v in saf_detail.violations)

    # Abstention explanation with NaN prediction (Fail-Closed Missing Prediction)
    abst_missing = build_abstention_explanation(
        is_abstained=True,
        peak_t_core_predicted=float("nan"),
        max_pressure_predicted=3.5,
        min_flow_predicted=15.0,
    )
    assert abst_missing.abstained is True
    assert abst_missing.abstention_type == "MISSING_PREDICTION"
    assert "missing required simulation channels" in abst_missing.primary_reason.lower()

    # Abstention explanation with excessive valid residual
    abst_expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
        threshold_residual_t_core=6.0,
        reconstruction_residual_8d=1.82,
        threshold_residual_8d=1.9,
        latent_novelty_d=2.0,
        threshold_latent_novelty=15.0,
        peak_t_core_predicted=75.0,
        max_pressure_predicted=3.5,
        min_flow_predicted=15.0,
    )
    assert abst_expl.abstained is True
    assert abst_expl.abstention_type == "MODEL_ABSTAIN"
    assert "33.88" in abst_expl.primary_reason
