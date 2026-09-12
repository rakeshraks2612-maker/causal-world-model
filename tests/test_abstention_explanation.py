"""Unit tests for PRISM Abstention Explanation Engine (Task 6.5).

Comprehensive test suite (21 tests) covering:
1. Model thermal residual abstention (R_T > tau)
2. Model 8D multivariate residual abstention (R_8D > tau)
3. Exact trust threshold for R_T (R_T == 6.00 -> PASS)
4. Residual just below threshold (R_T = 5.95 -> PASS)
5. Residual just above threshold (R_T = 6.05 -> TRIGGER)
6. Latent novelty abstention (D_latent > 15.0)
7. Latent boundary exact (D_latent == 15.0 -> PASS)
8. Missing thermal prediction (peak_t_core is None)
9. Missing pressure prediction (max_pressure is None)
10. Missing flow prediction (min_flow is None)
11. Multiple abstention reasons priority (MODEL_ABSTAIN > LATENT_NOVELTY)
12. Deterministic reason ordering and trigger recording
13. Blocked recommendation and action emission
14. Blocked counterfactual semantics
15. Safety implication distinction (not generic UNSAFE)
16. Recommended next step operator actions
17. Deterministic SHA-256 provenance hash
18. JSON serialization and schema roundtrip
19. Markdown audit formatting for abstained state
20. Markdown formatting for non-abstained state
21. DecisionEvidence contract integration
"""

import pytest
import json

from prism.explanation.abstention_explanation import (
    AbstentionExplanation,
    AbstentionEvidenceMetric,
    AbstentionProvenance,
    build_abstention_explanation,
    build_abstention_explanation_from_decision_evidence,
)
from prism.explanation.evidence import (
    DecisionEvidence,
    DecisionSummary,
    TrustEvidence,
    CausalEvidence,
    PredictedOutcome,
    SafetyEvidence,
    DecisionQuality,
    ProvenanceEvidence,
)


def test_model_thermal_residual_abstention():
    """1. Thermal reconstruction residual above threshold triggers MODEL_ABSTAIN."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=1.20,
        threshold_residual_8d=1.90,
        latent_novelty_d=3.5,
        threshold_latent_novelty=15.0,
        scenario_id="scenario_06",
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MODEL_ABSTAIN"
    assert "Thermal reconstruction residual R_T = 33.88°C exceeds trust threshold 6.00°C" in expl.primary_reason
    assert expl.affected_decision == "RECOMMENDATION_BLOCKED"

    # Verify metrics
    rt_metric = next(m for m in expl.evidence if "Thermal" in m.metric_name)
    assert rt_metric.is_trigger is True
    assert rt_metric.observed_value == 33.88
    assert rt_metric.threshold == 6.00
    assert rt_metric.excess == pytest.approx(27.88, abs=1e-2)
    assert rt_metric.direction == "GREATER_THAN_THRESHOLD"


def test_model_8d_residual_abstention():
    """2. 8D multivariate reconstruction residual above threshold triggers MODEL_ABSTAIN."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=2.50,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=2.45,
        threshold_residual_8d=1.90,
        latent_novelty_d=4.1,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MODEL_ABSTAIN"
    assert "8D residual R_8D = 2.45 exceeds threshold 1.90" in expl.primary_reason

    r8d_metric = next(m for m in expl.evidence if "8D" in m.metric_name)
    assert r8d_metric.is_trigger is True
    assert r8d_metric.excess == pytest.approx(0.55, abs=1e-2)


def test_exact_trust_threshold_rt():
    """3. Residual exactly at threshold (R_T == 6.00) does not trigger abstention."""
    expl = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=6.00,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=1.50,
        threshold_residual_8d=1.90,
        latent_novelty_d=5.0,
        peak_t_core_predicted=75.0,
        max_pressure_predicted=3.2,
        min_flow_predicted=22.0,
    )
    assert expl.abstained is False
    assert expl.abstention_type == "NONE"
    rt_metric = next(m for m in expl.evidence if "Thermal" in m.metric_name)
    assert rt_metric.is_trigger is False
    assert rt_metric.excess == pytest.approx(0.0, abs=1e-3)
    assert rt_metric.direction == "WITHIN_LIMITS"


def test_residual_just_below_threshold():
    """4. Residual just below threshold (R_T = 5.95) passes normally."""
    expl = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=5.95,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=1.85,
        threshold_residual_8d=1.90,
        latent_novelty_d=12.0,
        peak_t_core_predicted=75.0,
        max_pressure_predicted=3.2,
        min_flow_predicted=22.0,
    )
    assert expl.abstained is False
    assert expl.abstention_type == "NONE"
    assert all(not m.is_trigger for m in expl.evidence)


def test_residual_just_above_threshold():
    """5. Residual just above threshold (R_T = 6.05) triggers MODEL_ABSTAIN."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=6.05,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=1.20,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MODEL_ABSTAIN"
    rt_metric = next(m for m in expl.evidence if "Thermal" in m.metric_name)
    assert rt_metric.is_trigger is True
    assert rt_metric.excess == pytest.approx(0.05, abs=1e-3)


def test_latent_novelty_abstention():
    """6. Latent novelty distance exceeding support limit (D_latent > 15.0) triggers LATENT_NOVELTY."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=1.20,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=0.85,
        threshold_residual_8d=1.90,
        latent_novelty_d=18.4,
        threshold_latent_novelty=15.0,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "LATENT_NOVELTY"
    assert "D_latent = 18.40 exceeds support threshold 15.00" in expl.primary_reason
    assert expl.affected_decision == "CANDIDATE_SEARCH_ABORTED"

    nov_metric = next(m for m in expl.evidence if "Latent" in m.metric_name)
    assert nov_metric.is_trigger is True
    assert nov_metric.excess == pytest.approx(3.40, abs=1e-2)


def test_latent_boundary_exact():
    """7. Latent distance exactly at support limit (D_latent == 15.0) passes without trigger."""
    expl = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=1.20,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=0.85,
        threshold_residual_8d=1.90,
        latent_novelty_d=15.00,
        threshold_latent_novelty=15.0,
        peak_t_core_predicted=70.0,
        max_pressure_predicted=3.0,
        min_flow_predicted=20.0,
    )
    assert expl.abstained is False
    assert expl.abstention_type == "NONE"
    nov_metric = next(m for m in expl.evidence if "Latent" in m.metric_name)
    assert nov_metric.is_trigger is False
    assert nov_metric.excess == pytest.approx(0.0, abs=1e-3)


def test_missing_thermal_prediction():
    """8. Missing peak thermal prediction triggers MISSING_PREDICTION."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=1.0,
        reconstruction_residual_8d=0.5,
        latent_novelty_d=3.0,
        peak_t_core_predicted=None,
        max_pressure_predicted=3.5,
        min_flow_predicted=22.0,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MISSING_PREDICTION"
    assert "T_core" in expl.primary_reason
    assert expl.affected_decision == "EVALUATION_WITHHELD"


def test_missing_pressure_prediction():
    """9. Missing pressure prediction triggers MISSING_PREDICTION."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=1.0,
        reconstruction_residual_8d=0.5,
        latent_novelty_d=3.0,
        peak_t_core_predicted=75.0,
        max_pressure_predicted=None,
        min_flow_predicted=22.0,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MISSING_PREDICTION"
    assert "P_sys" in expl.primary_reason


def test_missing_flow_prediction():
    """10. Missing coolant flow prediction triggers MISSING_PREDICTION."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=1.0,
        reconstruction_residual_8d=0.5,
        latent_novelty_d=3.0,
        peak_t_core_predicted=75.0,
        max_pressure_predicted=3.2,
        min_flow_predicted=None,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MISSING_PREDICTION"
    assert "F_cool" in expl.primary_reason


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_predictions_fail_closed(bad_val):
    """10B. Non-finite predictions (NaN, +Inf, -Inf) are rejected and trigger MISSING_PREDICTION."""
    # Bad thermal
    expl_t = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=1.0,
        peak_t_core_predicted=bad_val,
        max_pressure_predicted=3.2,
        min_flow_predicted=20.0,
    )
    assert expl_t.abstained is True
    assert expl_t.abstention_type == "MISSING_PREDICTION"
    assert "T_core" in expl_t.primary_reason

    # Bad pressure
    expl_p = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=1.0,
        peak_t_core_predicted=70.0,
        max_pressure_predicted=bad_val,
        min_flow_predicted=20.0,
    )
    assert expl_p.abstained is True
    assert expl_p.abstention_type == "MISSING_PREDICTION"
    assert "P_sys" in expl_p.primary_reason

    # Bad flow
    expl_f = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=1.0,
        peak_t_core_predicted=70.0,
        max_pressure_predicted=3.2,
        min_flow_predicted=bad_val,
    )
    assert expl_f.abstained is True
    assert expl_f.abstention_type == "MISSING_PREDICTION"
    assert "F_cool" in expl_f.primary_reason



def test_multiple_abstention_reasons_priority():
    """11. When both residual and latent novelty breach, MODEL_ABSTAIN takes priority and all triggers are recorded."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=25.0,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=3.10,
        threshold_residual_8d=1.90,
        latent_novelty_d=22.0,
        threshold_latent_novelty=15.0,
    )
    assert expl.abstained is True
    assert expl.abstention_type == "MODEL_ABSTAIN"  # High priority over latent novelty

    triggers = [m for m in expl.evidence if m.is_trigger]
    assert len(triggers) == 3
    trigger_names = {t.metric_name for t in triggers}
    assert any("Thermal" in n for n in trigger_names)
    assert any("8D" in n for n in trigger_names)
    assert any("Latent" in n for n in trigger_names)


def test_deterministic_reason_ordering():
    """12. Explanations maintain deterministic reason construction and ordering."""
    expl1 = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=30.0,
        reconstruction_residual_8d=2.5,
        latent_novelty_d=18.0,
        timestamp_utc="2026-09-11T12:00:00Z",
    )
    expl2 = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=30.0,
        reconstruction_residual_8d=2.5,
        latent_novelty_d=18.0,
        timestamp_utc="2026-09-11T12:00:00Z",
    )
    assert expl1.primary_reason == expl2.primary_reason
    assert [m.metric_name for m in expl1.evidence] == [m.metric_name for m in expl2.evidence]


def test_blocked_recommendation_and_actions():
    """13. Abstention explicitly documents blocked candidate selection."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
    )
    assert len(expl.blocked_actions) > 0
    assert "candidate_intervention_selection" in expl.blocked_actions


def test_blocked_counterfactual_semantics():
    """14. Abstention explicitly blocks counterfactual recommendation emission."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
    )
    assert "counterfactual_recommendation_emission" in expl.blocked_actions


def test_safety_implication_distinction():
    """15. Safety implication clarifies that model is untrusted rather than physical plant violated."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
    )
    assert "untrusted" in expl.safety_implication.lower()
    assert "ungrounded" in expl.safety_implication.lower()


def test_recommended_next_step_operator_action():
    """16. Recommended next step provides actionable guidance for human-in-the-loop."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
    )
    assert "operator" in expl.recommended_next_step.lower()
    assert len(expl.recommended_next_step) > 20


def test_deterministic_provenance_and_hash():
    """17. Deterministic SHA-256 hash reproduces consistently across identical calls."""
    kwargs = dict(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
        threshold_residual_t_core=6.00,
        reconstruction_residual_8d=1.20,
        source_evidence_hash="source_hash_999",
        scenario_id="scenario_06",
        timestamp_utc="2026-09-11T12:00:00Z",
    )
    expl1 = build_abstention_explanation(**kwargs)
    expl2 = build_abstention_explanation(**kwargs)
    assert expl1.provenance.abstention_hash == expl2.provenance.abstention_hash
    assert len(expl1.provenance.abstention_hash) == 64
    assert expl1.provenance.source_evidence_hash == "source_hash_999"


def test_json_serialization_roundtrip():
    """18. AbstentionExplanation serializes to valid JSON dict with all keys."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
    )
    json_str = expl.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["abstained"] is True
    assert parsed["abstention_type"] == "MODEL_ABSTAIN"
    assert len(parsed["evidence"]) == 3
    assert "abstention_hash" in parsed["provenance"]


def test_markdown_formatting_abstained():
    """19. Markdown audit report formats abstained state with full evidence table."""
    expl = build_abstention_explanation(
        is_abstained=True,
        reconstruction_residual_t_core=33.88,
        threshold_residual_t_core=6.00,
    )
    md = expl.format_markdown()
    assert "# 🛡️ PRISM Decision Abstention Audit" in md
    assert "MODEL_ABSTAIN" in md
    assert "Thermal Reconstruction Residual (R_T)" in md
    assert "33.88 °C" in md
    assert "TRIGGER" in md


def test_markdown_formatting_non_abstained():
    """20. Markdown formatting handles trusted non-abstained operational state."""
    expl = build_abstention_explanation(
        is_abstained=False,
        reconstruction_residual_t_core=1.2,
        peak_t_core_predicted=70.0,
        max_pressure_predicted=3.0,
        min_flow_predicted=20.0,
    )
    md = expl.format_markdown()
    assert "NO_ABSTENTION" in md
    assert "Model Trusted" in md


def test_build_from_decision_evidence_integration():
    """21. Seamless construction from root DecisionEvidence contract."""
    dummy_decision = DecisionSummary(
        recommendation=None,
        decision_status="BLOCKED",
        abstention_reason="Thermal reconstruction residual R_T = 33.88°C exceeded threshold 6.00°C",
    )
    dummy_trust = TrustEvidence(
        trust_state="MODEL_ABSTAIN",
        reconstruction_residual_t_core=33.88,
        reconstruction_residual_8d=1.82,
        latent_novelty_d=2.45,
        epistemic_uncertainty_sigma=0.0,
        tau_residual_t=6.00,
        tau_novelty=15.0,
        trust_reason="Severe observation/latent inconsistency detected.",
    )
    dummy_prov = ProvenanceEvidence(
        model_version="baseline_005",
        dataset_version="v2.1",
        planner_version="v1.2",
        benchmark_version="v1.0",
        scenario_id="scenario_06",
        timestamp_utc="2026-09-11T12:00:00Z",
        decision_hash="parent_decision_hash_s6",
    )

    ev_obj = DecisionEvidence(
        decision=dummy_decision,
        trust=dummy_trust,
        causal_evidence=CausalEvidence(
            intervention_target="none",
            intervention_value=None,
            affected_variables=[],
            causal_path=[],
            direction_of_effect="NONE",
            counterfactual_comparison={},
        ),
        predicted_outcome=PredictedOutcome(
            peak_t_core=None,
            max_pressure=None,
            min_flow=None,
            mean_cpu_load=None,
            mean_electrical_power=None,
            planning_horizon=0,
        ),
        safety_evidence=SafetyEvidence(
            safety_state="ABSTAIN_REQUIRED",
            thermal_margin_c=None,
            pressure_margin_bar=None,
            flow_margin_l_min=None,
            latent_support_margin=None,
            is_safe=False,
        ),
        decision_quality=DecisionQuality(
            utility_score=None,
            second_best_candidate=None,
            second_best_utility=None,
            decision_margin=None,
        ),
        provenance=dummy_prov,
    )

    abst_expl = build_abstention_explanation_from_decision_evidence(ev_obj)
    assert abst_expl.abstained is True
    assert abst_expl.abstention_type == "MODEL_ABSTAIN"
    assert abst_expl.provenance.source_evidence_hash == "parent_decision_hash_s6"
    assert abst_expl.provenance.scenario_id == "scenario_06"
    assert "Thermal reconstruction residual" in abst_expl.primary_reason
