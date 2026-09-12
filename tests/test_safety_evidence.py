"""Unit tests for PRISM Safety Evidence Engine (Task 6.4 & 6.4A).

Comprehensive test suite (22 tests) verifying:
1. Thermal pass
2. Thermal exact boundary (T_eff = 95.0 -> UNSAFE)
3. Thermal violation
4. Pressure pass
5. Pressure exact boundary (P_eff = 5.50 -> UNSAFE)
6. Pressure violation
7. Flow pass
8. Flow exact boundary (F_eff = 8.00 -> UNSAFE)
9. Flow violation
10. Latent-support pass (D_latent < 15.0 -> PASS, margin > 0)
11. Latent-support exact boundary (D_latent = 15.0 -> PASS, margin == 0.0)
12. Latent-support violation (D_latent > 15.0 -> ABSTAIN_REQUIRED, margin < 0)
13. Uncertainty-adjusted thermal (k=2: T_eff = μ + 2σ)
14. Uncertainty-adjusted pressure (k=2: P_eff = μ + 2σ)
15. Uncertainty-adjusted flow (k=2: F_eff = μ - 2σ)
16. Limiting-constraint identification for safe candidates (min normalized headroom)
17. Limiting-constraint identification for unsafe candidates (max dimensionless normalized violation score)
18. Limiting-constraint deterministic tie-breaking
19. Multi-constraint violation reporting
20. Abstention state fail-closed behavior
21. Deterministic SHA-256 safety hash and provenance
22. DecisionEvidence contract integration
"""

import pytest
import numpy as np

from prism.explanation.safety_evidence import (
    SafetyEvidenceDetail,
    ConstraintDetail,
    SafetyViolationDetail,
    LimitingConstraintSummary,
    SafetyProvenance,
    evaluate_safety_evidence_detail,
    build_safety_evidence_from_decision_evidence,
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
from prism.planning.safety_constraints import SafetyViolationReason


def test_thermal_pass():
    """1. Thermal constraint passes when T_eff < 95.0°C."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=78.5,
        max_pressure=3.2,
        min_flow=25.0,
        latent_novelty=2.1,
        sigma_t_core=1.0,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["thermal"]
    assert c.is_passed is True
    assert c.raw_prediction == 78.5
    assert c.uncertainty_sigma == 1.0
    assert c.effective_value == pytest.approx(80.5, abs=1e-3)
    assert c.threshold == 95.0
    assert c.margin == pytest.approx(95.0 - 80.5, abs=1e-3)
    assert c.margin > 0.0
    assert c.violation_message is None


def test_thermal_exact_boundary():
    """2. Thermal constraint fails at exact boundary T_eff == 95.0°C (strict inequality)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=95.0,
        max_pressure=3.0,
        min_flow=25.0,
        latent_novelty=1.0,
        sigma_t_core=0.0,
    )
    c = evidence.constraints["thermal"]
    assert c.is_passed is False
    assert c.effective_value == 95.0
    assert c.margin == pytest.approx(0.0, abs=1e-3)
    assert evidence.is_safe is False
    assert len(evidence.violations) == 1
    assert evidence.violations[0].constraint_code == SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value


def test_thermal_violation():
    """3. Thermal constraint fails when T_eff > 95.0°C."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=96.2,
        max_pressure=3.0,
        min_flow=25.0,
        latent_novelty=2.0,
        sigma_t_core=0.5,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["thermal"]
    assert c.is_passed is False
    assert c.effective_value == pytest.approx(97.2, abs=1e-3)
    assert c.margin == pytest.approx(95.0 - 97.2, abs=1e-3)
    assert c.margin < 0.0
    assert "exceeds thermal limit" in c.violation_message
    assert evidence.is_safe is False
    assert evidence.overall_state == "UNSAFE"


def test_pressure_pass():
    """4. Pressure constraint passes when P_eff < 5.5 bar."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=75.0,
        max_pressure=4.1,
        min_flow=22.0,
        latent_novelty=1.5,
        sigma_pressure=0.2,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["pressure"]
    assert c.is_passed is True
    assert c.raw_prediction == 4.1
    assert c.effective_value == pytest.approx(4.5, abs=1e-3)
    assert c.threshold == 5.5
    assert c.margin == pytest.approx(5.5 - 4.5, abs=1e-3)
    assert c.margin > 0.0


def test_pressure_exact_boundary():
    """5. Pressure constraint fails at exact boundary P_eff == 5.50 bar (strict inequality)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=75.0,
        max_pressure=5.50,
        min_flow=20.0,
        latent_novelty=1.0,
        sigma_pressure=0.0,
    )
    c = evidence.constraints["pressure"]
    assert c.is_passed is False
    assert c.effective_value == 5.50
    assert c.margin == pytest.approx(0.0, abs=1e-3)
    assert evidence.is_safe is False
    assert len(evidence.violations) == 1
    assert evidence.violations[0].constraint_code == SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value


def test_pressure_violation():
    """6. Pressure constraint fails when P_eff > 5.5 bar."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=75.0,
        max_pressure=5.4,
        min_flow=20.0,
        latent_novelty=1.5,
        sigma_pressure=0.15,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["pressure"]
    assert c.is_passed is False
    assert c.effective_value == pytest.approx(5.7, abs=1e-3)
    assert c.margin == pytest.approx(5.5 - 5.7, abs=1e-3)
    assert c.margin < 0.0
    assert "exceeds pressure limit" in c.violation_message
    assert evidence.is_safe is False


def test_flow_pass():
    """7. Flow constraint passes when F_eff > 8.0 L/min."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=18.0,
        latent_novelty=1.0,
        sigma_flow=1.5,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["flow"]
    assert c.is_passed is True
    assert c.raw_prediction == 18.0
    assert c.effective_value == pytest.approx(15.0, abs=1e-3)
    assert c.threshold == 8.0
    assert c.margin == pytest.approx(15.0 - 8.0, abs=1e-3)
    assert c.margin > 0.0


def test_flow_exact_boundary():
    """8. Flow constraint fails at exact boundary F_eff == 8.00 L/min (strict inequality)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=8.00,
        latent_novelty=1.0,
        sigma_flow=0.0,
    )
    c = evidence.constraints["flow"]
    assert c.is_passed is False
    assert c.effective_value == 8.00
    assert c.margin == pytest.approx(0.0, abs=1e-3)
    assert evidence.is_safe is False
    assert len(evidence.violations) == 1
    assert evidence.violations[0].constraint_code == SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value


def test_flow_violation():
    """9. Flow constraint fails when F_eff < 8.0 L/min."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=9.5,
        latent_novelty=1.0,
        sigma_flow=1.0,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["flow"]
    assert c.is_passed is False
    assert c.effective_value == pytest.approx(7.5, abs=1e-3)
    assert c.margin == pytest.approx(7.5 - 8.0, abs=1e-3)
    assert c.margin < 0.0
    assert "violates flow minimum" in c.violation_message
    assert evidence.is_safe is False


def test_latent_support_pass():
    """10. Latent support passes when D_latent < 15.0."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=20.0,
        latent_novelty=6.5,
    )
    c = evidence.constraints["latent_support"]
    assert c.is_passed is True
    assert c.raw_prediction == 6.5
    assert c.effective_value == 6.5
    assert c.threshold == 15.0
    assert c.margin == pytest.approx(8.5, abs=1e-3)
    assert c.margin > 0.0


def test_latent_support_exact_boundary():
    """11. Latent support passes at exact boundary D_latent == 15.0 with margin == 0.0."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=20.0,
        latent_novelty=15.0,
    )
    c = evidence.constraints["latent_support"]
    assert c.is_passed is True
    assert c.effective_value == 15.0
    assert c.margin == pytest.approx(0.0, abs=1e-3)
    assert evidence.is_safe is True
    assert evidence.overall_state != "ABSTAIN_REQUIRED"


def test_latent_support_violation():
    """12. Latent support fails when D_latent > 15.0, triggering ABSTAIN_REQUIRED."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=70.0,
        max_pressure=3.0,
        min_flow=20.0,
        latent_novelty=18.4,
    )
    c = evidence.constraints["latent_support"]
    assert c.is_passed is False
    assert c.effective_value == 18.4
    assert c.margin == pytest.approx(15.0 - 18.4, abs=1e-3)
    assert c.margin < 0.0
    assert "exceeds support limit" in c.violation_message

    assert evidence.is_safe is False
    assert evidence.overall_state == "ABSTAIN_REQUIRED"
    assert len(evidence.violations) == 1
    assert evidence.violations[0].constraint_code == SafetyViolationReason.LATENT_NOVELTY_EXCEEDED.value
    assert evidence.violations[0].severity == "ABSTAIN_REQUIRED"


def test_uncertainty_adjusted_thermal():
    """13. Raw thermal passes (93.0 < 95.0) but k=2 adjusted fails (93.0 + 2*1.5 = 96.0 >= 95.0)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=93.0,
        max_pressure=3.0,
        min_flow=20.0,
        latent_novelty=1.0,
        sigma_t_core=1.5,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["thermal"]
    assert c.raw_prediction == 93.0
    assert c.effective_value == pytest.approx(96.0, abs=1e-3)
    assert c.is_passed is False
    assert evidence.is_safe is False


def test_uncertainty_adjusted_pressure():
    """14. Raw pressure passes (5.3 < 5.5) but k=2 adjusted fails (5.3 + 2*0.2 = 5.7 >= 5.5)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=75.0,
        max_pressure=5.3,
        min_flow=20.0,
        latent_novelty=1.0,
        sigma_pressure=0.2,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["pressure"]
    assert c.raw_prediction == 5.3
    assert c.effective_value == pytest.approx(5.7, abs=1e-3)
    assert c.is_passed is False
    assert evidence.is_safe is False


def test_uncertainty_adjusted_flow():
    """15. Raw flow passes (9.0 > 8.0) but k=2 adjusted fails (9.0 - 2*1.0 = 7.0 <= 8.0)."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=75.0,
        max_pressure=3.0,
        min_flow=9.0,
        latent_novelty=1.0,
        sigma_flow=1.0,
        use_uncertainty_bounds=True,
        k_multiplier=2.0,
    )
    c = evidence.constraints["flow"]
    assert c.raw_prediction == 9.0
    assert c.effective_value == pytest.approx(7.0, abs=1e-3)
    assert c.is_passed is False
    assert evidence.is_safe is False


def test_limiting_constraint_safe_candidates():
    """16. Identifies the limiting constraint via normalized headroom among safe candidates."""
    # Thermal is tightest: margin = 3.0°C / 75°C span = 4.0% headroom
    # Pressure: margin = 3.0 bar / 5.0 bar span = 60.0% headroom
    # Flow: margin = 27.0 L/min / 52.0 L/min span = 51.9% headroom
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=92.0,
        max_pressure=2.5,
        min_flow=35.0,
        latent_novelty=2.0,
        sigma_t_core=0.0,
        sigma_pressure=0.0,
        sigma_flow=0.0,
    )
    assert evidence.is_safe is True
    assert evidence.limiting_constraint.constraint_name == "thermal"
    assert evidence.limiting_constraint.variable_symbol == "T_core"
    assert evidence.limiting_constraint.raw_margin == pytest.approx(3.0, abs=1e-3)
    assert evidence.limiting_constraint.normalized_headroom_pct == pytest.approx(4.0, abs=1e-2)


def test_limiting_constraint_unsafe_dimensionless_normalization():
    """17. Unsafe limiting constraint uses dimensionless normalized violation score, avoiding raw unit bias."""
    # Scenario: Pressure overshoots by 0.5 bar (span 5.0 bar -> V_P = 0.5/5.0 = 0.10 = 10%)
    # Thermal overshoots by 1.5 °C (span 75.0 °C -> V_T = 1.5/75.0 = 0.02 = 2%)
    # In raw units, 1.5 > 0.5. But in dimensionless normalized violation, V_P (10%) > V_T (2%).
    # Therefore, Pressure must be correctly identified as the limiting constraint!
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=96.5,  # 96.5 - 95.0 = 1.5 °C breach (V_T = 0.020)
        max_pressure=6.0,  # 6.0 - 5.5 = 0.5 bar breach (V_P = 0.100)
        min_flow=25.0,     # safe
        latent_novelty=2.0,
        sigma_t_core=0.0,
        sigma_pressure=0.0,
        sigma_flow=0.0,
    )
    assert evidence.is_safe is False
    assert evidence.limiting_constraint.constraint_name == "pressure"
    assert evidence.limiting_constraint.variable_symbol == "P_sys"
    assert evidence.limiting_constraint.normalized_violation_score == pytest.approx(0.10, abs=1e-3)
    assert evidence.limiting_constraint.raw_margin == pytest.approx(-0.5, abs=1e-3)


def test_limiting_constraint_tie_breaking():
    """18. Deterministic tie-breaking across equal normalized violations."""
    # Both thermal and pressure breach by exactly 10% of their span:
    # Thermal: 95.0 + 0.10 * 75.0 = 102.5 °C (breach 7.5 °C, V_T = 0.10)
    # Pressure: 5.50 + 0.10 * 5.0 = 6.00 bar (breach 0.5 bar, V_P = 0.10)
    # Canonical tie-breaker selects "thermal" first.
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=102.5,
        max_pressure=6.0,
        min_flow=25.0,
        latent_novelty=2.0,
        sigma_t_core=0.0,
        sigma_pressure=0.0,
        sigma_flow=0.0,
    )
    assert evidence.is_safe is False
    assert evidence.limiting_constraint.constraint_name == "thermal"
    assert evidence.limiting_constraint.normalized_violation_score == pytest.approx(0.10, abs=1e-3)


def test_multi_constraint_violation():
    """19. Multi-constraint violation accurately captures all breaches without early-exit truncation."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=98.0,
        max_pressure=5.8,
        min_flow=6.5,
        latent_novelty=1.0,
        sigma_t_core=0.0,
        sigma_pressure=0.0,
        sigma_flow=0.0,
    )
    assert evidence.is_safe is False
    assert len(evidence.violations) == 3
    violation_codes = {v.constraint_code for v in evidence.violations}
    assert SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value in violation_codes
    assert SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value in violation_codes
    assert SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value in violation_codes


def test_abstention_state_fail_closed():
    """20. Abstention state from trust gateway sets ABSTAIN_REQUIRED with fail-closed non-evaluation."""
    evidence = evaluate_safety_evidence_detail(
        peak_t_core=None,
        max_pressure=None,
        min_flow=None,
        latent_novelty=None,
        is_model_abstained=True,
        abstention_reason="Thermal reconstruction residual R_T = 33.88°C exceeded threshold 6.00°C",
    )
    assert evidence.overall_state == "ABSTAIN_REQUIRED"
    assert evidence.is_safe is False
    assert evidence.limiting_constraint.constraint_name == "model_trust_gateway"
    assert evidence.constraints["thermal"].effective_value is None
    assert evidence.constraints["thermal"].margin is None
    assert evidence.constraints["thermal"].is_passed is False
    assert "MODEL_ABSTAIN" in evidence.constraints["thermal"].violation_message


def test_deterministic_output_and_provenance():
    """21. Deterministic hashing and immutable provenance consistency."""
    kwargs = dict(
        peak_t_core=84.2,
        max_pressure=4.1,
        min_flow=19.5,
        latent_novelty=3.2,
        sigma_t_core=0.8,
        sigma_pressure=0.1,
        sigma_flow=0.5,
        source_evidence_hash="test_source_hash_abc123",
        scenario_id="scenario_01",
        candidate_id="cand_pump_02",
        timestamp_utc="2026-09-11T12:00:00Z",
    )
    ev1 = evaluate_safety_evidence_detail(**kwargs)
    ev2 = evaluate_safety_evidence_detail(**kwargs)

    assert ev1.provenance.safety_hash == ev2.provenance.safety_hash
    assert len(ev1.provenance.safety_hash) == 64
    assert ev1.provenance.source_evidence_hash == "test_source_hash_abc123"
    assert ev1.provenance.scenario_id == "scenario_01"

    # Verify markdown formatting renders properly
    md = ev1.format_markdown()
    assert "PRISM Safety Audit & Constraint Evidence" in md
    assert "Physical Safety Constraints" in md
    assert "Model-Support Constraint" in md
    assert "T_core" in md
    assert "P_sys" in md
    assert "F_cool" in md
    assert "D_latent" in md
    assert ev1.provenance.safety_hash in md


def test_build_safety_evidence_from_decision_evidence():
    """22. Seamless extraction and construction from DecisionEvidence."""
    dummy_decision = DecisionSummary(
        recommendation="cand_01",
        decision_status="RECOMMENDED",
        abstention_reason=None,
    )
    dummy_trust = TrustEvidence(
        trust_state="MODEL_TRUSTED",
        reconstruction_residual_t_core=0.45,
        reconstruction_residual_8d=0.82,
        latent_novelty_d=2.1,
        epistemic_uncertainty_sigma=1.2,
        tau_residual_t=6.0,
        tau_novelty=15.0,
        trust_reason="Observations match representation space.",
    )
    dummy_predictions = PredictedOutcome(
        peak_t_core=82.0,
        max_pressure=3.8,
        min_flow=24.0,
        mean_cpu_load=45.0,
        mean_electrical_power=250.0,
        planning_horizon=40,
    )
    dummy_prov = ProvenanceEvidence(
        model_version="baseline_005",
        dataset_version="v2.1",
        planner_version="v1.2",
        benchmark_version="v1.0",
        scenario_id="scenario_01",
        timestamp_utc="2026-09-11T12:00:00Z",
        decision_hash="parent_evidence_hash_999",
    )

    ev_obj = DecisionEvidence(
        decision=dummy_decision,
        trust=dummy_trust,
        causal_evidence=CausalEvidence(
            intervention_target="valve",
            intervention_value=1.0,
            affected_variables=["coolant_flow", "core_temperature"],
            causal_path=["valve", "coolant_flow", "core_temperature"],
            direction_of_effect="DECREASE_TEMPERATURE",
            counterfactual_comparison={},
        ),
        predicted_outcome=dummy_predictions,
        safety_evidence=SafetyEvidence(
            safety_state="SAFE",
            thermal_margin_c=10.6,
            pressure_margin_bar=1.4,
            flow_margin_l_min=14.4,
            latent_support_margin=12.9,
            is_safe=True,
        ),
        decision_quality=DecisionQuality(
            utility_score=0.92,
            second_best_candidate="cand_02",
            second_best_utility=0.85,
            decision_margin=0.07,
        ),
        provenance=dummy_prov,
    )

    safety_detail = build_safety_evidence_from_decision_evidence(ev_obj)
    assert safety_detail.is_safe is True
    assert safety_detail.overall_state == "SAFE"
    assert safety_detail.provenance.source_evidence_hash == "parent_evidence_hash_999"
    assert safety_detail.constraints["thermal"].effective_value == pytest.approx(82.0 + 2 * 1.2, abs=1e-3)
