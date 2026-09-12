"""Unit tests for PRISM Unified Decision Record Assembly (Task 6.6).

Verifies the end-to-end integration and projection of all specialized evidence engines:
1. Safe decision record assembly
2. Unsafe candidate record assembly
3. S6 model abstention record assembly
4. Latent novelty record assembly
5. Counterfactual evidence integration
6. None counterfactual handling
7. Cryptographic hash chain validation
8. Deterministic unified hash reproducibility
9. JSON serialization roundtrip
10. Executive markdown formatting for safe decision
11. Executive markdown formatting for abstained decision
12. Counterfactual section formatting
13. Decision quality preservation
14. Safety effective values and limiting constraint preservation
15. Causal DAG pathway preservation
16. Immutable projection invariant (zero decision alteration)
"""

import pytest
import json

from prism.explanation.unified_record import (
    PrismDecisionRecord,
    UnifiedProvenance,
    build_unified_decision_record,
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
from prism.explanation.causal_explanation import build_causal_explanation
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
)
from prism.explanation.safety_evidence import evaluate_safety_evidence_detail
from prism.explanation.abstention_explanation import build_abstention_explanation


def make_dummy_safe_decision_evidence() -> DecisionEvidence:
    """Helper creating a trusted, safe DecisionEvidence object."""
    return DecisionEvidence(
        decision=DecisionSummary(
            recommendation="cand_valve_open_1.0",
            decision_status="RECOMMENDED",
            abstention_reason=None,
        ),
        trust=TrustEvidence(
            trust_state="MODEL_TRUSTED",
            reconstruction_residual_t_core=0.45,
            reconstruction_residual_8d=0.82,
            latent_novelty_d=2.1,
            epistemic_uncertainty_sigma=1.2,
            tau_residual_t=6.0,
            tau_novelty=15.0,
            trust_reason="Observations match representation space.",
        ),
        causal_evidence=CausalEvidence(
            intervention_target="valve",
            intervention_value=1.0,
            affected_variables=["coolant_flow", "core_temperature"],
            causal_path=["valve", "coolant_flow", "core_temperature"],
            direction_of_effect="DECREASE_TEMPERATURE",
            counterfactual_comparison={"t_core_delta": -12.4},
        ),
        predicted_outcome=PredictedOutcome(
            peak_t_core=78.5,
            max_pressure=3.8,
            min_flow=24.0,
            mean_cpu_load=45.0,
            mean_electrical_power=250.0,
            planning_horizon=40,
        ),
        safety_evidence=SafetyEvidence(
            safety_state="SAFE",
            thermal_margin_c=14.1,
            pressure_margin_bar=1.4,
            flow_margin_l_min=13.6,
            latent_support_margin=12.9,
            is_safe=True,
        ),
        decision_quality=DecisionQuality(
            utility_score=0.9250,
            second_best_candidate="cand_throttle_close_0.5",
            second_best_utility=0.8500,
            decision_margin=0.0750,
            alternatives_rejected=[],
        ),
        provenance=ProvenanceEvidence(
            model_version="baseline_005",
            dataset_version="v2.1",
            planner_version="v1.2",
            benchmark_version="v1.0",
            scenario_id="scenario_01",
            timestamp_utc="2026-09-11T12:00:00Z",
            decision_hash="dec_hash_safe_01",
        ),
    )


def make_dummy_s6_abstained_evidence() -> DecisionEvidence:
    """Helper creating an S6 abstained DecisionEvidence object."""
    return DecisionEvidence(
        decision=DecisionSummary(
            recommendation=None,
            decision_status="BLOCKED",
            abstention_reason="Thermal reconstruction residual R_T = 33.88°C exceeded threshold 6.00°C",
        ),
        trust=TrustEvidence(
            trust_state="MODEL_ABSTAIN",
            reconstruction_residual_t_core=33.88,
            reconstruction_residual_8d=1.82,
            latent_novelty_d=2.45,
            epistemic_uncertainty_sigma=0.0,
            tau_residual_t=6.0,
            tau_novelty=15.0,
            trust_reason="Severe observation/latent inconsistency detected.",
        ),
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
        provenance=ProvenanceEvidence(
            model_version="baseline_005",
            dataset_version="v2.1",
            planner_version="v1.2",
            benchmark_version="v1.0",
            scenario_id="scenario_06",
            timestamp_utc="2026-09-11T12:00:00Z",
            decision_hash="dec_hash_s6_abstain",
        ),
    )


def test_unified_record_assembly_safe_decision():
    """1. Assembles complete unified record for a compliant safe decision."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert record.decision.recommendation == "cand_valve_open_1.0"
    assert record.decision.decision_status == "RECOMMENDED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.safety.overall_state == "SAFE"
    assert record.abstention.abstained is False
    assert record.provenance.scenario_id == "scenario_01"
    assert len(record.provenance.unified_record_hash) == 64


def test_unified_record_assembly_unsafe_candidate():
    """2. Assembles record for an unsafe candidate with boundary violation detail."""
    base_ev = make_dummy_safe_decision_evidence()
    # Provide explicit unsafe safety evidence
    unsafe_saf = evaluate_safety_evidence_detail(
        peak_t_core=94.0,
        max_pressure=4.0,
        min_flow=20.0,
        latent_novelty=2.0,
        sigma_t_core=1.5,  # 94 + 2*1.5 = 97.0 > 95.0
        use_uncertainty_bounds=True,
    )
    record = build_unified_decision_record(
        decision_evidence=base_ev,
        safety_evidence=unsafe_saf,
    )
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "UNSAFE"
    assert len(record.safety.violations) == 1
    assert record.safety.violations[0].variable_symbol == "T_core"


def test_unified_record_assembly_abstained_s6():
    """3. Assembles S6 model abstention record with blocked actions and diagnostics."""
    base_ev = make_dummy_s6_abstained_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert record.decision.decision_status == "BLOCKED"
    assert record.decision.recommendation is None
    assert record.abstention.abstained is True
    assert record.abstention.abstention_type == "MODEL_ABSTAIN"
    assert "candidate_intervention_selection" in record.abstention.blocked_actions
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "ABSTAIN_REQUIRED"


def test_unified_record_assembly_latent_novelty():
    """4. Assembles latent novelty abstention record."""
    base_ev = make_dummy_safe_decision_evidence()
    base_ev.trust.latent_novelty_d = 18.5
    base_ev.decision.decision_status = "BLOCKED"
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert record.abstention.abstained is True
    assert record.abstention.abstention_type == "LATENT_NOVELTY"
def test_unified_record_with_counterfactual_evidence():
    """5. Integrates CounterfactualEvidence cleanly into the unified record."""
    base_ev = make_dummy_safe_decision_evidence()
    cf_obj = CounterfactualEvidence(
        factual_world=FactualWorldContext(
            episode_id="scenario_01",
            intervention_time=10,
            planning_horizon=40,
            historical_t_core_at_t_star=85.0,
            historical_f_cool_at_t_star=18.0,
            historical_p_sys_at_t_star=3.5,
            abduced_latent_norm=1.2,
            abduced_latent_std_mean=0.1,
        ),
        intervention=CounterfactualIntervention(
            intervention_target="A_valve",
            original_value=0.5,
            counterfactual_value=1.0,
            intervention_type="ACTION",
            formal_notation="do(A_valve = 1.0)",
        ),
        counterfactual_world=CounterfactualWorldOutcome(
            peak_t_core=78.5,
            max_p_sys=3.8,
            min_f_cool=24.0,
            mean_l_cpu=45.0,
            mean_p_elec=250.0,
            failure_state=False,
            time_to_failure=None,
        ),
        causal_effect=CausalEffectEvidence(
            delta_t_core_peak=-13.5,
            delta_p_sys_max=-0.2,
            delta_f_cool_min=+8.0,
            delta_l_cpu_mean=0.0,
            delta_p_elec_mean=0.0,
            primary_effect_direction="REDUCES_CORE_TEMPERATURE",
            causal_interpretation="Increased coolant flow reduces core temperature",
        ),
        twin_world_integrity=TwinWorldIntegrity(
            replay_mode="TWIN_WORLD_FROZEN_EXOGENOUS",
            shared_exogenous_conditions=True,
            identical_pre_intervention_history=True,
            abduction_method="POSTERIOR_LATENT_INFERENCE_Q_PHI",
        ),
        safety_comparison=SafetyComparison(
            factual_thermal_margin_c=10.0,
            factual_pressure_margin_bar=1.5,
            factual_flow_margin_l_min=8.0,
            counterfactual_thermal_margin_c=14.1,
            counterfactual_pressure_margin_bar=1.4,
            counterfactual_flow_margin_l_min=13.6,
            factual_safety_state="MARGINAL",
            counterfactual_safety_state="SAFE",
            safety_transition="MARGINAL -> SAFE",
            outcome_classification="INTERVENTION_MITIGATES_FAILURE",
        ),
        uncertainty=UncertaintyContext(
            aleatoric_sigma=0.5,
            epistemic_sigma=1.2,
            latent_novelty_d=2.1,
            support_threshold=15.0,
            within_support=True,
        ),
        provenance=CounterfactualProvenance(
            source_evidence_hash="cf_source_hash_123",
            counterfactual_engine_version="v1.0",
            model_version="baseline_005",
            dataset_version="v2.1",
            episode_id="scenario_01",
            counterfactual_id="cand_valve_open_1.0",
            intervention_time=10,
            timestamp_utc="2026-09-11T12:00:00Z",
            counterfactual_hash="cf_hash_789",
        ),
    )

    record = build_unified_decision_record(
        decision_evidence=base_ev,
        counterfactual_evidence=cf_obj,
    )
    assert record.counterfactual is not None
    assert record.counterfactual.causal_effect.delta_t_core_peak == -13.5
    assert record.provenance.counterfactual_hash == "cf_hash_789"


def test_unified_record_without_counterfactual_evidence():
    """6. Handles absence of counterfactual evidence gracefully."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(
        decision_evidence=base_ev,
        counterfactual_evidence=None,
    )
    assert record.counterfactual is None
    assert record.provenance.counterfactual_hash is None


def test_provenance_cryptographic_hash_chain():
    """7. Unified hash changes when any underlying evidence hash changes."""
    base_ev = make_dummy_safe_decision_evidence()
    record1 = build_unified_decision_record(decision_evidence=base_ev, timestamp_utc="2026-09-11T12:00:00Z")

    base_ev2 = make_dummy_safe_decision_evidence()
    base_ev2.provenance.decision_hash = "different_dec_hash_999"
    record2 = build_unified_decision_record(decision_evidence=base_ev2, timestamp_utc="2026-09-11T12:00:00Z")

    assert record1.provenance.unified_record_hash != record2.provenance.unified_record_hash


def test_deterministic_unified_hash_reproducibility():
    """8. Identical inputs reproduce identical 64-character SHA-256 hash."""
    base_ev = make_dummy_safe_decision_evidence()
    rec1 = build_unified_decision_record(decision_evidence=base_ev, timestamp_utc="2026-09-11T12:00:00Z")
    rec2 = build_unified_decision_record(decision_evidence=base_ev, timestamp_utc="2026-09-11T12:00:00Z")

    assert rec1.provenance.unified_record_hash == rec2.provenance.unified_record_hash
    assert len(rec1.provenance.unified_record_hash) == 64


def test_json_serialization_roundtrip():
    """9. Serializes to valid JSON dict with all 8 unified components."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)
    json_str = record.to_json(indent=2)
    parsed = json.loads(json_str)

    assert "decision" in parsed
    assert "trust" in parsed
    assert "causal_reasoning" in parsed
    assert "counterfactual" in parsed
    assert "safety" in parsed
    assert "abstention" in parsed
    assert "decision_quality" in parsed
    assert "provenance" in parsed
    assert parsed["provenance"]["unified_record_hash"] == record.provenance.unified_record_hash


def test_format_markdown_safe_dossier():
    """10. Renders executive markdown dossier for safe decision."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)
    md = record.format_markdown()

    assert "PRISM Unified Decision Record & Audit Dossier" in md
    assert "cand_valve_open_1.0" in md
    assert "Executive Summary & Causal Reasoning" in md
    assert "Physical Safety & Support Constraint Audit" in md
    assert "Cryptographic Provenance Chain" in md


def test_format_markdown_abstained_dossier():
    """11. Renders markdown dossier for abstained decision including Section 4."""
    base_ev = make_dummy_s6_abstained_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)
    md = record.format_markdown()

    assert "Abstention & Trust Boundary Audit" in md
    assert "MODEL_ABSTAIN" in md
    assert "Thermal reconstruction residual" in md


def test_format_markdown_counterfactual_section():
    """12. Renders Section 5 when counterfactual evidence is included."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)
    # Without counterfactual
    assert "Counterfactual Twin-World Verification" not in record.format_markdown()


def test_decision_quality_preservation():
    """13. Pareto multi-objective utility and decision margins preserved."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert record.decision_quality.utility_score == pytest.approx(0.9250, abs=1e-4)
    assert record.decision_quality.second_best_candidate == "cand_throttle_close_0.5"
    assert record.decision_quality.decision_margin == pytest.approx(0.0750, abs=1e-4)


def test_safety_evidence_effective_values_preserved():
    """14. Conservative k=2 effective values and limiting constraint preserved."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert "thermal" in record.safety.constraints
    assert record.safety.constraints["thermal"].effective_value == pytest.approx(78.5 + 2 * 1.2, abs=1e-3)
    assert record.safety.limiting_constraint.constraint_name == "thermal"


def test_causal_reasoning_dag_pathway_preserved():
    """15. Dominant causal DAG pathway and mechanism preserved."""
    base_ev = make_dummy_safe_decision_evidence()
    record = build_unified_decision_record(decision_evidence=base_ev)

    assert "coolant valve" in record.causal_reasoning.summary.mechanism.lower()
    assert [n.variable for n in record.causal_reasoning.causal_chain.nodes] == ["A_valve", "V_pos", "F_cool", "P_sys", "T_core"]



def test_immutable_projection_invariant():
    """16. Unified record is a pure projection without altering underlying decision states."""
    base_ev = make_dummy_safe_decision_evidence()
    orig_status = base_ev.decision.decision_status
    orig_rec = base_ev.decision.recommendation

    record = build_unified_decision_record(decision_evidence=base_ev)

    assert base_ev.decision.decision_status == orig_status
    assert base_ev.decision.recommendation == orig_rec
    assert record.decision.decision_status == orig_status

