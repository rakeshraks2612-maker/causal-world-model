"""PRISM Abstention Explanation Engine (Task 6.5).

Provides independent, deterministic, and machine-readable explanations of why
PRISM refused to make a decision or withheld recommendations.

Enforces clear separation:
1. Model Abstention ("I don't trust my prediction because representation residual is high.")
2. Latent Novelty ("Candidate state is out-of-distribution in latent space.")
3. Missing Prediction ("Required telemetry or simulation outcome is missing.")
4. Epistemic Uncertainty ("Model variance exceeds confidence bound.")
5. Safety Evaluation Unavailable ("Downstream safety evidence cannot be computed.")

Ensures abstentions never collapse into a generic "UNSAFE" explanation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional, Any

from prism.explanation.evidence import DecisionEvidence


# =============================================================================
# Machine-Readable Data Structures
# =============================================================================

@dataclass
class AbstentionEvidenceMetric:
    """Quantitative evidence metric driving or evaluating an abstention decision."""
    metric_name: str
    observed_value: Optional[float]
    threshold: Optional[float]
    excess: Optional[float]
    direction: str  # "GREATER_THAN_THRESHOLD", "LESS_THAN_THRESHOLD", "UNAVAILABLE", "WITHIN_LIMITS"
    unit: str
    is_trigger: bool = False


@dataclass
class AbstentionProvenance:
    """Traceable, immutable metadata linking abstention explanations to evidence hashes."""
    source_evidence_hash: Optional[str]
    engine_version: str
    model_version: str
    scenario_id: Optional[str]
    timestamp_utc: str
    abstention_hash: str


@dataclass
class AbstentionExplanation:
    """Root Machine-Readable PRISM Abstention Explanation Contract."""
    abstained: bool
    abstention_type: str  # "MODEL_ABSTAIN", "LATENT_NOVELTY", "EPISTEMIC_UNCERTAINTY", "MISSING_PREDICTION", "SAFETY_EVALUATION_UNAVAILABLE", "NONE"
    headline: str
    primary_reason: str
    evidence: List[AbstentionEvidenceMetric]
    affected_decision: str  # "RECOMMENDATION_BLOCKED", "CANDIDATE_SEARCH_ABORTED", "EVALUATION_WITHHELD", "NO_ABSTENTION"
    blocked_actions: List[str]
    safety_implication: str
    recommended_next_step: str
    provenance: AbstentionProvenance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_markdown(self) -> str:
        """Render a clean, human-auditable markdown report."""
        if not self.abstained:
            lines = [
                "# 🛡️ PRISM Abstention Audit",
                "**Decision Status:** `NO_ABSTENTION` | **Model Trusted:** `True`",
                "",
                "The world model and candidate evaluations are fully trusted. No abstention was triggered.",
                f"**Abstention Hash:** `{self.provenance.abstention_hash}`",
            ]
            return "\n".join(lines)

        lines = [
            "# 🛡️ PRISM Decision Abstention Audit",
            f"**Abstention Type:** `{self.abstention_type}` | **Decision Status:** `{self.affected_decision}`",
            f"**Headline:** {self.headline}",
            "",
            "## 1. Primary Abstention Rationale",
            self.primary_reason,
            "",
            "## 2. Quantitative Diagnostic Evidence",
            "",
            "| Diagnostic Metric | Observed Value | Trust Threshold | Excess Delta | Status | Trigger |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for m in self.evidence:
            obs_str = f"{m.observed_value:.2f} {m.unit}" if m.observed_value is not None else "N/A"
            thr_str = f"{m.threshold:.2f} {m.unit}" if m.threshold is not None else "N/A"
            exc_str = f"{m.excess:+.2f} {m.unit}" if m.excess is not None else "N/A"
            status_str = f"`{m.direction}`"
            trig_str = "🚨 **TRIGGER**" if m.is_trigger else "✅ Normal"
            lines.append(f"| **{m.metric_name}** | {obs_str} | {thr_str} | {exc_str} | {status_str} | {trig_str} |")

        lines.extend([
            "",
            "## 3. Operational Consequence & Safety Implications",
            f"- **Blocked Actions:** {', '.join(f'`{a}`' for a in self.blocked_actions) if self.blocked_actions else 'None'}",
            f"- **Safety Posture:** {self.safety_implication}",
            f"- **Recommended Operator Action:** {self.recommended_next_step}",
            "",
            "---",
            f"**Abstention Hash:** `{self.provenance.abstention_hash}` | **Engine Version:** `{self.provenance.engine_version}`",
        ])
        return "\n".join(lines)


# =============================================================================
# Deterministic Hash & Builder
# =============================================================================

def compute_deterministic_abstention_hash(data_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonically serialized abstention dict."""
    clean_dict = {k: v for k, v in data_dict.items() if k != "abstention_hash"}
    canonical_bytes = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


import math

def is_valid_finite_number(val: Any) -> bool:
    """Validate that a value is a non-null, finite numerical scalar."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def build_abstention_explanation(
    is_abstained: bool,
    reconstruction_residual_t_core: Optional[float] = None,
    threshold_residual_t_core: float = 6.00,
    reconstruction_residual_8d: Optional[float] = None,
    threshold_residual_8d: float = 1.90,
    latent_novelty_d: Optional[float] = None,
    threshold_latent_novelty: float = 15.0,
    epistemic_uncertainty_sigma: Optional[float] = None,
    threshold_epistemic_uncertainty: Optional[float] = None,
    peak_t_core_predicted: Optional[float] = None,
    max_pressure_predicted: Optional[float] = None,
    min_flow_predicted: Optional[float] = None,
    explicit_abstention_reason: Optional[str] = None,
    source_evidence_hash: Optional[str] = None,
    scenario_id: Optional[str] = None,
    model_version: str = "baseline_005",
    engine_version: str = "v1.0_authoritative_abstention",
    timestamp_utc: Optional[str] = None,
) -> AbstentionExplanation:
    """Construct an independent, comprehensive AbstentionExplanation object."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

    # 1. Evaluate Evidence Metrics
    evidence_metrics: List[AbstentionEvidenceMetric] = []
    triggers: List[str] = []

    # Thermal reconstruction residual (R_T)
    if is_valid_finite_number(reconstruction_residual_t_core):
        rt_val = float(reconstruction_residual_t_core)  # type: ignore
        excess_rt = rt_val - threshold_residual_t_core
        is_trig_rt = (excess_rt > 0.0)
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Thermal Reconstruction Residual (R_T)",
                observed_value=rt_val,
                threshold=threshold_residual_t_core,
                excess=excess_rt,
                direction="GREATER_THAN_THRESHOLD" if is_trig_rt else "WITHIN_LIMITS",
                unit="°C",
                is_trigger=is_trig_rt,
            )
        )
        if is_trig_rt:
            triggers.append("MODEL_ABSTAIN_THERMAL")
    else:
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Thermal Reconstruction Residual (R_T)",
                observed_value=None,
                threshold=threshold_residual_t_core,
                excess=None,
                direction="UNAVAILABLE",
                unit="°C",
                is_trigger=False,
            )
        )

    # 8D reconstruction residual (R_8D)
    if is_valid_finite_number(reconstruction_residual_8d):
        r8d_val = float(reconstruction_residual_8d)  # type: ignore
        excess_r8d = r8d_val - threshold_residual_8d
        is_trig_r8d = (excess_r8d > 0.0)
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Multivariate 8D Reconstruction Residual (R_8D)",
                observed_value=r8d_val,
                threshold=threshold_residual_8d,
                excess=excess_r8d,
                direction="GREATER_THAN_THRESHOLD" if is_trig_r8d else "WITHIN_LIMITS",
                unit="z-norm",
                is_trigger=is_trig_r8d,
            )
        )
        if is_trig_r8d:
            triggers.append("MODEL_ABSTAIN_8D")
    else:
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Multivariate 8D Reconstruction Residual (R_8D)",
                observed_value=None,
                threshold=threshold_residual_8d,
                excess=None,
                direction="UNAVAILABLE",
                unit="z-norm",
                is_trigger=False,
            )
        )

    # Latent novelty / Mahalanobis distance (D_latent)
    if is_valid_finite_number(latent_novelty_d):
        nov_val = float(latent_novelty_d)  # type: ignore
        excess_nov = nov_val - threshold_latent_novelty
        is_trig_nov = (excess_nov > 0.0)
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Latent Manifold Novelty Distance (D_latent)",
                observed_value=nov_val,
                threshold=threshold_latent_novelty,
                excess=excess_nov,
                direction="GREATER_THAN_THRESHOLD" if is_trig_nov else "WITHIN_LIMITS",
                unit="d_M",
                is_trigger=is_trig_nov,
            )
        )
        if is_trig_nov:
            triggers.append("LATENT_NOVELTY")
    else:
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Latent Manifold Novelty Distance (D_latent)",
                observed_value=None,
                threshold=threshold_latent_novelty,
                excess=None,
                direction="UNAVAILABLE",
                unit="d_M",
                is_trigger=False,
            )
        )

    # Missing or non-finite prediction check (Fail-closed: NaN/Inf/None are strictly invalid)
    missing_fields: List[str] = []
    if not is_valid_finite_number(peak_t_core_predicted):
        missing_fields.append("T_core")
    if not is_valid_finite_number(max_pressure_predicted):
        missing_fields.append("P_sys")
    if not is_valid_finite_number(min_flow_predicted):
        missing_fields.append("F_cool")

    if missing_fields:
        triggers.append("MISSING_PREDICTION")

    # Epistemic uncertainty check (Only evaluated if supplied by an authoritative upstream trigger)
    if is_valid_finite_number(epistemic_uncertainty_sigma) and is_valid_finite_number(threshold_epistemic_uncertainty):
        sigma_val = float(epistemic_uncertainty_sigma)  # type: ignore
        thr_unc = float(threshold_epistemic_uncertainty)  # type: ignore
        excess_unc = sigma_val - thr_unc
        is_trig_unc = (excess_unc > 0.0)
        evidence_metrics.append(
            AbstentionEvidenceMetric(
                metric_name="Epistemic Uncertainty Dispersion (σ)",
                observed_value=sigma_val,
                threshold=thr_unc,
                excess=excess_unc,
                direction="GREATER_THAN_THRESHOLD" if is_trig_unc else "WITHIN_LIMITS",
                unit="σ",
                is_trigger=is_trig_unc,
            )
        )
        if is_trig_unc:
            triggers.append("EPISTEMIC_UNCERTAINTY")


    # 2. Determine Abstention Type and Rationale
    if not is_abstained and len(triggers) == 0:
        abstention_type = "NONE"
        headline = "System Fully Operational: Recommendation Emitted"
        primary_reason = "All telemetry matches representation space and candidate simulations passed trust bounds."
        affected_decision = "NO_ABSTENTION"
        blocked_actions = []
        safety_implication = "Predictive simulations are considered decision-grade."
        recommended_next_step = "Proceed with recommended autonomous intervention plan."
        actual_abstained = False

    else:
        actual_abstained = True
        # Canonical priority ordering: MODEL_ABSTAIN > LATENT_NOVELTY > MISSING_PREDICTION > EPISTEMIC_UNCERTAINTY
        if "MODEL_ABSTAIN_THERMAL" in triggers or "MODEL_ABSTAIN_8D" in triggers or (is_abstained and not triggers and explicit_abstention_reason):
            abstention_type = "MODEL_ABSTAIN"
            headline = "Autonomous Planning Aborted: Model Trust Boundary Exceeded"
            affected_decision = "RECOMMENDATION_BLOCKED"
            blocked_actions = ["candidate_intervention_selection", "counterfactual_recommendation_emission"]
            safety_implication = "World model forecasts are untrusted due to high reconstruction residual. Autonomous control is withheld to prevent ungrounded actuation."
            recommended_next_step = "Hold current actuator states, flag telemetry anomaly to operator, and perform human-in-the-loop diagnostic."

            # Construct detailed primary reason
            reasons = []
            if "MODEL_ABSTAIN_THERMAL" in triggers and reconstruction_residual_t_core is not None:
                reasons.append(f"Thermal reconstruction residual R_T = {reconstruction_residual_t_core:.2f}°C exceeds trust threshold {threshold_residual_t_core:.2f}°C by {reconstruction_residual_t_core - threshold_residual_t_core:.2f}°C")
            if "MODEL_ABSTAIN_8D" in triggers and reconstruction_residual_8d is not None:
                reasons.append(f"8D residual R_8D = {reconstruction_residual_8d:.2f} exceeds threshold {threshold_residual_8d:.2f} by {reconstruction_residual_8d - threshold_residual_8d:.2f}")
            if explicit_abstention_reason and not reasons:
                reasons.append(explicit_abstention_reason)
            primary_reason = f"Model Trust Gateway Triggered: {'; '.join(reasons)}."

        elif "LATENT_NOVELTY" in triggers:
            abstention_type = "LATENT_NOVELTY"
            headline = "Candidate Evaluation Blocked: Latent Manifold Novelty Exceeded"
            affected_decision = "CANDIDATE_SEARCH_ABORTED"
            blocked_actions = ["out_of_distribution_candidate_execution", "unsupported_latent_rollout"]
            safety_implication = "Candidate trajectory traverses an unsupported region of the latent state space (D_latent > 15.0). Dynamics extrapolation cannot be certified."
            recommended_next_step = "Restrict intervention search space to supported manifold envelope or require operator confirmation."
            primary_reason = f"Latent Novelty Boundary Breached: D_latent = {latent_novelty_d:.2f} exceeds support threshold {threshold_latent_novelty:.2f} by {(latent_novelty_d or 0.0) - threshold_latent_novelty:.2f}."

        elif "MISSING_PREDICTION" in triggers:
            abstention_type = "MISSING_PREDICTION"
            headline = "Safety Evaluation Unavailable: Required Predictive Telemetry Missing"
            affected_decision = "EVALUATION_WITHHELD"
            blocked_actions = ["candidate_recommendation", "safety_certification"]
            safety_implication = "Fail-closed safety gate blocked decision because required predictive state dimensions are unavailable. Missing predictions are never assumed safe."
            recommended_next_step = "Inspect simulator telemetry pipeline to restore complete state forecast dimensions."
            primary_reason = f"Prediction Incomplete: Missing required simulation channels [{', '.join(missing_fields)}]."

        elif "EPISTEMIC_UNCERTAINTY" in triggers:
            abstention_type = "EPISTEMIC_UNCERTAINTY"
            headline = "Planning Withheld: Model Epistemic Uncertainty Exceeded"
            affected_decision = "RECOMMENDATION_BLOCKED"
            blocked_actions = ["candidate_intervention_selection"]
            safety_implication = "Epistemic variance across ensemble or rollout heads is too wide for certified decision-making."
            recommended_next_step = "Request operator oversight or select conservative fail-safe actions."
            primary_reason = f"Epistemic uncertainty σ = {epistemic_uncertainty_sigma:.2f} exceeds decision threshold {threshold_epistemic_uncertainty:.2f}."

        else:
            abstention_type = "SAFETY_EVALUATION_UNAVAILABLE"
            headline = "Decision Blocked: Upstream Evaluation Withheld"
            affected_decision = "EVALUATION_WITHHELD"
            blocked_actions = ["all_autonomous_recommendations"]
            safety_implication = "Decision pipeline withheld recommendation."
            recommended_next_step = "Review system audit trail."
            primary_reason = explicit_abstention_reason or "Upstream gateway withheld candidate evaluations."

    # 3. Build Provenance & Deterministic Hash
    prelim_dict = {
        "abstained": actual_abstained,
        "abstention_type": abstention_type,
        "headline": headline,
        "primary_reason": primary_reason,
        "evidence": [asdict(m) for m in evidence_metrics],
        "affected_decision": affected_decision,
        "blocked_actions": blocked_actions,
        "safety_implication": safety_implication,
        "recommended_next_step": recommended_next_step,
        "source_evidence_hash": source_evidence_hash,
        "engine_version": engine_version,
        "model_version": model_version,
        "scenario_id": scenario_id,
        "timestamp_utc": ts,
    }
    abstention_hash = compute_deterministic_abstention_hash(prelim_dict)

    prov = AbstentionProvenance(
        source_evidence_hash=source_evidence_hash,
        engine_version=engine_version,
        model_version=model_version,
        scenario_id=scenario_id,
        timestamp_utc=ts,
        abstention_hash=abstention_hash,
    )

    return AbstentionExplanation(
        abstained=actual_abstained,
        abstention_type=abstention_type,
        headline=headline,
        primary_reason=primary_reason,
        evidence=evidence_metrics,
        affected_decision=affected_decision,
        blocked_actions=blocked_actions,
        safety_implication=safety_implication,
        recommended_next_step=recommended_next_step,
        provenance=prov,
    )


def build_abstention_explanation_from_decision_evidence(
    evidence: DecisionEvidence,
) -> AbstentionExplanation:
    """Build a detailed AbstentionExplanation directly from a DecisionEvidence object."""
    is_abstained = (
        evidence.decision.decision_status == "BLOCKED"
        or evidence.trust.trust_state == "MODEL_ABSTAIN"
    )
    explicit_reason = evidence.decision.abstention_reason or evidence.trust.trust_reason if is_abstained else None

    # Extract prediction completeness
    peak_t = None
    max_p = None
    min_f = None
    if evidence.predicted_outcome and not is_abstained:
        peak_t = evidence.predicted_outcome.peak_t_core
        max_p = evidence.predicted_outcome.max_pressure
        min_f = evidence.predicted_outcome.min_flow

    return build_abstention_explanation(
        is_abstained=is_abstained,
        reconstruction_residual_t_core=evidence.trust.reconstruction_residual_t_core if evidence.trust else None,
        threshold_residual_t_core=evidence.trust.tau_residual_t if evidence.trust else 6.00,
        reconstruction_residual_8d=evidence.trust.reconstruction_residual_8d if evidence.trust else None,
        threshold_residual_8d=1.90,
        latent_novelty_d=evidence.trust.latent_novelty_d if evidence.trust else None,
        threshold_latent_novelty=evidence.trust.tau_novelty if evidence.trust else 15.0,
        epistemic_uncertainty_sigma=evidence.trust.epistemic_uncertainty_sigma if evidence.trust else None,
        peak_t_core_predicted=peak_t,
        max_pressure_predicted=max_p,
        min_flow_predicted=min_f,
        explicit_abstention_reason=explicit_reason,
        source_evidence_hash=evidence.provenance.decision_hash,
        scenario_id=evidence.provenance.scenario_id,
        model_version=evidence.provenance.model_version,
        timestamp_utc=evidence.provenance.timestamp_utc,
    )
