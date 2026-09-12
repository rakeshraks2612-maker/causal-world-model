"""PRISM Safety Evidence Engine (Task 6.4).

Provides independent, deterministic, and machine-readable audit trails for all
hard physical and epistemic safety constraints (T_core, P_sys, F_cool, D_latent).
Distinguishes raw predictions from k=2 uncertainty-adjusted effective values,
calculates exact physical margins, identifies limiting constraints, and tracks
cryptographic safety provenance.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


from prism.planning.safety_constraints import (
    SafetySeverity,
    SafetyViolationReason,
    DecisionSafetyConfig,
    SafetyResult,
    SafetyConstraintEngine,
)
from prism.explanation.evidence import DecisionEvidence


# =============================================================================
# Machine-Readable Safety Evidence Data Structures
# =============================================================================

@dataclass
class ConstraintDetail:
    """Detailed evaluation of a single physical or epistemic safety constraint."""
    name: str
    variable_symbol: str
    unit: str
    raw_prediction: Optional[float]
    uncertainty_sigma: Optional[float]
    k_multiplier: float
    effective_value: Optional[float]
    threshold: float
    comparison_operator: str  # "<", ">", "<="
    margin: Optional[float]   # Positive = safe buffer, Negative = violation
    normalized_headroom_pct: Optional[float]
    is_passed: bool
    is_marginal: bool
    violation_message: Optional[str] = None


@dataclass
class SafetyViolationDetail:
    """Structured record of a specific hard safety boundary breach."""
    constraint_code: str
    variable_symbol: str
    observed_effective_value: float
    threshold: float
    margin_violated_by: float  # Absolute overshoot
    severity: str
    message: str


@dataclass
class LimitingConstraintSummary:
    """Identification of the bottleneck / closest boundary constraint."""
    constraint_name: str
    variable_symbol: str
    raw_margin: Optional[float]
    unit: str
    normalized_headroom_pct: Optional[float]
    rationale: str
    normalized_violation_score: Optional[float] = None


@dataclass
class SafetyProvenance:
    """Traceable, immutable metadata linking safety evaluations to evidence hashes."""
    source_evidence_hash: Optional[str]
    safety_engine_version: str
    model_version: str
    scenario_id: Optional[str]
    candidate_id: Optional[str]
    timestamp_utc: str
    safety_hash: str


@dataclass
class SafetyEvidenceDetail:
    """Root Machine-Readable PRISM Safety Evidence Contract."""
    overall_state: str  # "SAFE", "MARGINAL", "UNSAFE", "ABSTAIN_REQUIRED"
    is_safe: bool
    constraints: Dict[str, ConstraintDetail]
    limiting_constraint: LimitingConstraintSummary
    violations: List[SafetyViolationDetail]
    uncertainty_adjusted: bool
    provenance: SafetyProvenance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_markdown(self) -> str:
        """Render a clean, human-auditable markdown report."""
        lines = [
            f"# 🛡️ PRISM Safety Audit & Constraint Evidence",
            f"**Overall Safety Posture:** `{self.overall_state}` | **Candidate Safe:** `{self.is_safe}`",
            f"**Uncertainty Adjustment:** `{'Active (k=2.0)' if self.uncertainty_adjusted else 'None (Raw Point Estimate)'}`",
            "",
            "## 1. Physical Safety Constraints",
            "",
            "| Constraint | Symbol | Raw (μ) | Uncertainty (σ) | Effective Value | Threshold | Margin | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        # Physical constraints
        for c_key in ["thermal", "pressure", "flow"]:
            if c_key in self.constraints:
                c = self.constraints[c_key]
                raw_str = f"{c.raw_prediction:.2f} {c.unit}" if c.raw_prediction is not None else "N/A"
                sig_str = f"{c.uncertainty_sigma:.2f} {c.unit}" if c.uncertainty_sigma is not None else "N/A"
                eff_str = f"{c.effective_value:.2f} {c.unit}" if c.effective_value is not None else "N/A"
                thresh_str = f"{c.comparison_operator} {c.threshold:.2f} {c.unit}"
                m_str = f"{c.margin:+.2f} {c.unit}" if c.margin is not None else "N/A"
                status_str = "✅ PASS" if c.is_passed else "❌ FAIL"
                if c.is_marginal and c.is_passed:
                    status_str = "⚠️ MARGINAL"
                lines.append(f"| **{c.name.title()}** | `{c.variable_symbol}` | {raw_str} | {sig_str} | {eff_str} | {thresh_str} | {m_str} | {status_str} |")

        lines.extend([
            "",
            "## 2. Model-Support Constraint",
            "",
            "| Constraint | Symbol | Observed Novelty | Threshold | Margin | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        if "latent_support" in self.constraints:
            c = self.constraints["latent_support"]
            eff_str = f"{c.effective_value:.2f} {c.unit}" if c.effective_value is not None else "N/A"
            thresh_str = f"{c.comparison_operator} {c.threshold:.2f} {c.unit}"
            m_str = f"{c.margin:+.2f} {c.unit}" if c.margin is not None else "N/A"
            status_str = "✅ PASS" if c.is_passed else "❌ FAIL"
            if c.is_marginal and c.is_passed:
                status_str = "⚠️ MARGINAL"
            lines.append(f"| **Latent Novelty Support** | `{c.variable_symbol}` | {eff_str} | {thresh_str} | {m_str} | {status_str} |")

        lines.extend([
            "",
            "## 3. Limiting Constraint Analysis",
            f"- **Limiting Boundary:** `{self.limiting_constraint.constraint_name.title()}` ({self.limiting_constraint.variable_symbol})",
            f"- **Buffer / Headroom:** {f'{self.limiting_constraint.raw_margin:+.2f} {self.limiting_constraint.unit}' if self.limiting_constraint.raw_margin is not None else 'N/A'}"
            f"{f' ({self.limiting_constraint.normalized_headroom_pct:+.1f}% headroom)' if self.limiting_constraint.normalized_headroom_pct is not None else ''}",
            f"- **Assessment:** {self.limiting_constraint.rationale}",
        ])

        if self.violations:
            lines.extend([
                "",
                "## 4. Detected Boundary Violations",
            ])
            for v in self.violations:
                lines.append(f"- **`{v.constraint_code}`** [{v.severity}]: {v.message}")

        lines.extend([
            "",
            "---",
            f"**Safety Hash:** `{self.provenance.safety_hash}` | **Engine Version:** `{self.provenance.safety_engine_version}`",
        ])
        return "\n".join(lines)


# =============================================================================
# Deterministic Hash & Builder
# =============================================================================

def compute_deterministic_safety_hash(data_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonically serialized safety dict."""
    clean_dict = {k: v for k, v in data_dict.items() if k != "safety_hash"}
    canonical_bytes = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def evaluate_safety_evidence_detail(
    peak_t_core: Optional[float],
    max_pressure: Optional[float],
    min_flow: Optional[float],
    latent_novelty: Optional[float] = 0.0,
    sigma_t_core: float = 0.0,
    sigma_pressure: float = 0.0,
    sigma_flow: float = 0.0,
    use_uncertainty_bounds: bool = True,
    k_multiplier: float = 2.0,
    is_model_abstained: bool = False,
    abstention_reason: Optional[str] = None,
    source_evidence_hash: Optional[str] = None,
    scenario_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    model_version: str = "baseline_005",
    safety_engine_version: str = "v1.2_authoritative_gate",
    timestamp_utc: Optional[str] = None,
) -> SafetyEvidenceDetail:
    """Construct an independent, comprehensive SafetyEvidenceDetail object."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
    cfg = DecisionSafetyConfig(uncertainty_multiplier_k=k_multiplier)

    # 1. Handle Abstention State (No candidate evaluated or non-finite predictions)
    is_non_finite = not all(
        isinstance(v, (int, float)) and math.isfinite(v)
        for v in [peak_t_core, max_pressure, min_flow]
        if v is not None
    )
    if is_model_abstained or peak_t_core is None or max_pressure is None or min_flow is None or is_non_finite:

        constraints = {
            "thermal": ConstraintDetail(
                name="thermal",
                variable_symbol="T_core",
                unit="°C",
                raw_prediction=peak_t_core,
                uncertainty_sigma=sigma_t_core if sigma_t_core > 0 else None,
                k_multiplier=k_multiplier if use_uncertainty_bounds else 0.0,
                effective_value=None,
                threshold=cfg.thermal_hard_limit,
                comparison_operator="<",
                margin=None,
                normalized_headroom_pct=None,
                is_passed=False,
                is_marginal=False,
                violation_message="Evaluation withheld: Model trust gateway triggered MODEL_ABSTAIN",
            ),
            "pressure": ConstraintDetail(
                name="pressure",
                variable_symbol="P_sys",
                unit="bar",
                raw_prediction=max_pressure,
                uncertainty_sigma=sigma_pressure if sigma_pressure > 0 else None,
                k_multiplier=k_multiplier if use_uncertainty_bounds else 0.0,
                effective_value=None,
                threshold=cfg.pressure_hard_limit,
                comparison_operator="<",
                margin=None,
                normalized_headroom_pct=None,
                is_passed=False,
                is_marginal=False,
                violation_message="Evaluation withheld: Model trust gateway triggered MODEL_ABSTAIN",
            ),
            "flow": ConstraintDetail(
                name="flow",
                variable_symbol="F_cool",
                unit="L/min",
                raw_prediction=min_flow,
                uncertainty_sigma=sigma_flow if sigma_flow > 0 else None,
                k_multiplier=k_multiplier if use_uncertainty_bounds else 0.0,
                effective_value=None,
                threshold=cfg.flow_hard_limit,
                comparison_operator=">",
                margin=None,
                normalized_headroom_pct=None,
                is_passed=False,
                is_marginal=False,
                violation_message="Evaluation withheld: Model trust gateway triggered MODEL_ABSTAIN",
            ),
            "latent_support": ConstraintDetail(
                name="latent_support",
                variable_symbol="D_latent",
                unit="d_M",
                raw_prediction=latent_novelty,
                uncertainty_sigma=None,
                k_multiplier=0.0,
                effective_value=latent_novelty,
                threshold=cfg.latent_novelty_hard_limit,
                comparison_operator="<=",
                margin=float(cfg.latent_novelty_hard_limit - (latent_novelty or 0.0)) if latent_novelty is not None else None,
                normalized_headroom_pct=float((cfg.latent_novelty_hard_limit - (latent_novelty or 0.0)) / cfg.latent_novelty_hard_limit * 100.0) if latent_novelty is not None else None,
                is_passed=bool(latent_novelty <= cfg.latent_novelty_hard_limit) if latent_novelty is not None else False,
                is_marginal=bool(latent_novelty > cfg.latent_marginal_threshold) if latent_novelty is not None else False,
                violation_message=None,
            ),
        }

        limiting = LimitingConstraintSummary(
            constraint_name="model_trust_gateway",
            variable_symbol="R_T",
            raw_margin=None,
            unit="N/A",
            normalized_headroom_pct=None,
            normalized_violation_score=None,
            rationale=f"Planning aborted due to model abstention: {abstention_reason or 'Observation inconsistency detected'}.",
        )

        violations = [
            SafetyViolationDetail(
                constraint_code="ABSTAIN_REQUIRED",
                variable_symbol="R_T",
                observed_effective_value=0.0,
                threshold=0.0,
                margin_violated_by=0.0,
                severity="ABSTAIN_REQUIRED",
                message=f"Model Trust Gateway Aborted: {abstention_reason or 'Telemetry violates representation consistency.'}",
            )
        ]

        overall_state = "ABSTAIN_REQUIRED"
        is_safe = False

    else:
        # 2. Compute Uncertainty-Adjusted Effective Values
        k = k_multiplier if use_uncertainty_bounds else 0.0
        eff_t_core = float(peak_t_core + k * max(0.0, sigma_t_core))
        eff_pressure = float(max_pressure + k * max(0.0, sigma_pressure))
        eff_flow = float(min_flow - k * max(0.0, sigma_flow))
        eff_novelty = float(latent_novelty if latent_novelty is not None else 0.0)

        # Exact Physical & Epistemic Margins
        # Positive = safe buffer, Zero = exact boundary, Negative = violation
        margin_t = float(cfg.thermal_hard_limit - eff_t_core)
        margin_p = float(cfg.pressure_hard_limit - eff_pressure)
        margin_f = float(eff_flow - cfg.flow_hard_limit)
        margin_nov = float(cfg.latent_novelty_hard_limit - eff_novelty)

        # Declared Nominal Operating Spans
        SPAN_T = 75.0   # 20.0 to 95.0 °C
        SPAN_P = 5.0    # 0.5 to 5.5 bar
        SPAN_F = 52.0   # 8.0 to 60.0 L/min
        SPAN_D = 15.0   # 0.0 to 15.0 d_M

        # Normalized Headroom (% of nominal operating span)
        norm_t = float(margin_t / SPAN_T * 100.0)
        norm_p = float(margin_p / SPAN_P * 100.0)
        norm_f = float(margin_f / SPAN_F * 100.0)
        norm_nov = float(margin_nov / SPAN_D * 100.0)

        # Boundary Pass/Fail Rules (Strict physical inequality vs non-strict latent support)
        pass_t = (eff_t_core < cfg.thermal_hard_limit)
        pass_p = (eff_pressure < cfg.pressure_hard_limit)
        pass_f = (eff_flow > cfg.flow_hard_limit)
        pass_nov = (eff_novelty <= cfg.latent_novelty_hard_limit)

        marg_t = (pass_t and eff_t_core >= cfg.thermal_marginal_threshold)
        marg_p = (pass_p and eff_pressure >= cfg.pressure_marginal_threshold)
        marg_f = (pass_f and eff_flow <= cfg.flow_marginal_threshold)
        marg_nov = (pass_nov and eff_novelty >= cfg.latent_marginal_threshold)

        # Violation messages
        msg_t = None if pass_t else f"T_core effective ({eff_t_core:.2f}°C) exceeds thermal limit ({cfg.thermal_hard_limit:.1f}°C) by {abs(margin_t):.2f}°C."
        msg_p = None if pass_p else f"P_sys effective ({eff_pressure:.2f} bar) exceeds pressure limit ({cfg.pressure_hard_limit:.2f} bar) by {abs(margin_p):.2f} bar."
        msg_f = None if pass_f else f"F_cool effective ({eff_flow:.2f} L/min) violates flow minimum ({cfg.flow_hard_limit:.1f} L/min) by {abs(margin_f):.2f} L/min."
        msg_nov = None if pass_nov else f"Latent novelty ({eff_novelty:.2f}) exceeds support limit ({cfg.latent_novelty_hard_limit:.1f}) by {abs(margin_nov):.2f}."

        constraints = {
            "thermal": ConstraintDetail(
                name="thermal",
                variable_symbol="T_core",
                unit="°C",
                raw_prediction=float(peak_t_core),
                uncertainty_sigma=float(sigma_t_core) if sigma_t_core > 0 else None,
                k_multiplier=k,
                effective_value=eff_t_core,
                threshold=cfg.thermal_hard_limit,
                comparison_operator="<",
                margin=margin_t,
                normalized_headroom_pct=norm_t,
                is_passed=pass_t,
                is_marginal=marg_t,
                violation_message=msg_t,
            ),
            "pressure": ConstraintDetail(
                name="pressure",
                variable_symbol="P_sys",
                unit="bar",
                raw_prediction=float(max_pressure),
                uncertainty_sigma=float(sigma_pressure) if sigma_pressure > 0 else None,
                k_multiplier=k,
                effective_value=eff_pressure,
                threshold=cfg.pressure_hard_limit,
                comparison_operator="<",
                margin=margin_p,
                normalized_headroom_pct=norm_p,
                is_passed=pass_p,
                is_marginal=marg_p,
                violation_message=msg_p,
            ),
            "flow": ConstraintDetail(
                name="flow",
                variable_symbol="F_cool",
                unit="L/min",
                raw_prediction=float(min_flow),
                uncertainty_sigma=float(sigma_flow) if sigma_flow > 0 else None,
                k_multiplier=k,
                effective_value=eff_flow,
                threshold=cfg.flow_hard_limit,
                comparison_operator=">",
                margin=margin_f,
                normalized_headroom_pct=norm_f,
                is_passed=pass_f,
                is_marginal=marg_f,
                violation_message=msg_f,
            ),
            "latent_support": ConstraintDetail(
                name="latent_support",
                variable_symbol="D_latent",
                unit="d_M",
                raw_prediction=float(latent_novelty if latent_novelty is not None else 0.0),
                uncertainty_sigma=None,
                k_multiplier=0.0,
                effective_value=eff_novelty,
                threshold=cfg.latent_novelty_hard_limit,
                comparison_operator="<=",
                margin=margin_nov,
                normalized_headroom_pct=norm_nov,
                is_passed=pass_nov,
                is_marginal=marg_nov,
                violation_message=msg_nov,
            ),
        }

        violations: List[SafetyViolationDetail] = []
        if not pass_t:
            violations.append(
                SafetyViolationDetail(
                    constraint_code=SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value,
                    variable_symbol="T_core",
                    observed_effective_value=eff_t_core,
                    threshold=cfg.thermal_hard_limit,
                    margin_violated_by=abs(margin_t),
                    severity="UNSAFE",
                    message=msg_t or "Thermal limit exceeded",
                )
            )
        if not pass_p:
            violations.append(
                SafetyViolationDetail(
                    constraint_code=SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value,
                    variable_symbol="P_sys",
                    observed_effective_value=eff_pressure,
                    threshold=cfg.pressure_hard_limit,
                    margin_violated_by=abs(margin_p),
                    severity="UNSAFE",
                    message=msg_p or "Pressure limit exceeded",
                )
            )
        if not pass_f:
            violations.append(
                SafetyViolationDetail(
                    constraint_code=SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value,
                    variable_symbol="F_cool",
                    observed_effective_value=eff_flow,
                    threshold=cfg.flow_hard_limit,
                    margin_violated_by=abs(margin_f),
                    severity="UNSAFE",
                    message=msg_f or "Flow minimum violated",
                )
            )
        if not pass_nov:
            violations.append(
                SafetyViolationDetail(
                    constraint_code=SafetyViolationReason.LATENT_NOVELTY_EXCEEDED.value,
                    variable_symbol="D_latent",
                    observed_effective_value=eff_novelty,
                    threshold=cfg.latent_novelty_hard_limit,
                    margin_violated_by=abs(margin_nov),
                    severity="ABSTAIN_REQUIRED",
                    message=msg_nov or "Latent novelty support limit exceeded",
                )
            )

        is_safe = (len(violations) == 0)

        # Determine Overall State
        if not is_safe:
            has_abstain_sev = any(v.severity == "ABSTAIN_REQUIRED" for v in violations)
            overall_state = "ABSTAIN_REQUIRED" if has_abstain_sev else "UNSAFE"
        elif any(c.is_marginal for c in constraints.values()):
            overall_state = "MARGINAL"
        else:
            overall_state = "SAFE"

        # 3. Identify Limiting Constraint
        # Canonical tie-breaking order: thermal -> pressure -> flow -> latent_support
        canonical_order = {"thermal": 0, "pressure": 1, "flow": 2, "latent_support": 3}

        if not is_safe:
            # Pick the breached constraint with the highest dimensionless normalized violation score V_c = (-margin) / span
            # Mathematically identical to the minimum normalized_headroom_pct among breached constraints
            breached_constraints = [c for c in constraints.values() if not c.is_passed]
            worst_c = min(
                breached_constraints,
                key=lambda c: (
                    c.normalized_headroom_pct if c.normalized_headroom_pct is not None else 0.0,
                    canonical_order.get(c.name, 99)
                )
            )
            v_score = (-worst_c.normalized_headroom_pct / 100.0) if worst_c.normalized_headroom_pct is not None else 0.0
            limiting = LimitingConstraintSummary(
                constraint_name=worst_c.name,
                variable_symbol=worst_c.variable_symbol,
                raw_margin=worst_c.margin,
                unit=worst_c.unit,
                normalized_headroom_pct=worst_c.normalized_headroom_pct,
                normalized_violation_score=v_score,
                rationale=f"Hard safety boundary breached: {worst_c.name.title()} exceeds limit with dimensionless normalized violation score {v_score:.3f} ({abs(worst_c.normalized_headroom_pct or 0.0):.1f}% over span, raw breach: {abs(worst_c.margin or 0.0):.2f} {worst_c.unit}).",
            )
        else:
            # Pick the compliant constraint with the lowest normalized headroom %
            closest_c = min(
                constraints.values(),
                key=lambda c: (
                    c.normalized_headroom_pct if c.normalized_headroom_pct is not None else 100.0,
                    canonical_order.get(c.name, 99)
                )
            )
            limiting = LimitingConstraintSummary(
                constraint_name=closest_c.name,
                variable_symbol=closest_c.variable_symbol,
                raw_margin=closest_c.margin,
                unit=closest_c.unit,
                normalized_headroom_pct=closest_c.normalized_headroom_pct,
                normalized_violation_score=0.0,
                rationale=f"{closest_c.name.title()} has the tightest operating buffer ({closest_c.margin:+.2f} {closest_c.unit}, {closest_c.normalized_headroom_pct:.1f}% headroom).",
            )


    # Provenance
    prelim_dict = {
        "overall_state": overall_state,
        "is_safe": is_safe,
        "constraints": {k: asdict(v) for k, v in constraints.items()},
        "limiting_constraint": asdict(limiting),
        "violations": [asdict(v) for v in violations],
        "uncertainty_adjusted": use_uncertainty_bounds,
        "source_evidence_hash": source_evidence_hash,
        "safety_engine_version": safety_engine_version,
        "model_version": model_version,
        "scenario_id": scenario_id,
        "candidate_id": candidate_id,
        "timestamp_utc": ts,
    }
    safety_hash = compute_deterministic_safety_hash(prelim_dict)

    prov = SafetyProvenance(
        source_evidence_hash=source_evidence_hash,
        safety_engine_version=safety_engine_version,
        model_version=model_version,
        scenario_id=scenario_id,
        candidate_id=candidate_id,
        timestamp_utc=ts,
        safety_hash=safety_hash,
    )

    return SafetyEvidenceDetail(
        overall_state=overall_state,
        is_safe=is_safe,
        constraints=constraints,
        limiting_constraint=limiting,
        violations=violations,
        uncertainty_adjusted=use_uncertainty_bounds,
        provenance=prov,
    )


def build_safety_evidence_from_decision_evidence(
    evidence: DecisionEvidence,
    k_multiplier: float = 2.0,
    use_uncertainty_bounds: bool = True,
) -> SafetyEvidenceDetail:
    """Build a detailed SafetyEvidenceDetail directly from a DecisionEvidence object."""
    is_abstained = (
        evidence.decision.decision_status == "BLOCKED"
        or evidence.trust.trust_state == "MODEL_ABSTAIN"
    )
    abstention_reason = evidence.decision.abstention_reason or evidence.trust.trust_reason if is_abstained else None

    # Extract prediction details if available
    peak_t = None
    max_p = None
    min_f = None
    sig_t = 0.0
    sig_p = 0.0
    sig_f = 0.0
    latent_nov = evidence.trust.latent_novelty_d if evidence.trust else 0.0

    if evidence.predicted_outcome:
        peak_t = evidence.predicted_outcome.peak_t_core
        max_p = evidence.predicted_outcome.max_pressure
        min_f = evidence.predicted_outcome.min_flow
        # If epistemic uncertainty is provided at trust level
        if evidence.trust and evidence.trust.epistemic_uncertainty_sigma > 0:
            sig_t = evidence.trust.epistemic_uncertainty_sigma

    return evaluate_safety_evidence_detail(
        peak_t_core=peak_t,
        max_pressure=max_p,
        min_flow=min_f,
        latent_novelty=latent_nov,
        sigma_t_core=sig_t,
        sigma_pressure=sig_p,
        sigma_flow=sig_f,
        use_uncertainty_bounds=use_uncertainty_bounds,
        k_multiplier=k_multiplier,
        is_model_abstained=is_abstained,
        abstention_reason=abstention_reason,
        source_evidence_hash=evidence.provenance.decision_hash,
        scenario_id=evidence.provenance.scenario_id,
        candidate_id=evidence.decision.recommendation,
        model_version=evidence.provenance.model_version,
        timestamp_utc=evidence.provenance.timestamp_utc,
    )



