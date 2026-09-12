"""PRISM Unified Decision Record Assembly (Task 6.6).

Assembles the complete suite of specialized PRISM evidence and explanation engines
into a single, coherent, machine-readable, and cryptographically auditable Decision Record.

Components Unified:
1. Decision (Recommendation, status, timestamp)
2. Trust Evidence (Reconstruction residuals, predictive uncertainty, latent manifold support)
3. Causal Reasoning (SCM DAG mechanism, causal chain, predicted effects, alternative rankings)
4. Counterfactual Evidence (Factual vs twin-world rollouts, causal deltas, twin-world integrity)
5. Safety Evidence (k=2 effective values, exact margins, limiting constraints, machine-readable violations)
6. Abstention Explanation (Diagnostic triggers, excess, blocked actions, human-in-the-loop next steps)
7. Decision Quality (Pareto multi-objective utility, margin over second-best, alternative ranking)
8. Provenance (Comprehensive cryptographic hash chain and version metadata)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional, Any

from prism.explanation.evidence import (
    DecisionSummary,
    TrustEvidence,
    CausalEvidence,
    PredictedOutcome,
    SafetyEvidence,
    DecisionQuality,
    ProvenanceEvidence,
    DecisionEvidence,
)
from prism.explanation.causal_explanation import (
    CausalExplanation,
    build_causal_explanation,
)
from prism.explanation.counterfactual_evidence import (
    CounterfactualEvidence,
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


# =============================================================================
# Unified Provenance & Decision Record Data Structures
# =============================================================================

@dataclass
class UnifiedProvenance:
    """Complete cryptographic audit chain linking all evidence sub-engines."""
    model_version: str
    dataset_version: str
    planner_version: str
    benchmark_version: str
    unified_engine_version: str
    scenario_id: Optional[str]
    timestamp_utc: str
    decision_evidence_hash: Optional[str]
    safety_hash: Optional[str]
    abstention_hash: Optional[str]
    counterfactual_hash: Optional[str]
    unified_record_hash: str


@dataclass
class PrismDecisionRecord:
    """Root Machine-Readable PRISM Unified Decision Record Contract."""
    decision: DecisionSummary
    trust: TrustEvidence
    causal_reasoning: CausalExplanation
    counterfactual: Optional[CounterfactualEvidence]
    safety: SafetyEvidenceDetail
    abstention: AbstentionExplanation
    decision_quality: DecisionQuality
    provenance: UnifiedProvenance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_markdown(self) -> str:
        """Render a unified, human-auditable executive decision dossier."""
        dominant_chain_str = " -> ".join(n.variable for n in self.causal_reasoning.causal_chain.nodes) if self.causal_reasoning.causal_chain.nodes else "None"
        net_dir = self.causal_reasoning.causal_chain.nodes[-1].predicted_direction if self.causal_reasoning.causal_chain.nodes else "STABLE"

        lines = [
            "# 🏛️ PRISM Unified Decision Record & Audit Dossier",
            f"**Scenario ID:** `{self.provenance.scenario_id or 'N/A'}` | **Decision Status:** `{self.decision.decision_status}`",
            f"**Recommendation:** `{self.decision.recommendation or 'NONE (ABSTAINED)'}` | **Safe:** `{self.safety.is_safe}`",
            f"**Trust State:** `{self.trust.trust_state}` | **Overall Safety State:** `{self.safety.overall_state}`",
            "",
            "---",
            "",
            "## 1. Executive Summary & Causal Reasoning",
            f"- **Headline:** {self.causal_reasoning.summary.headline}",
            f"- **Primary Mechanism:** {self.causal_reasoning.summary.mechanism}",
            f"- **Dominant Pathway:** `{dominant_chain_str}`",
            f"- **Net Direction of Effect:** `{net_dir}`",
            "",
            "## 2. Upfront Model Trust & Telemetry Integrity",
            f"- **Trust Classification:** `{self.trust.trust_state}`",
            f"- **Thermal Reconstruction Residual (R_T):** `{self.trust.reconstruction_residual_t_core:.2f} °C` (Threshold: `{self.trust.tau_residual_t:.2f} °C`)",
            f"- **Multivariate 8D Residual (R_8D):** `{self.trust.reconstruction_residual_8d:.2f} z-norm` (Threshold: `1.90 z-norm`)",
            f"- **Latent Manifold Novelty (D_latent):** `{self.trust.latent_novelty_d:.2f} d_M` (Threshold: `{self.trust.tau_novelty:.2f} d_M`)",
            f"- **Trust Assessment:** {self.trust.trust_reason}",
            "",
            "## 3. Physical Safety & Support Constraint Audit",
            f"- **Limiting Constraint:** `{self.safety.limiting_constraint.constraint_name.title()}` ({self.safety.limiting_constraint.variable_symbol})",
            f"- **Limiting Buffer / Headroom:** {f'{self.safety.limiting_constraint.raw_margin:+.2f} {self.safety.limiting_constraint.unit}' if self.safety.limiting_constraint.raw_margin is not None else 'N/A'}"
            f"{f' ({self.safety.limiting_constraint.normalized_headroom_pct:+.1f}% headroom)' if self.safety.limiting_constraint.normalized_headroom_pct is not None else ''}",
            f"- **Safety Assessment:** {self.safety.limiting_constraint.rationale}",
            "",
            "| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]


        for c_key in ["thermal", "pressure", "flow", "latent_support"]:
            if c_key in self.safety.constraints:
                c = self.safety.constraints[c_key]
                eff_s = f"{c.effective_value:.2f} {c.unit}" if c.effective_value is not None else "N/A"
                thr_s = f"{c.comparison_operator} {c.threshold:.2f} {c.unit}"
                m_s = f"{c.margin:+.2f} {c.unit}" if c.margin is not None else "N/A"
                st_s = "✅ PASS" if c.is_passed else "❌ FAIL"
                if c.is_marginal and c.is_passed:
                    st_s = "⚠️ MARGINAL"
                lines.append(f"| **{c.name.title()}** | `{c.variable_symbol}` | {eff_s} | {thr_s} | {m_s} | {st_s} |")

        if self.safety.violations:
            lines.extend([
                "",
                "### Detected Safety Violations",
            ])
            for v in self.safety.violations:
                lines.append(f"- **`{v.constraint_code}`** [{v.severity}]: {v.message}")

        if self.abstention.abstained:
            lines.extend([
                "",
                "## 4. Abstention & Trust Boundary Audit",
                f"- **Abstention Type:** `{self.abstention.abstention_type}`",
                f"- **Primary Reason:** {self.abstention.primary_reason}",
                f"- **Blocked Actions:** {', '.join(f'`{a}`' for a in self.abstention.blocked_actions)}",
                f"- **Safety Posture:** {self.abstention.safety_implication}",
                f"- **Recommended Operator Action:** {self.abstention.recommended_next_step}",
            ])

        if self.counterfactual is not None:
            lines.extend([
                "",
                "## 5. Counterfactual Twin-World Verification",
                f"- **Intervention Target:** `{self.counterfactual.intervention.intervention_target}` = `{self.counterfactual.intervention.counterfactual_value}` (Factual: `{self.counterfactual.intervention.original_value}`)",
                f"- **Peak T_core Delta:** `{self.counterfactual.causal_effect.delta_t_core_peak:+.2f} °C`" if self.counterfactual.causal_effect.delta_t_core_peak is not None else "- **Peak T_core Delta:** `N/A`",
                f"- **Max Pressure Delta:** `{self.counterfactual.causal_effect.delta_p_sys_max:+.2f} bar`" if self.counterfactual.causal_effect.delta_p_sys_max is not None else "- **Max Pressure Delta:** `N/A`",
                f"- **Min Flow Delta:** `{self.counterfactual.causal_effect.delta_f_cool_min:+.2f} L/min`" if self.counterfactual.causal_effect.delta_f_cool_min is not None else "- **Min Flow Delta:** `N/A`",
                f"- **Exogenous Conditions Shared:** `{self.counterfactual.twin_world_integrity.shared_exogenous_conditions}`",
                f"- **Pre-Intervention History Identical:** `{self.counterfactual.twin_world_integrity.identical_pre_intervention_history}`",
                f"- **Counterfactual Safety Transition:** `{self.counterfactual.safety_comparison.factual_safety_state} -> {self.counterfactual.safety_comparison.counterfactual_safety_state}`",
            ])


        lines.extend([
            "",
            "## 6. Decision Quality & Optimization Gap",
            f"- **Utility Score:** `{self.decision_quality.utility_score:.4f}`" if self.decision_quality.utility_score is not None else "- **Utility Score:** `N/A (Abstained)`",
            f"- **Second-Best Alternative:** `{self.decision_quality.second_best_candidate or 'None'}`",
            f"- **Decision Margin over Second-Best:** `{self.decision_quality.decision_margin:.4f}`" if self.decision_quality.decision_margin is not None else "- **Decision Margin:** `N/A`",
            "",
            "---",
            "## 7. Cryptographic Provenance Chain",
            f"- **Unified Record Hash (SHA-256):** `{self.provenance.unified_record_hash}`",
            f"- **Decision Evidence Hash:** `{self.provenance.decision_evidence_hash or 'N/A'}`",
            f"- **Safety Evidence Hash:** `{self.provenance.safety_hash or 'N/A'}`",
            f"- **Abstention Evidence Hash:** `{self.provenance.abstention_hash or 'N/A'}`",
            f"- **Counterfactual Hash:** `{self.provenance.counterfactual_hash or 'N/A'}`",
            f"- **Model / Planner:** `{self.provenance.model_version}` / `{self.provenance.planner_version}` | **Timestamp (UTC):** `{self.provenance.timestamp_utc}`",
        ])
        return "\n".join(lines)


# =============================================================================
# Deterministic Hash & Unified Assembly Builder
# =============================================================================

def compute_deterministic_unified_hash(data_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonically serialized unified record dict."""
    clean_dict = {k: v for k, v in data_dict.items() if k != "unified_record_hash"}
    canonical_bytes = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def build_unified_decision_record(
    decision_evidence: Optional[DecisionEvidence] = None,
    causal_explanation: Optional[CausalExplanation] = None,
    counterfactual_evidence: Optional[CounterfactualEvidence] = None,
    safety_evidence: Optional[SafetyEvidenceDetail] = None,
    abstention_explanation: Optional[AbstentionExplanation] = None,
    scenario_id: Optional[str] = None,
    model_version: str = "baseline_005",
    dataset_version: str = "v2.1",
    planner_version: str = "v1.2",
    benchmark_version: str = "v1.0",
    unified_engine_version: str = "v1.0_unified_assembly",
    timestamp_utc: Optional[str] = None,
) -> PrismDecisionRecord:
    """Assemble all specialized PRISM evidence sub-engines into a unified Decision Record."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

    # 1. Resolve or Build DecisionEvidence as base contract
    if decision_evidence is None:
        raise ValueError("decision_evidence is required as the root evidence projection.")

    # 2. Resolve Causal Explanation
    if causal_explanation is None:
        causal_explanation = build_causal_explanation(decision_evidence, timestamp_utc=ts)

    # 3. Resolve Safety Evidence
    if safety_evidence is None:
        safety_evidence = build_safety_evidence_from_decision_evidence(decision_evidence)

    # 4. Resolve Abstention Explanation
    if abstention_explanation is None:
        abstention_explanation = build_abstention_explanation_from_decision_evidence(decision_evidence)


    # 5. Extract Sub-Engine Hashes
    dec_hash = decision_evidence.provenance.decision_hash
    saf_hash = safety_evidence.provenance.safety_hash
    abs_hash = abstention_explanation.provenance.abstention_hash
    cf_hash = counterfactual_evidence.provenance.counterfactual_hash if counterfactual_evidence else None
    scen_id = scenario_id or decision_evidence.provenance.scenario_id

    # 6. Build Cryptographic Unified Hash
    prelim_dict = {
        "decision": decision_evidence.decision.to_dict() if hasattr(decision_evidence.decision, "to_dict") else asdict(decision_evidence.decision),
        "trust": asdict(decision_evidence.trust),
        "causal_reasoning": causal_explanation.to_dict(),
        "counterfactual": counterfactual_evidence.to_dict() if counterfactual_evidence else None,
        "safety": safety_evidence.to_dict(),
        "abstention": abstention_explanation.to_dict(),
        "decision_quality": asdict(decision_evidence.decision_quality),
        "model_version": model_version,
        "dataset_version": dataset_version,
        "planner_version": planner_version,
        "benchmark_version": benchmark_version,
        "unified_engine_version": unified_engine_version,
        "scenario_id": scen_id,
        "decision_evidence_hash": dec_hash,
        "safety_hash": saf_hash,
        "abstention_hash": abs_hash,
        "counterfactual_hash": cf_hash,
        "timestamp_utc": ts,
    }
    unified_hash = compute_deterministic_unified_hash(prelim_dict)

    prov = UnifiedProvenance(
        model_version=model_version,
        dataset_version=dataset_version,
        planner_version=planner_version,
        benchmark_version=benchmark_version,
        unified_engine_version=unified_engine_version,
        scenario_id=scen_id,
        timestamp_utc=ts,
        decision_evidence_hash=dec_hash,
        safety_hash=saf_hash,
        abstention_hash=abs_hash,
        counterfactual_hash=cf_hash,
        unified_record_hash=unified_hash,
    )

    return PrismDecisionRecord(
        decision=decision_evidence.decision,
        trust=decision_evidence.trust,
        causal_reasoning=causal_explanation,
        counterfactual=counterfactual_evidence,
        safety=safety_evidence,
        abstention=abstention_explanation,
        decision_quality=decision_evidence.decision_quality,
        provenance=prov,
    )
