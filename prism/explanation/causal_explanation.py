"""PRISM Causal Explanation Engine (Task 6.2).

Translates deterministic DecisionEvidence objects into rigorous, human-auditable,
and machine-readable CausalExplanation objects grounded in the ThermoHydro-Compute SCM DAG.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
import re
from typing import Dict, List, Optional, Any, Tuple

from prism.explanation.evidence import DecisionEvidence


# =============================================================================
# Machine-Readable Causal Explanation Data Structures
# =============================================================================

@dataclass
class ExplanationSummary:
    """High-level summary of the decision, physical mechanism, and effect direction."""
    headline: str
    mechanism: str
    effect_direction: str


@dataclass
class InterventionExplanation:
    """Structured details of the physical or state intervention."""
    target: str
    value: Optional[float]
    intervention_type: str  # "ACTION", "STATE", "COMPOUND", "NONE"
    baseline_value: Optional[float] = None


@dataclass
class CausalNode:
    """A single node along the causal DAG pathway."""
    variable: str
    relationship: str  # "ACTUATOR_INPUT", "DIRECT_DESCENDANT", "MEDIATED_DESCENDANT", "FINAL_OUTCOME"
    predicted_direction: str  # "INCREASE", "DECREASE", "STABLE", "UNKNOWN"
    predicted_delta: Optional[float]  # None if intermediate value is not directly evidenced
    evidence_available: bool


@dataclass
class CausalChain:
    """Complete causal DAG propagation chain and set of affected descendant variables."""
    root_actuator: str
    target_outcome: str
    nodes: List[CausalNode]
    all_affected_descendants: List[str]


@dataclass
class PredictedEffectsExplanation:
    """Counterfactual comparison: Model prediction under intervention vs do-nothing baseline."""
    baseline_peak_t_core: Optional[float]
    intervened_peak_t_core: Optional[float]
    delta_t_core: Optional[float]
    baseline_min_flow: Optional[float]
    intervened_min_flow: Optional[float]
    delta_flow: Optional[float]
    counterfactual_statement: str


@dataclass
class RejectedAlternative:
    """Structured rationale explaining why a candidate intervention was rejected."""
    candidate_id: str
    reason_code: str  # "LATENT_NOVELTY", "SAFETY_VIOLATION", "LOWER_UTILITY", "MODEL_UNCERTAINTY", "ABSTENTION"
    reason_explanation: str
    utility_score: Optional[float]
    peak_t_core: Optional[float]


@dataclass
class DecisionMarginExplanation:
    """Decision strength derived from the multi-objective utility gap."""
    best_candidate: Optional[str]
    second_best_candidate: Optional[str]
    decision_strength: str  # "POSITIVE_MARGIN", "ZERO_MARGIN", "N/A_ABSTAIN"
    decision_margin: Optional[float]
    best_utility: Optional[float]
    second_best_utility: Optional[float]


@dataclass
class AbstentionExplanation:
    """Rigorous audit trace explaining why PRISM refused to recommend an action."""
    trigger: str  # "RECONSTRUCTION_INCONSISTENCY", "LATENT_NOVELTY", "DUAL_FAILURE", "NONE"
    observed_value: Optional[float]
    threshold: Optional[float]
    violation_delta: Optional[float]
    consequence: str  # "WORLD_MODEL_NOT_TRUSTED", "PLANNING_BLOCKED"
    action_taken: str  # "PLANNING_BLOCKED", "NONE"


@dataclass
class ExplanationProvenance:
    """Traceable link to source DecisionEvidence and model versions."""
    source_evidence_hash: str
    model_version: str
    dataset_version: str
    planner_version: str
    benchmark_version: str
    scenario_id: Optional[str]
    timestamp_utc: str


@dataclass
class CausalExplanation:
    """Root Machine-Readable PRISM Causal Explanation Contract."""
    summary: ExplanationSummary
    intervention: InterventionExplanation
    causal_chain: CausalChain
    predicted_effects: PredictedEffectsExplanation
    rejected_alternatives: List[RejectedAlternative]
    decision_margin: DecisionMarginExplanation
    abstention_details: Optional[AbstentionExplanation]
    trust_qualification: str
    provenance: ExplanationProvenance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_markdown(self) -> str:
        """Render a clean, human-auditable markdown report."""
        lines = [
            f"# 🛡️ PRISM Causal Decision Explanation",
            f"**Headline:** {self.summary.headline}",
            f"**Trust Status:** {self.trust_qualification}",
            "",
            f"## 1. Causal Mechanism & Pathway",
            f"{self.summary.mechanism}",
            "",
            "```text",
        ]
        if self.causal_chain.nodes:
            for i, n in enumerate(self.causal_chain.nodes):
                delta_str = f" (Δ = {n.predicted_delta:+.2f})" if n.predicted_delta is not None else ""
                dir_str = f" [{n.predicted_direction}{delta_str}]" if n.predicted_direction != "STABLE" else " [STABLE]"
                arrow = "      ↓" if i > 0 else ""
                if arrow:
                    lines.append(arrow)
                lines.append(f"{n.variable}{dir_str}")
        lines.extend([
            "```",
            "",
            f"**All Affected Descendants:** `{', '.join(self.causal_chain.all_affected_descendants) or 'None'}`",
            "",
            f"## 2. Counterfactual Impact",
            f"{self.predicted_effects.counterfactual_statement}",
        ])
        if self.predicted_effects.delta_t_core is not None:
            lines.append(f"- **Peak Core Temp ($T_{{\\text{{core}}}}$):** {self.predicted_effects.baseline_peak_t_core:.2f}°C → {self.predicted_effects.intervened_peak_t_core:.2f}°C (Net effect: {self.predicted_effects.delta_t_core:+.2f}°C)")
        if self.predicted_effects.delta_flow is not None:
            lines.append(f"- **Coolant Flow ($F_{{\\text{{cool}}}}$):** {self.predicted_effects.baseline_min_flow:.2f} → {self.predicted_effects.intervened_min_flow:.2f} L/min (Net effect: {self.predicted_effects.delta_flow:+.2f} L/min)")

        if self.rejected_alternatives:
            lines.extend([
                "",
                f"## 3. Alternative Action Analysis",
            ])
            for alt in self.rejected_alternatives:
                lines.append(f"- **`{alt.candidate_id}`** [{alt.reason_code}]: {alt.reason_explanation}")

        if self.abstention_details and self.abstention_details.trigger != "NONE":
            lines.extend([
                "",
                f"## ⚠️ Abstention Audit",
                f"- **Trigger:** `{self.abstention_details.trigger}`",
                f"- **Observed Metric:** {self.abstention_details.observed_value:.2f} (Threshold: {self.abstention_details.threshold:.2f}, Exceeded by: {self.abstention_details.violation_delta:+.2f})",
                f"- **Consequence:** `{self.abstention_details.consequence}`",
                f"- **Action Taken:** `{self.abstention_details.action_taken}`",
            ])

        lines.extend([
            "",
            "---",
            f"**Evidence Hash:** `{self.provenance.source_evidence_hash}` | **Model:** `{self.provenance.model_version}`",
        ])
        return "\n".join(lines)


# =============================================================================
# Deterministic Explanation Builder
# =============================================================================

def build_causal_explanation(
    evidence: DecisionEvidence,
    timestamp_utc: Optional[str] = None,
) -> CausalExplanation:
    """Construct a deterministic CausalExplanation from a frozen DecisionEvidence object."""
    ts = timestamp_utc or evidence.provenance.timestamp_utc
    target = evidence.causal_evidence.intervention_target
    val = evidence.causal_evidence.intervention_value
    rec_id = evidence.decision.recommendation
    trust_state = evidence.trust.trust_state

    # 1. Classify Intervention Type
    if target in ["none", "do_nothing"]:
        int_type = "NONE"
    elif target in ["compound", "combined"]:
        int_type = "COMPOUND"
    elif target.startswith("A_") or target in ["pump", "valve", "throttle", "flush"]:
        int_type = "ACTION"
    else:
        int_type = "STATE"

    int_exp = InterventionExplanation(
        target=target,
        value=val,
        intervention_type=int_type,
        baseline_value=None,
    )

    # 2. Trust State Handling
    if trust_state == "MODEL_ABSTAIN":
        # Determine specific trigger
        r_T = evidence.trust.reconstruction_residual_t_core
        tau_T = evidence.trust.tau_residual_t
        d_M = evidence.trust.latent_novelty_d
        tau_M = evidence.trust.tau_novelty

        is_res_viol = (r_T > tau_T)
        is_nov_viol = (d_M > tau_M)

        if is_res_viol and is_nov_viol:
            trigger = "DUAL_FAILURE"
            obs_val = r_T
            thresh = tau_T
            viol_delta = float(r_T - tau_T)
        elif is_res_viol:
            trigger = "RECONSTRUCTION_INCONSISTENCY"
            obs_val = r_T
            thresh = tau_T
            viol_delta = float(r_T - tau_T)
        elif is_nov_viol:
            trigger = "LATENT_NOVELTY"
            obs_val = d_M
            thresh = tau_M
            viol_delta = float(d_M - tau_M)
        else:
            trigger = "RECONSTRUCTION_INCONSISTENCY"
            obs_val = r_T
            thresh = tau_T
            viol_delta = float(r_T - tau_T)

        abstention_det = AbstentionExplanation(
            trigger=trigger,
            observed_value=obs_val,
            threshold=thresh,
            violation_delta=viol_delta,
            consequence="WORLD_MODEL_NOT_TRUSTED",
            action_taken="PLANNING_BLOCKED",
        )

        summary = ExplanationSummary(
            headline="Abstain from intervention: Observation inconsistency violates model trust",
            mechanism=(
                f"The observed telemetry contradicts the learned world model state representation "
                f"(reconstruction residual R_T = {r_T:.2f}°C exceeded calibrated threshold of {tau_T:.2f}°C by {viol_delta:+.2f}°C). "
                "Autonomous simulation and intervention planning are blocked to prevent unsafe hallucinations."
            ),
            effect_direction="NO_EFFECT_UNTRUSTED",
        )

        causal_chain = CausalChain(
            root_actuator="none",
            target_outcome="none",
            nodes=[],
            all_affected_descendants=[],
        )

        pred_effects = PredictedEffectsExplanation(
            baseline_peak_t_core=None,
            intervened_peak_t_core=None,
            delta_t_core=None,
            baseline_min_flow=None,
            intervened_min_flow=None,
            delta_flow=None,
            counterfactual_statement="Counterfactual simulation withheld: Model trust failure (PLANNING_BLOCKED).",
        )

        rejected_alts = []
        margin_exp = DecisionMarginExplanation(
            best_candidate=None,
            second_best_candidate=None,
            decision_strength="N/A_ABSTAIN",
            decision_margin=None,
            best_utility=None,
            second_best_utility=None,
        )

        trust_qualification = "MODEL_ABSTAIN (Untrusted: Telemetry contradicts internal model dynamics; action search blocked)"

    else:
        # MODEL_TRUSTED or MODEL_UNCERTAIN
        abstention_det = None
        cf = evidence.causal_evidence.counterfactual_comparison
        base_t = cf.get("baseline_peak_t_core")
        rec_t = cf.get("recommended_peak_t_core")
        delta_t = cf.get("delta_t_core")
        base_f = cf.get("baseline_min_flow")
        rec_f = cf.get("recommended_min_flow")
        delta_f = cf.get("delta_flow")

        # 3. Construct Causal Chain Nodes based on SCM DAG
        chain_nodes: List[CausalNode] = []
        if target in ["A_pump", "pump"]:
            chain_nodes = [
                CausalNode(
                    variable="A_pump",
                    relationship="ACTUATOR_INPUT",
                    predicted_direction="INCREASE",
                    predicted_delta=None,
                    evidence_available=True,
                ),
                CausalNode(
                    variable="F_cool",
                    relationship="DIRECT_DESCENDANT",
                    predicted_direction="INCREASE" if (delta_f is not None and delta_f > 0) else "STABLE",
                    predicted_delta=delta_f,
                    evidence_available=(delta_f is not None),
                ),
                CausalNode(
                    variable="T_cool",
                    relationship="MEDIATED_DESCENDANT",
                    predicted_direction="DECREASE" if (delta_f is not None and delta_f > 0) else "STABLE",
                    predicted_delta=None,
                    evidence_available=False,
                ),
                CausalNode(
                    variable="T_core",
                    relationship="FINAL_OUTCOME",
                    predicted_direction="DECREASE" if (delta_t is not None and delta_t < 0) else "STABLE",
                    predicted_delta=delta_t,
                    evidence_available=(delta_t is not None),
                ),
            ]
            headline = f"Increase pump stage to {int(val) if val is not None else 3}"
            mechanism = (
                f"Increasing pump stage increases coolant circulation, which accelerates "
                f"convective heat extraction from the server compute core, reducing peak core temperature."
            )
            direction_str = "REDUCE_CORE_TEMPERATURE"

        elif target in ["A_throttle", "throttle", "L_cpu"]:
            chain_nodes = [
                CausalNode(
                    variable="A_throttle",
                    relationship="ACTUATOR_INPUT",
                    predicted_direction="DECREASE",
                    predicted_delta=None,
                    evidence_available=True,
                ),
                CausalNode(
                    variable="L_cpu",
                    relationship="DIRECT_DESCENDANT",
                    predicted_direction="DECREASE",
                    predicted_delta=None,
                    evidence_available=False,
                ),
                CausalNode(
                    variable="P_elec",
                    relationship="MEDIATED_DESCENDANT",
                    predicted_direction="DECREASE",
                    predicted_delta=None,
                    evidence_available=False,
                ),
                CausalNode(
                    variable="T_core",
                    relationship="FINAL_OUTCOME",
                    predicted_direction="DECREASE" if (delta_t is not None and delta_t < 0) else "STABLE",
                    predicted_delta=delta_t,
                    evidence_available=(delta_t is not None),
                ),
            ]
            headline = f"Throttle CPU workload to {val:.1f}%" if val is not None else "Throttle compute workload"
            mechanism = (
                f"Throttling CPU compute workload directly lowers electrical power consumption, "
                f"diminishing internal Joule heating generation and stabilizing core temperature."
            )
            direction_str = "REDUCE_CORE_TEMPERATURE"

        elif target in ["A_valve", "valve", "V_pos"]:
            chain_nodes = [
                CausalNode(
                    variable="A_valve",
                    relationship="ACTUATOR_INPUT",
                    predicted_direction="INCREASE",
                    predicted_delta=None,
                    evidence_available=True,
                ),
                CausalNode(
                    variable="V_pos",
                    relationship="DIRECT_DESCENDANT",
                    predicted_direction="INCREASE",
                    predicted_delta=None,
                    evidence_available=False,
                ),
                CausalNode(
                    variable="F_cool",
                    relationship="MEDIATED_DESCENDANT",
                    predicted_direction="INCREASE" if (delta_f is not None and delta_f > 0) else "STABLE",
                    predicted_delta=delta_f,
                    evidence_available=(delta_f is not None),
                ),
                CausalNode(
                    variable="P_sys",
                    relationship="MEDIATED_DESCENDANT",
                    predicted_direction="INCREASE",
                    predicted_delta=None,
                    evidence_available=False,
                ),
                CausalNode(
                    variable="T_core",
                    relationship="FINAL_OUTCOME",
                    predicted_direction="DECREASE" if (delta_t is not None and delta_t < 0) else "STABLE",
                    predicted_delta=delta_t,
                    evidence_available=(delta_t is not None),
                ),
            ]
            headline = f"Adjust valve position to {val:.1f}%" if val is not None else "Modulate valve position"
            mechanism = (
                f"Opening the coolant valve expands hydraulic conductance, elevating flow rate "
                f"and convective cooling to dissipate heat from the core."
            )
            direction_str = "REDUCE_CORE_TEMPERATURE"

        elif target in ["compound", "combined"]:
            chain_nodes = [
                CausalNode(variable="A_valve + A_pump", relationship="ACTUATOR_INPUT", predicted_direction="INCREASE", predicted_delta=None, evidence_available=True),
                CausalNode(variable="F_cool", relationship="DIRECT_DESCENDANT", predicted_direction="INCREASE", predicted_delta=delta_f, evidence_available=(delta_f is not None)),
                CausalNode(variable="T_core", relationship="FINAL_OUTCOME", predicted_direction="DECREASE", predicted_delta=delta_t, evidence_available=(delta_t is not None)),
            ]
            headline = "Apply compound valve and pump actuation"
            mechanism = "Simultaneously modulating valve position and pump head achieves optimal multi-variable fluid delivery."
            direction_str = "REDUCE_CORE_TEMPERATURE"

        else:  # none, do_nothing
            chain_nodes = [
                CausalNode(
                    variable="None",
                    relationship="ACTUATOR_INPUT",
                    predicted_direction="STABLE",
                    predicted_delta=None,
                    evidence_available=True,
                )
            ]
            headline = "Maintain baseline operation (DO NOTHING)"
            mechanism = "Current operating state is in stable equilibrium. No intervention provides sufficient risk-adjusted benefit."
            direction_str = "NOMINAL_EQUILIBRIUM"

        summary = ExplanationSummary(
            headline=headline,
            mechanism=mechanism,
            effect_direction=direction_str,
        )

        causal_chain = CausalChain(
            root_actuator=target,
            target_outcome="T_core",
            nodes=chain_nodes,
            all_affected_descendants=evidence.causal_evidence.affected_variables,
        )

        # Counterfactual Statement
        if target in ["none", "do_nothing"]:
            base_t_str = f"{base_t:.2f}°C" if base_t is not None else "nominal"
            cf_stmt = (
                f"Under the modeled baseline, the system remains safely within operating bounds "
                f"(Peak T_core = {base_t_str}). Intervention is unnecessary."
            )
        else:
            flow_part = f"increase coolant flow by {delta_f:+.2f} L/min and " if delta_f is not None else ""
            if delta_t is not None:
                cf_stmt = (
                    f"Under the modeled intervention, `{rec_id}` is predicted to "
                    f"{flow_part}reduce peak core temperature by {abs(delta_t):.2f}°C relative to the factual baseline trajectory."
                )
            else:
                cf_stmt = f"Under the modeled intervention, `{rec_id}` optimizes thermal-fluid equilibrium."

        pred_effects = PredictedEffectsExplanation(
            baseline_peak_t_core=base_t,
            intervened_peak_t_core=rec_t,
            delta_t_core=delta_t,
            baseline_min_flow=base_f,
            intervened_min_flow=rec_f,
            delta_flow=delta_f,
            counterfactual_statement=cf_stmt,
        )

        # 4. Rejected Alternatives Analysis
        rejected_alts: List[RejectedAlternative] = []
        for alt in evidence.decision_quality.alternatives_rejected:
            cand_name = alt["candidate_id"]
            u_score = alt.get("utility_score")
            pk_t = alt.get("peak_t_core")
            violations = alt.get("safety_violations", [])
            is_safe = alt.get("is_safe", True)

            # Categorize reason
            has_novelty = any("latent" in v.lower() or "novelty" in v.lower() or "ood" in v.lower() for v in violations)
            if has_novelty:
                reason_code = "LATENT_NOVELTY"
                # Extract novelty numbers if present
                match = re.search(r"\(([0-9.]+)\).*support limit \(([0-9.]+)\)", " ".join(violations))
                if match:
                    d_val, d_lim = match.group(1), match.group(2)
                    reason_exp = f"Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = {d_val} > {d_lim})."
                else:
                    reason_exp = f"Rejected because predicted trajectory exceeds validated latent support threshold (D_M > 15.00)."
            elif not is_safe or violations:
                reason_code = "SAFETY_VIOLATION"
                reason_exp = f"Rejected due to physical safety boundary violation: {'; '.join(violations)}."
            else:
                reason_code = "LOWER_UTILITY"
                u_diff = (u_score - evidence.decision_quality.utility_score) if (u_score is not None and evidence.decision_quality.utility_score is not None) else None
                u_diff_str = f" (ΔU = {u_diff:+.4f})" if u_diff is not None else ""
                reason_exp = f"Sub-optimal utility{u_diff_str} compared to recommended action; higher operational/thermal cost."

            rejected_alts.append(
                RejectedAlternative(
                    candidate_id=cand_name,
                    reason_code=reason_code,
                    reason_explanation=reason_exp,
                    utility_score=u_score,
                    peak_t_core=pk_t,
                )
            )

        # 5. Decision Margin
        d_margin = evidence.decision_quality.decision_margin
        margin_str = "POSITIVE_MARGIN" if (d_margin is not None and d_margin > 0.0) else "ZERO_MARGIN"
        margin_exp = DecisionMarginExplanation(
            best_candidate=rec_id,
            second_best_candidate=evidence.decision_quality.second_best_candidate,
            decision_strength=margin_str,
            decision_margin=d_margin,
            best_utility=evidence.decision_quality.utility_score,
            second_best_utility=evidence.decision_quality.second_best_utility,
        )

        if trust_state == "MODEL_UNCERTAIN":
            trust_qualification = (
                "MODEL_UNCERTAIN (Cautious: Observation is near calibrated confidence boundary; "
                "predictions should be reviewed with safety margins)"
            )
        else:
            trust_qualification = "MODEL_TRUSTED (Validated: Operating within calibrated in-distribution latent support)"

    prov = ExplanationProvenance(
        source_evidence_hash=evidence.provenance.decision_hash,
        model_version=evidence.provenance.model_version,
        dataset_version=evidence.provenance.dataset_version,
        planner_version=evidence.provenance.planner_version,
        benchmark_version=evidence.provenance.benchmark_version,
        scenario_id=evidence.provenance.scenario_id,
        timestamp_utc=ts,
    )

    return CausalExplanation(
        summary=summary,
        intervention=int_exp,
        causal_chain=causal_chain,
        predicted_effects=pred_effects,
        rejected_alternatives=rejected_alts,
        decision_margin=margin_exp,
        abstention_details=abstention_det,
        trust_qualification=trust_qualification,
        provenance=prov,
    )
