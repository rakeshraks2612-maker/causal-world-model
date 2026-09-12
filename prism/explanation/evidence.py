"""PRISM Decision Evidence Contract (Task 6.1).

Defines the machine-readable DecisionEvidence data structures, deterministic
causal path extraction from the SCM DAG, exact safety margin calculations,
and cryptographic provenance tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from prism.planning.planner import PlanRecommendation
from prism.planning.objectives import CandidateEvaluation
from prism.evaluation.model_trust import TrustDiagnostics, ModelTrustState


# Canonical Physical Hard Safety Boundaries
SAFETY_CONSTRAINTS = {
    "max_t_core": 95.0,        # °C
    "max_pressure": 5.5,       # bar
    "min_flow": 8.0,           # L/min
    "max_latent_d": 15.0,      # Mahalanobis distance
}


@dataclass
class DecisionSummary:
    """Core decision outcome emitted by PRISM."""
    recommendation: Optional[str]
    decision_status: str  # "RECOMMENDED", "NO_ACTION_REQUIRED", "BLOCKED"
    abstention_reason: Optional[str] = None


@dataclass
class TrustEvidence:
    """Upfront model trust and observational-latent consistency evidence."""
    trust_state: str  # "MODEL_TRUSTED", "MODEL_UNCERTAIN", "MODEL_ABSTAIN"
    reconstruction_residual_t_core: float
    reconstruction_residual_8d: float
    latent_novelty_d: float
    epistemic_uncertainty_sigma: float
    tau_residual_t: float
    tau_novelty: float
    trust_reason: str


@dataclass
class CausalEvidence:
    """Causal DAG pathway and counterfactual comparison."""
    intervention_target: str
    intervention_value: Optional[float]
    affected_variables: List[str]
    causal_path: List[str]
    direction_of_effect: str
    counterfactual_comparison: Dict[str, Any]


@dataclass
class PredictedOutcome:
    """Simulated state trajectory outcomes over planning horizon."""
    peak_t_core: Optional[float]
    max_pressure: Optional[float]
    min_flow: Optional[float]
    mean_cpu_load: Optional[float]
    mean_electrical_power: Optional[float]
    planning_horizon: int


@dataclass
class SafetyEvidence:
    """Rigorous physical safety margins relative to hard operating limits."""
    safety_state: str  # "SAFE", "MARGINAL", "UNSAFE", "ABSTAIN_REQUIRED"
    thermal_margin_c: Optional[float]
    pressure_margin_bar: Optional[float]
    flow_margin_l_min: Optional[float]
    latent_support_margin: Optional[float]
    hard_constraints: Dict[str, float] = field(default_factory=lambda: dict(SAFETY_CONSTRAINTS))
    is_safe: bool = False


@dataclass
class DecisionQuality:
    """Optimization quality, Pareto trade-offs, and alternative ranking."""
    utility_score: Optional[float]
    second_best_candidate: Optional[str]
    second_best_utility: Optional[float]
    decision_margin: Optional[float]
    alternatives_rejected: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvenanceEvidence:
    """Traceable, immutable provenance metadata and cryptographic hash."""
    model_version: str
    dataset_version: str
    planner_version: str
    benchmark_version: str
    scenario_id: Optional[str]
    timestamp_utc: str
    decision_hash: str


@dataclass
class DecisionEvidence:
    """Root Machine-Readable PRISM Decision Evidence Contract."""
    decision: DecisionSummary
    trust: TrustEvidence
    causal_evidence: CausalEvidence
    predicted_outcome: PredictedOutcome
    safety_evidence: SafetyEvidence
    decision_quality: DecisionQuality
    provenance: ProvenanceEvidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_ascii_report(self) -> str:
        """Format an executive text summary of the decision and evidence."""
        lines = [
            "================================================================================",
            "                        PRISM DECISION INTELLIGENCE EVIDENCE                    ",
            "================================================================================",
            f" Recommendation : {self.decision.recommendation or 'NONE (ABSTAIN)'}",
            f" Decision Status: {self.decision.decision_status}",
            f" Model Trust    : {self.trust.trust_state} (Residual T_core: {self.trust.reconstruction_residual_t_core:.2f}°C / Threshold: {self.trust.tau_residual_t:.2f}°C)",
            f" Physical Safety: {self.safety_evidence.safety_state} (Safe: {self.safety_evidence.is_safe})",
            "--------------------------------------------------------------------------------",
            " CAUSAL MECHANISM & PATHWAY:",
            f" Target: {self.causal_evidence.intervention_target} | Direction: {self.causal_evidence.direction_of_effect}",
            f" Path  : {' -> '.join(self.causal_evidence.causal_path)}",
            "--------------------------------------------------------------------------------",
            " SAFETY MARGINS:",
        ]
        if self.safety_evidence.thermal_margin_c is not None:
            lines.append(f" - Thermal Margin : {self.safety_evidence.thermal_margin_c:+.2f}°C (Limit: {self.safety_evidence.hard_constraints['max_t_core']}°C)")
        if self.safety_evidence.pressure_margin_bar is not None:
            lines.append(f" - Pressure Margin: {self.safety_evidence.pressure_margin_bar:+.2f} bar (Limit: {self.safety_evidence.hard_constraints['max_pressure']} bar)")
        if self.safety_evidence.flow_margin_l_min is not None:
            lines.append(f" - Flow Margin    : {self.safety_evidence.flow_margin_l_min:+.2f} L/min (Limit: {self.safety_evidence.hard_constraints['min_flow']} L/min)")
        if self.safety_evidence.latent_support_margin is not None:
            lines.append(f" - Latent Margin  : {self.safety_evidence.latent_support_margin:+.2f} d_M (Limit: {self.safety_evidence.hard_constraints['max_latent_d']})")
        lines.extend([
            "--------------------------------------------------------------------------------",
            " PROVENANCE & INTEGRITY:",
            f" Model: {self.provenance.model_version} | Hash: {self.provenance.decision_hash[:16]}... | Time: {self.provenance.timestamp_utc}",
            "================================================================================",
        ])
        return "\n".join(lines)


# =============================================================================
# Deterministic Causal DAG Path Lookup
# =============================================================================
def get_causal_dag_path(target: str) -> Tuple[List[str], List[str], str]:
    """Derive deterministic causal path, affected variables, and direction from SCM DAG."""
    if target in ["A_pump", "pump"]:
        return (
            ["A_pump", "F_cool", "T_cool", "T_core"],
            ["F_cool", "P_sys", "T_cool", "T_core", "Vib_pump"],
            "INCREASE_COOLANT_CIRCULATION_REDUCE_CORE_TEMP",
        )
    elif target in ["A_throttle", "throttle", "L_cpu"]:
        return (
            ["A_throttle", "L_cpu", "P_elec", "T_core"],
            ["L_cpu", "P_elec", "T_core"],
            "REDUCE_COMPUTE_LOAD_DECREASE_JOULE_HEATING",
        )
    elif target in ["A_valve", "valve", "V_pos"]:
        return (
            ["A_valve", "V_pos", "F_cool", "P_sys", "T_core"],
            ["V_pos", "F_cool", "P_sys", "T_core"],
            "EXPAND_VALVE_CONDUCTANCE_INCREASE_FLOW_HEAT_DISSIPATION",
        )
    elif target in ["compound", "combined"]:
        return (
            ["A_valve", "A_pump", "V_pos", "F_cool", "T_cool", "T_core"],
            ["V_pos", "F_cool", "P_sys", "T_cool", "T_core", "Vib_pump"],
            "SIMULTANEOUS_MULTIVARIATE_ACTUATION",
        )
    else:  # none, do_nothing
        return (
            ["None"],
            [],
            "NOMINAL_PHYSICAL_EQUILIBRIUM",
        )


def compute_deterministic_evidence_hash(data_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonically formatted evidence dict."""
    # Exclude dynamic timestamp or hash field itself if present
    clean_dict = {k: v for k, v in data_dict.items() if k != "decision_hash"}
    canonical_bytes = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


# =============================================================================
# Builder Function
# =============================================================================
def build_decision_evidence(
    recommendation: PlanRecommendation,
    trust_diagnostic: TrustDiagnostics,
    scenario_id: Optional[str] = None,
    model_version: str = "baseline_005",
    dataset_version: str = "unified_v1_40_40_20",
    planner_version: str = "v1.2_trust_gated",
    benchmark_version: str = "v1.0",
    timestamp_utc: Optional[str] = None,
    tau_residual_t: float = 6.08,
    tau_novelty: float = 15.0,
) -> DecisionEvidence:
    """Deterministically construct a complete DecisionEvidence object."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
    
    # 1. Trust Evidence
    if trust_diagnostic.state == ModelTrustState.MODEL_TRUSTED:
        trust_state_str = "MODEL_TRUSTED"
    elif trust_diagnostic.state == ModelTrustState.MODEL_UNCERTAIN:
        trust_state_str = "MODEL_UNCERTAIN"
    else:
        trust_state_str = "MODEL_ABSTAIN"
        
    t_res_tau = getattr(trust_diagnostic, "tau_residual_t", tau_residual_t)
    t_nov_tau = getattr(trust_diagnostic, "tau_novelty", tau_novelty)

    trust_ev = TrustEvidence(
        trust_state=trust_state_str,
        reconstruction_residual_t_core=float(trust_diagnostic.residual_t_core),
        reconstruction_residual_8d=float(trust_diagnostic.residual_8d_norm),
        latent_novelty_d=float(trust_diagnostic.latent_mahalanobis_d),
        epistemic_uncertainty_sigma=float(trust_diagnostic.epistemic_sigma),
        tau_residual_t=float(t_res_tau),
        tau_novelty=float(t_nov_tau),
        trust_reason=trust_diagnostic.rationale,
    )

    # 2. Decision & Abstention Handling
    best_cand: Optional[CandidateEvaluation] = recommendation.recommended_candidate
    all_cands: List[CandidateEvaluation] = recommendation.all_evaluated_candidates or []
    
    if trust_diagnostic.state == ModelTrustState.MODEL_ABSTAIN or recommendation.should_abstain:
        # Abstention state
        is_blocked = (trust_diagnostic.state == ModelTrustState.MODEL_ABSTAIN)
        dec_status = "BLOCKED" if is_blocked else "ABSTAIN_REQUIRED"
        dec_summary = DecisionSummary(
            recommendation=None,
            decision_status=dec_status,
            abstention_reason=recommendation.abstention_reason or trust_diagnostic.rationale,
        )
        
        causal_ev = CausalEvidence(
            intervention_target="none",
            intervention_value=None,
            affected_variables=[],
            causal_path=["None"],
            direction_of_effect="NO_SIMULATION_PERMITTED_UNDER_UNTRUSTED_REGIME",
            counterfactual_comparison={"status": "BLOCKED_BY_TRUST_GATEWAY"},
        )
        
        pred_outcome = PredictedOutcome(
            peak_t_core=None,
            max_pressure=None,
            min_flow=None,
            mean_cpu_load=None,
            mean_electrical_power=None,
            planning_horizon=40,
        )
        
        safety_ev = SafetyEvidence(
            safety_state="ABSTAIN_REQUIRED",
            thermal_margin_c=None,
            pressure_margin_bar=None,
            flow_margin_l_min=None,
            latent_support_margin=float(SAFETY_CONSTRAINTS["max_latent_d"] - trust_diagnostic.latent_mahalanobis_d),
            is_safe=False,
        )
        
        dec_quality = DecisionQuality(
            utility_score=None,
            second_best_candidate=None,
            second_best_utility=None,
            decision_margin=None,
            alternatives_rejected=[],
        )

    else:
        # Trusted Execution
        cand_id = best_cand.candidate_id if best_cand else "cand_do_nothing"
        is_do_nothing = (cand_id == "cand_do_nothing")
        dec_status = "NO_ACTION_REQUIRED" if is_do_nothing else "RECOMMENDED"
        
        dec_summary = DecisionSummary(
            recommendation=cand_id,
            decision_status=dec_status,
            abstention_reason=None,
        )
        
        # Causal & DAG Info
        spec = best_cand.spec if best_cand else None
        if is_do_nothing or spec is None:
            target_name = "none"
            val = None
        elif isinstance(spec, list):
            target_name = "compound"
            val = None
        else:
            target_name = spec.target
            val = float(spec.value)
            
        c_path, aff_vars, direction = get_causal_dag_path(target_name)
        
        # Counterfactual contrast relative to baseline
        base_sim = recommendation.baseline_simulation
        base_peak_t = float(np.max(base_sim.baseline_observations[:, 0])) if base_sim is not None else None
        base_min_f = float(np.min(base_sim.baseline_observations[:, 3])) if base_sim is not None else None
        
        cf_comp = {
            "baseline_peak_t_core": base_peak_t,
            "baseline_min_flow": base_min_f,
            "recommended_peak_t_core": float(best_cand.peak_t_core) if best_cand else None,
            "recommended_min_flow": float(best_cand.min_flow) if best_cand else None,
            "delta_t_core": float(best_cand.peak_t_core - base_peak_t) if (best_cand and base_peak_t) else 0.0,
            "delta_flow": float(best_cand.causal_delta_f_cool) if best_cand else 0.0,
        }
        
        causal_ev = CausalEvidence(
            intervention_target=target_name,
            intervention_value=val,
            affected_variables=aff_vars,
            causal_path=c_path,
            direction_of_effect=direction,
            counterfactual_comparison=cf_comp,
        )
        
        # Predicted Outcome
        p_elec = getattr(best_cand, "mean_power", getattr(best_cand, "mean_electrical_power", None)) if best_cand else None
        pred_outcome = PredictedOutcome(
            peak_t_core=float(best_cand.peak_t_core) if best_cand else None,
            max_pressure=float(best_cand.max_pressure) if best_cand else None,
            min_flow=float(best_cand.min_flow) if best_cand else None,
            mean_cpu_load=float(best_cand.mean_cpu_load) if best_cand else None,
            mean_electrical_power=float(p_elec) if p_elec is not None else None,
            planning_horizon=40,
        )
        
        # Exact Safety Margins
        therm_margin = float(SAFETY_CONSTRAINTS["max_t_core"] - best_cand.peak_t_core) if best_cand else None
        press_margin = float(SAFETY_CONSTRAINTS["max_pressure"] - best_cand.max_pressure) if best_cand else None
        flow_margin = float(best_cand.min_flow - SAFETY_CONSTRAINTS["min_flow"]) if best_cand else None
        latent_margin = float(SAFETY_CONSTRAINTS["max_latent_d"] - trust_diagnostic.latent_mahalanobis_d)
        
        is_safe = bool(best_cand.is_safe) if best_cand else False
        safety_status_str = "SAFE" if is_safe else "UNSAFE"
        
        safety_ev = SafetyEvidence(
            safety_state=safety_status_str,
            thermal_margin_c=therm_margin,
            pressure_margin_bar=press_margin,
            flow_margin_l_min=flow_margin,
            latent_support_margin=latent_margin,
            is_safe=is_safe,
        )
        
        # Decision Quality & Alternatives
        # Sort candidates by utility descending
        sorted_cands = sorted(all_cands, key=lambda c: c.utility_score, reverse=True)
        second_cand = None
        for c in sorted_cands:
            if best_cand and c.candidate_id != best_cand.candidate_id:
                second_cand = c
                break
                
        dec_margin = float(best_cand.utility_score - second_cand.utility_score) if (best_cand and second_cand) else None
        
        rej_alts = []
        for c in all_cands:
            if best_cand and c.candidate_id == best_cand.candidate_id:
                continue
            rej_alts.append({
                "candidate_id": c.candidate_id,
                "is_safe": bool(c.is_safe),
                "utility_score": float(c.utility_score),
                "peak_t_core": float(c.peak_t_core),
                "safety_violations": c.safety_violations,
            })
            
        dec_quality = DecisionQuality(
            utility_score=float(best_cand.utility_score) if best_cand else None,
            second_best_candidate=second_cand.candidate_id if second_cand else None,
            second_best_utility=float(second_cand.utility_score) if second_cand else None,
            decision_margin=dec_margin,
            alternatives_rejected=rej_alts,
        )

    # 3. Provenance & Cryptographic Hash
    prelim_dict = {
        "decision": asdict(dec_summary),
        "trust": asdict(trust_ev),
        "causal_evidence": asdict(causal_ev),
        "predicted_outcome": asdict(pred_outcome),
        "safety_evidence": asdict(safety_ev),
        "decision_quality": asdict(dec_quality),
        "model_version": model_version,
        "dataset_version": dataset_version,
        "planner_version": planner_version,
        "benchmark_version": benchmark_version,
        "scenario_id": scenario_id,
        "timestamp_utc": ts,
    }
    dec_hash = compute_deterministic_evidence_hash(prelim_dict)
    
    prov_ev = ProvenanceEvidence(
        model_version=model_version,
        dataset_version=dataset_version,
        planner_version=planner_version,
        benchmark_version=benchmark_version,
        scenario_id=scenario_id,
        timestamp_utc=ts,
        decision_hash=dec_hash,
    )

    return DecisionEvidence(
        decision=dec_summary,
        trust=trust_ev,
        causal_evidence=causal_ev,
        predicted_outcome=pred_outcome,
        safety_evidence=safety_ev,
        decision_quality=dec_quality,
        provenance=prov_ev,
    )
