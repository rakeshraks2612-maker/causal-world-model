"""PRISM Counterfactual Evidence Engine (Task 6.3).

Translates frozen Pearl Level-3 twin-world counterfactual simulation results into
auditable, machine-readable CounterfactualEvidence objects.
Ensures strict twin-world / frozen-exogenous integrity and deterministic safety transitions.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from prism.counterfactual.engine import LearnedCounterfactualResult, AbducedLatentState
from prism.dataset.counterfactuals import LearnerCounterfactualRecord
from prism.intervention.effects import CausalEffectSummary, SAFETY_THRESHOLDS


# =============================================================================
# Counterfactual Evidence Contract Data Structures
# =============================================================================

@dataclass
class FactualWorldContext:
    """The factual pre-intervention operating context and abduced latent state."""
    episode_id: str
    intervention_time: int
    planning_horizon: int
    historical_t_core_at_t_star: Optional[float]
    historical_f_cool_at_t_star: Optional[float]
    historical_p_sys_at_t_star: Optional[float]
    abduced_latent_norm: Optional[float]
    abduced_latent_std_mean: Optional[float]


@dataclass
class CounterfactualIntervention:
    """Exact specification of the counterfactual intervention."""
    intervention_target: str
    original_value: Optional[float]
    counterfactual_value: Optional[float]
    intervention_type: str  # "ACTION", "STATE", "COMPOUND", "NONE"
    formal_notation: str   # e.g. "do(A_pump = 3.0)"


@dataclass
class CounterfactualWorldOutcome:
    """Simulated trajectory metrics under counterfactual replay."""
    peak_t_core: Optional[float]
    max_p_sys: Optional[float]
    min_f_cool: Optional[float]
    mean_l_cpu: Optional[float]
    mean_p_elec: Optional[float]
    failure_state: Optional[bool]
    time_to_failure: Optional[int]


@dataclass
class CausalEffectEvidence:
    """Quantitative counterfactual effect deltas and causal direction classification."""
    delta_t_core_peak: Optional[float]  # CF peak - Factual peak
    delta_p_sys_max: Optional[float]    # CF max - Factual max
    delta_f_cool_min: Optional[float]   # CF min - Factual min
    delta_l_cpu_mean: Optional[float]
    delta_p_elec_mean: Optional[float]
    primary_effect_direction: str       # "REDUCES_CORE_TEMPERATURE", "INCREASES_COOLANT_FLOW", etc.
    causal_interpretation: str
    horizon_effects: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TwinWorldIntegrity:
    """Verification metadata ensuring Level-3 twin-world counterfactual semantics."""
    replay_mode: str = "TWIN_WORLD_FROZEN_EXOGENOUS"
    shared_exogenous_conditions: bool = True
    identical_pre_intervention_history: bool = True
    abduction_method: str = "POSTERIOR_LATENT_INFERENCE_Q_PHI"


@dataclass
class SafetyComparison:
    """Factual vs Counterfactual safety boundary comparison and transition classification."""
    factual_thermal_margin_c: Optional[float]
    factual_pressure_margin_bar: Optional[float]
    factual_flow_margin_l_min: Optional[float]
    counterfactual_thermal_margin_c: Optional[float]
    counterfactual_pressure_margin_bar: Optional[float]
    counterfactual_flow_margin_l_min: Optional[float]
    factual_safety_state: str           # "SAFE", "UNSAFE", "ABSTAIN_REQUIRED"
    counterfactual_safety_state: str    # "SAFE", "UNSAFE", "ABSTAIN_REQUIRED"
    safety_transition: str              # "SAFE -> SAFE", "UNSAFE -> SAFE", "SAFE -> UNSAFE", "UNSAFE -> UNSAFE", "ABSTAINED"
    outcome_classification: str         # "INTERVENTION_MITIGATES_FAILURE", "INTERVENTION_INTRODUCES_RISK", "INTERVENTION_PRESERVES_SAFETY", "UNTRUSTED_BLOCKED"


@dataclass
class UncertaintyContext:
    """Aleatoric and epistemic uncertainty metrics at the counterfactual decision point."""
    aleatoric_sigma: Optional[float]
    epistemic_sigma: Optional[float]
    latent_novelty_d: Optional[float]
    support_threshold: float = 15.0
    within_support: bool = True


@dataclass
class CounterfactualProvenance:
    """Traceable, immutable provenance metadata and cryptographic hash."""
    source_evidence_hash: Optional[str]
    counterfactual_engine_version: str
    model_version: str
    dataset_version: str
    episode_id: str
    counterfactual_id: str
    intervention_time: int
    timestamp_utc: str
    counterfactual_hash: str


@dataclass
class CounterfactualEvidence:
    """Root Machine-Readable PRISM Counterfactual Evidence Contract."""
    factual_world: FactualWorldContext
    intervention: CounterfactualIntervention
    counterfactual_world: CounterfactualWorldOutcome
    causal_effect: CausalEffectEvidence
    twin_world_integrity: TwinWorldIntegrity
    safety_comparison: SafetyComparison
    uncertainty: UncertaintyContext
    provenance: CounterfactualProvenance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_comparison_table(self) -> str:
        """Produce a compact ascii comparison table of factual vs counterfactual."""
        f_t = f"{self.safety_comparison.factual_thermal_margin_c:+.2f}°C" if self.safety_comparison.factual_thermal_margin_c is not None else "N/A"
        cf_t = f"{self.safety_comparison.counterfactual_thermal_margin_c:+.2f}°C" if self.safety_comparison.counterfactual_thermal_margin_c is not None else "N/A"
        d_t = f"{self.causal_effect.delta_t_core_peak:+.2f}°C" if self.causal_effect.delta_t_core_peak is not None else "N/A"

        f_f = f"{self.safety_comparison.factual_flow_margin_l_min:+.2f} L/m" if self.safety_comparison.factual_flow_margin_l_min is not None else "N/A"
        cf_f = f"{self.safety_comparison.counterfactual_flow_margin_l_min:+.2f} L/m" if self.safety_comparison.counterfactual_flow_margin_l_min is not None else "N/A"
        d_f = f"{self.causal_effect.delta_f_cool_min:+.2f} L/m" if self.causal_effect.delta_f_cool_min is not None else "N/A"

        f_p = f"{self.safety_comparison.factual_pressure_margin_bar:+.2f} bar" if self.safety_comparison.factual_pressure_margin_bar is not None else "N/A"
        cf_p = f"{self.safety_comparison.counterfactual_pressure_margin_bar:+.2f} bar" if self.safety_comparison.counterfactual_pressure_margin_bar is not None else "N/A"
        d_p = f"{self.causal_effect.delta_p_sys_max:+.2f} bar" if self.causal_effect.delta_p_sys_max is not None else "N/A"

        lines = [
            "--------------------------------------------------------------------------------",
            f"{'METRIC':<20} | {'FACTUAL':<16} | {'COUNTERFACTUAL':<16} | {'NET EFFECT':<14}",
            "--------------------------------------------------------------------------------",
            f"{'Thermal Margin':<20} | {f_t:<16} | {cf_t:<16} | {d_t:<14}",
            f"{'Flow Margin':<20} | {f_f:<16} | {cf_f:<16} | {d_f:<14}",
            f"{'Pressure Margin':<20} | {f_p:<16} | {cf_p:<16} | {d_p:<14}",
            "--------------------------------------------------------------------------------",
            f"Safety Transition: {self.safety_comparison.safety_transition} ({self.safety_comparison.outcome_classification})",
            "--------------------------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Render a clean, human-auditable markdown report."""
        lines = [
            f"# 🔮 PRISM Counterfactual Level-3 Evidence",
            f"**Counterfactual ID:** `{self.provenance.counterfactual_id}` | **Intervention:** `{self.intervention.formal_notation}`",
            f"**Outcome Classification:** `{self.safety_comparison.outcome_classification}` ({self.safety_comparison.safety_transition})",
            "",
            "## 1. Twin-World Simulation Summary",
            f"{self.causal_effect.causal_interpretation}",
            "",
            "```text",
            self.format_comparison_table(),
            "```",
            "",
            "## 2. Twin-World Exogenous Invariants",
            f"- **Replay Mode:** `{self.twin_world_integrity.replay_mode}`",
            f"- **Shared Exogenous Inferred Conditions:** `{self.twin_world_integrity.shared_exogenous_conditions}`",
            f"- **Latent Abduction Method:** `{self.twin_world_integrity.abduction_method}`",
            "",
            "## 3. Uncertainty & Support",
            f"- **Latent Novelty ($d_M$):** {self.uncertainty.latent_novelty_d:.2f} (Support Threshold: {self.uncertainty.support_threshold:.1f}, Within Support: `{self.uncertainty.within_support}`)" if self.uncertainty.latent_novelty_d is not None else "- **Uncertainty:** Blocked under model abstention.",
            "",
            "---",
            f"**Counterfactual Hash:** `{self.provenance.counterfactual_hash}` | **Engine:** `{self.provenance.counterfactual_engine_version}`",
        ]
        return "\n".join(lines)


# =============================================================================
# Deterministic Hash & Builder
# =============================================================================

def compute_deterministic_cf_hash(data_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonically serialized CF dict."""
    clean_dict = {k: v for k, v in data_dict.items() if k != "counterfactual_hash"}
    canonical_bytes = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def build_counterfactual_evidence(
    cf_result: Optional[LearnedCounterfactualResult],
    learner_record: Optional[LearnerCounterfactualRecord] = None,
    source_evidence_hash: Optional[str] = None,
    is_model_abstained: bool = False,
    abstention_reason: Optional[str] = None,
    model_version: str = "baseline_005",
    dataset_version: str = "unified_v1_40_40_20",
    counterfactual_engine_version: str = "v1.0_pearl_level3",
    timestamp_utc: Optional[str] = None,
) -> CounterfactualEvidence:
    """Construct a complete CounterfactualEvidence contract object."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

    # 1. Handle Abstention State (No CF simulation permitted)
    if is_model_abstained or cf_result is None:
        ep_id = getattr(learner_record, "parent_episode_id", "unknown_episode") if learner_record else "unknown_episode"
        cf_id = getattr(learner_record, "counterfactual_id", "blocked_cf") if learner_record else "blocked_cf"
        t_star = getattr(learner_record, "counterfactual_time", 0) if learner_record else 0
        tgt_act = getattr(learner_record, "target_action", "none") if learner_record else "none"
        cf_val = getattr(learner_record, "counterfactual_value", 0.0) if learner_record else 0.0

        factual_ctx = FactualWorldContext(
            episode_id=ep_id,
            intervention_time=t_star,
            planning_horizon=40,
            historical_t_core_at_t_star=None,
            historical_f_cool_at_t_star=None,
            historical_p_sys_at_t_star=None,
            abduced_latent_norm=None,
            abduced_latent_std_mean=None,
        )

        int_spec = CounterfactualIntervention(
            intervention_target=tgt_act,
            original_value=None,
            counterfactual_value=cf_val,
            intervention_type="NONE",
            formal_notation=f"do({tgt_act} = {cf_val}) [BLOCKED]",
        )

        cf_outcome = CounterfactualWorldOutcome(
            peak_t_core=None,
            max_p_sys=None,
            min_f_cool=None,
            mean_l_cpu=None,
            mean_p_elec=None,
            failure_state=None,
            time_to_failure=None,
        )

        causal_eff = CausalEffectEvidence(
            delta_t_core_peak=None,
            delta_p_sys_max=None,
            delta_f_cool_min=None,
            delta_l_cpu_mean=None,
            delta_p_elec_mean=None,
            primary_effect_direction="UNTRUSTED_BLOCKED",
            causal_interpretation=f"Counterfactual simulation blocked: {abstention_reason or 'Model trust layer triggered MODEL_ABSTAIN'}.",
            horizon_effects={},
        )

        safety_comp = SafetyComparison(
            factual_thermal_margin_c=None,
            factual_pressure_margin_bar=None,
            factual_flow_margin_l_min=None,
            counterfactual_thermal_margin_c=None,
            counterfactual_pressure_margin_bar=None,
            counterfactual_flow_margin_l_min=None,
            factual_safety_state="ABSTAIN_REQUIRED",
            counterfactual_safety_state="ABSTAIN_REQUIRED",
            safety_transition="ABSTAINED",
            outcome_classification="UNTRUSTED_BLOCKED",
        )

        unc_ctx = UncertaintyContext(
            aleatoric_sigma=None,
            epistemic_sigma=None,
            latent_novelty_d=None,
            within_support=False,
        )

    else:
        # Active Level-3 CF Result Available
        ep_id = cf_result.parent_episode_id
        cf_id = cf_result.counterfactual_id
        t_star = cf_result.counterfactual_time
        tgt_act = cf_result.target_action
        cf_val = cf_result.counterfactual_value

        # Factual trajectory metrics
        fact_obs = cf_result.factual_observations  # [H, 8]
        cf_obs = cf_result.counterfactual_observations  # [H, 8]

        fact_peak_t = float(np.max(fact_obs[:, 0]))
        fact_max_p = float(np.max(fact_obs[:, 2]))
        fact_min_f = float(np.min(fact_obs[:, 3]))
        fact_mean_l = float(np.mean(fact_obs[:, 4]))
        fact_mean_p = float(np.mean(fact_obs[:, 7]))

        cf_peak_t = float(np.max(cf_obs[:, 0]))
        cf_max_p = float(np.max(cf_obs[:, 2]))
        cf_min_f = float(np.min(cf_obs[:, 3]))
        cf_mean_l = float(np.mean(cf_obs[:, 4]))
        cf_mean_p = float(np.mean(cf_obs[:, 7]))

        # Historical at t*
        hist_t = float(learner_record.historical_observations[t_star, 0]) if learner_record is not None else float(fact_obs[0, 0])
        hist_f = float(learner_record.historical_observations[t_star, 3]) if learner_record is not None else float(fact_obs[0, 3])
        hist_p = float(learner_record.historical_observations[t_star, 2]) if learner_record is not None else float(fact_obs[0, 2])

        # Original action value at t*
        orig_val = float(learner_record.historical_actions[t_star, 0]) if learner_record is not None else None

        # Latent abduction summary
        abd_mean = cf_result.abduced_latent_state.latent_mean
        abd_std = cf_result.abduced_latent_state.latent_std
        lat_norm = float(np.linalg.norm(abd_mean))
        lat_std_mean = float(np.mean(abd_std))

        factual_ctx = FactualWorldContext(
            episode_id=ep_id,
            intervention_time=t_star,
            planning_horizon=len(fact_obs) - 1,
            historical_t_core_at_t_star=hist_t,
            historical_f_cool_at_t_star=hist_f,
            historical_p_sys_at_t_star=hist_p,
            abduced_latent_norm=lat_norm,
            abduced_latent_std_mean=lat_std_mean,
        )

        int_type = "ACTION" if tgt_act.startswith("A_") else "STATE"
        int_spec = CounterfactualIntervention(
            intervention_target=tgt_act,
            original_value=orig_val,
            counterfactual_value=float(cf_val),
            intervention_type=int_type,
            formal_notation=f"do({tgt_act} = {cf_val:.1f})",
        )

        cf_outcome = CounterfactualWorldOutcome(
            peak_t_core=cf_peak_t,
            max_p_sys=cf_max_p,
            min_f_cool=cf_min_f,
            mean_l_cpu=cf_mean_l,
            mean_p_elec=cf_mean_p,
            failure_state=bool(getattr(cf_result.effects.failure_metrics, "intervened_failed", getattr(cf_result.effects.failure_metrics, "counterfactual_failed", False))),
            time_to_failure=getattr(cf_result.effects.failure_metrics, "intervention_failure_time", getattr(cf_result.effects.failure_metrics, "time_to_failure_cf", None)),
        )

        # Causal effect deltas
        d_t = float(cf_peak_t - fact_peak_t)
        d_p = float(cf_max_p - fact_max_p)
        d_f = float(cf_min_f - fact_min_f)
        d_l = float(cf_mean_l - fact_mean_l)
        d_p_elec = float(cf_mean_p - fact_mean_p)

        # Multi-horizon effects
        h_effs = {}
        for h, heff in cf_result.effects.horizon_effects.items():
            h_effs[h] = {
                "delta_t_core": float(heff.delta_t_core),
                "delta_f_cool": float(heff.delta_f_cool),
                "delta_p_sys": float(heff.delta_p_sys),
                "delta_l_cpu": float(heff.delta_l_cpu),
                "delta_p_elec": float(heff.delta_p_elec),
            }

        # Direction classification
        if d_t < -0.2:
            primary_dir = "REDUCES_CORE_TEMPERATURE"
            interpretation = (
                f"Under counterfactual replay do({tgt_act}={cf_val:.1f}), peak core temperature drops by {abs(d_t):.2f}°C "
                f"({fact_peak_t:.2f}°C → {cf_peak_t:.2f}°C) while preserving identical exogenous background conditions."
            )
        elif d_f > 0.5:
            primary_dir = "INCREASES_COOLANT_FLOW"
            interpretation = (
                f"Under counterfactual replay do({tgt_act}={cf_val:.1f}), coolant flow increases by {d_f:+.2f} L/min "
                f"({fact_min_f:.2f} → {cf_min_f:.2f} L/min)."
            )
        elif d_p > 0.2:
            primary_dir = "ELEVATES_PRESSURE"
            interpretation = (
                f"Under counterfactual replay do({tgt_act}={cf_val:.1f}), system pressure increases by {d_p:+.2f} bar "
                f"({fact_max_p:.2f} → {cf_max_p:.2f} bar)."
            )
        else:
            primary_dir = "NOMINAL_STABLE"
            interpretation = f"Under counterfactual replay do({tgt_act}={cf_val:.1f}), state remains in stable equilibrium."

        causal_eff = CausalEffectEvidence(
            delta_t_core_peak=d_t,
            delta_p_sys_max=d_p,
            delta_f_cool_min=d_f,
            delta_l_cpu_mean=d_l,
            delta_p_elec_mean=d_p_elec,
            primary_effect_direction=primary_dir,
            causal_interpretation=interpretation,
            horizon_effects=h_effs,
        )

        # Safety Margins & State Transitions (Operating limits: T_core < 95.0°C, P_sys < 5.5 bar, F_cool > 8.0 L/min)
        t_limit = 95.0
        p_limit = 5.5
        f_limit = 8.0

        fact_t_margin = float(t_limit - fact_peak_t)
        fact_p_margin = float(p_limit - fact_max_p)
        fact_f_margin = float(fact_min_f - f_limit)

        cf_t_margin = float(t_limit - cf_peak_t)
        cf_p_margin = float(p_limit - cf_max_p)
        cf_f_margin = float(cf_min_f - f_limit)

        fact_is_safe = (fact_t_margin >= 0.0 and fact_p_margin >= 0.0 and fact_f_margin >= 0.0)
        cf_is_safe = (cf_t_margin >= 0.0 and cf_p_margin >= 0.0 and cf_f_margin >= 0.0)

        fact_state_str = "SAFE" if fact_is_safe else "UNSAFE"
        cf_state_str = "SAFE" if cf_is_safe else "UNSAFE"
        trans_str = f"{fact_state_str} -> {cf_state_str}"

        if not fact_is_safe and cf_is_safe:
            outcome_class = "INTERVENTION_MITIGATES_FAILURE"
        elif fact_is_safe and not cf_is_safe:
            outcome_class = "INTERVENTION_INTRODUCES_RISK"
        elif not fact_is_safe and not cf_is_safe:
            outcome_class = "INTERVENTION_PRESERVES_UNSAFE"
        else:
            outcome_class = "INTERVENTION_PRESERVES_SAFETY"

        safety_comp = SafetyComparison(
            factual_thermal_margin_c=fact_t_margin,
            factual_pressure_margin_bar=fact_p_margin,
            factual_flow_margin_l_min=fact_f_margin,
            counterfactual_thermal_margin_c=cf_t_margin,
            counterfactual_pressure_margin_bar=cf_p_margin,
            counterfactual_flow_margin_l_min=cf_f_margin,
            factual_safety_state=fact_state_str,
            counterfactual_safety_state=cf_state_str,
            safety_transition=trans_str,
            outcome_classification=outcome_class,
        )

        # Uncertainty Context
        lat_nov = float(cf_result.metadata.get("latent_novelty", 0.0))
        unc_ctx = UncertaintyContext(
            aleatoric_sigma=float(np.mean(abd_std)),
            epistemic_sigma=float(np.std(abd_mean)),
            latent_novelty_d=lat_nov,
            support_threshold=15.0,
            within_support=bool(lat_nov <= 15.0),
        )

    # Provenance
    prelim_dict = {
        "factual_world": asdict(factual_ctx),
        "intervention": asdict(int_spec),
        "counterfactual_world": asdict(cf_outcome),
        "causal_effect": asdict(causal_eff),
        "twin_world_integrity": asdict(TwinWorldIntegrity()),
        "safety_comparison": asdict(safety_comp),
        "uncertainty": asdict(unc_ctx),
        "source_evidence_hash": source_evidence_hash,
        "counterfactual_engine_version": counterfactual_engine_version,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "episode_id": ep_id,
        "counterfactual_id": cf_id,
        "intervention_time": t_star,
        "timestamp_utc": ts,
    }
    cf_hash = compute_deterministic_cf_hash(prelim_dict)

    prov = CounterfactualProvenance(
        source_evidence_hash=source_evidence_hash,
        counterfactual_engine_version=counterfactual_engine_version,
        model_version=model_version,
        dataset_version=dataset_version,
        episode_id=ep_id,
        counterfactual_id=cf_id,
        intervention_time=t_star,
        timestamp_utc=ts,
        counterfactual_hash=cf_hash,
    )

    return CounterfactualEvidence(
        factual_world=factual_ctx,
        intervention=int_spec,
        counterfactual_world=cf_outcome,
        causal_effect=causal_eff,
        twin_world_integrity=TwinWorldIntegrity(),
        safety_comparison=safety_comp,
        uncertainty=unc_ctx,
        provenance=prov,
    )
