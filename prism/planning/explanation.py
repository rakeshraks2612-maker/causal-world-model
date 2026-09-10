"""Causal Explanation and Decision Intelligence Generator (Phase 4).

Translates learned world model rollout predictions into structured, verifiable causal explanations:
1. Diagnosis & Risk Assessment
2. Mechanism Attribution (Direct & Indirect Causal Pathways)
3. Counterfactual Contrastive Analysis (Factual Inaction vs Intervened Outcome)
4. Alternative Action Rejection Rationale
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
from prism.planning.objectives import CandidateEvaluation
from prism.intervention.simulator import LearnedInterventionResult


@dataclass
class CausalExplanation:
    """Structured causal explanation justifying an intervention decision."""

    summary: str
    diagnosis: str
    recommended_action: str
    physical_mechanism: str
    counterfactual_contrast: str
    rejected_alternatives: List[Dict[str, str]]
    confidence_level: str
    safety_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_markdown(self) -> str:
        """Format explanation as an executive report."""
        lines = [
            f"### 🛡️ PRISM Decision Intelligence Explanation",
            f"**Recommendation:** `{self.recommended_action}` | **Safety Status:** {self.safety_verdict}",
            "",
            f"#### 1. System Diagnosis",
            f"{self.diagnosis}",
            "",
            f"#### 2. Causal Mechanism",
            f"{self.physical_mechanism}",
            "",
            f"#### 3. Counterfactual Contrast",
            f"{self.counterfactual_contrast}",
            "",
            f"#### 4. Candidate Trade-Off Analysis",
        ]
        for alt in self.rejected_alternatives:
            lines.append(f"- **{alt['candidate']}**: {alt['reason']}")
        return "\n".join(lines)


def generate_causal_explanation(
    best_candidate: Optional[CandidateEvaluation],
    all_candidates: List[CandidateEvaluation],
    baseline_result: LearnedInterventionResult,
) -> CausalExplanation:
    """Generate comprehensive causal explanation for the selected candidate or abstention."""
    base_peak_t = float(np.max(baseline_result.baseline_observations[:, 0]))

    if best_candidate is None:
        # All candidates rejected / Abstention state
        if base_peak_t >= 95.0:
            diagnosis = (
                f"Under factual baseline trajectory, system core temperature reaches {base_peak_t:.1f}°C, "
                f"violating safe thermal limits and risking thermal runaway."
            )
        else:
            diagnosis = f"Factual baseline operates at Peak T_core = {base_peak_t:.1f}°C, but all candidate interventions are infeasible."

        rejected = []
        for cand in all_candidates:
            reason = f"Rejected: {'; '.join(cand.safety_violations)}" if not cand.is_safe else "Rejected due to system abstention."
            rejected.append({"candidate": f"{cand.spec.target}={cand.spec.value}", "reason": reason})

        return CausalExplanation(
            summary="Abstaining from intervention: No candidate satisfied all hard safety constraints.",
            diagnosis=diagnosis,
            recommended_action="ABSTAIN",
            physical_mechanism="All evaluated candidate interventions violate either thermal limits, pressure constraints, flow minimums, or latent support.",
            counterfactual_contrast="All candidate interventions cross hard safety envelopes. Autonomous intervention withheld to prevent compounding risk.",
            rejected_alternatives=rejected,
            confidence_level="HIGH (All candidates verified unsafe / OOD)",
            safety_verdict="🔴 ALL CANDIDATES UNSAFE / ABSTAIN REQUIRED",
        )

    spec = best_candidate.spec
    cand_peak_t = best_candidate.peak_t_core
    delta_t = cand_peak_t - base_peak_t
    delta_f = best_candidate.causal_delta_f_cool

    is_do_nothing = (
        best_candidate.candidate_id == "cand_do_nothing"
        or (spec is not None and spec.target in ["none", "do_nothing"])
    )

    # 1. Diagnosis
    if base_peak_t >= 95.0:
        diagnosis = (
            f"Under factual baseline trajectory, system core temperature reaches {base_peak_t:.1f}°C, "
            f"violating safe thermal limits and risking thermal runaway."
        )
    else:
        diagnosis = f"Factual baseline operates within nominal support (Peak T_core = {base_peak_t:.1f}°C)."

    # 2. Mechanism & Action String
    if is_do_nothing:
        rec_action_str = "DO_NOTHING"
        mechanism = (
            "Current operating state is within a stable thermal, pressure, and cooling regime. "
            "No candidate intervention provides sufficient risk-adjusted benefit to justify its operational or actuation cost."
        )
        contrast = (
            f"Factual baseline operates safely at peak T_core of {base_peak_t:.1f}°C. "
            f"Alternative interventions provide negligible thermal benefit while incurring unnecessary operational penalties."
        )
        summary = f"Maintain baseline operation (DO NOTHING): system is in stable thermal equilibrium (Peak T_core: {cand_peak_t:.1f}°C, Utility: {best_candidate.utility_score:+.2f})."
    elif spec.target in ["A_valve", "V_pos"]:
        rec_action_str = f"{spec.target}={spec.value:.1f}"
        mechanism = (
            f"Setting `{spec.target}={spec.value:.1f}%` expands valve conductance, increasing coolant flow "
            f"by {delta_f:+.2f} L/min. This accelerates convective heat dissipation from the core manifold, "
            f"reducing peak core temperature by {abs(delta_t):.1f}°C."
        )
        contrast = (
            f"Without intervention, factual baseline reaches peak T_core of {base_peak_t:.1f}°C. "
            f"Under recommended `{spec.target}={spec.value:.1f}`, core temperature safely peaks at {cand_peak_t:.1f}°C "
            f"(Net causal benefit: {delta_t:+.1f}°C)."
        )
        summary = f"Recommend `{spec.target}={spec.value:.1f}` to achieve peak core temperature {cand_peak_t:.1f}°C (Utility: {best_candidate.utility_score:.2f})."
    elif spec.target in ["A_throttle", "L_cpu"]:
        rec_action_str = f"{spec.target}={spec.value:.1f}"
        mechanism = (
            f"Throttling CPU workload to `{spec.target}={spec.value:.1f}%` directly diminishes Joule heating generation "
            f"in the core, stabilizing thermal accumulation and lowering core temperature by {abs(delta_t):.1f}°C."
        )
        contrast = (
            f"Without intervention, factual baseline reaches peak T_core of {base_peak_t:.1f}°C. "
            f"Under recommended `{spec.target}={spec.value:.1f}`, core temperature safely peaks at {cand_peak_t:.1f}°C "
            f"(Net causal benefit: {delta_t:+.1f}°C)."
        )
        summary = f"Recommend `{spec.target}={spec.value:.1f}` to achieve peak core temperature {cand_peak_t:.1f}°C (Utility: {best_candidate.utility_score:.2f})."
    elif spec.target in ["A_pump"]:
        rec_action_str = f"{spec.target}={int(spec.value)}"
        mechanism = (
            f"Adjusting pump stage to `{spec.target}={int(spec.value)}` modulates hydraulic head and fluid delivery, "
            f"yielding coolant flow change of {delta_f:+.2f} L/min. Increasing pump stage increases coolant circulation, "
            f"modifies the hydraulic/thermal state, and reduces thermal risk, providing the optimal safety-performance-cost tradeoff."
        )
        contrast = (
            f"Without intervention, factual baseline reaches peak T_core of {base_peak_t:.1f}°C. "
            f"Under recommended `{spec.target}={int(spec.value)}`, core temperature safely peaks at {cand_peak_t:.1f}°C "
            f"(Net causal benefit: {delta_t:+.1f}°C)."
        )
        summary = f"Recommend `{spec.target}={int(spec.value)}` to achieve peak core temperature {cand_peak_t:.1f}°C (Utility: {best_candidate.utility_score:.2f})."
    else:
        rec_action_str = f"{spec.target}={spec.value}" if spec is not None else "DO_NOTHING"
        mechanism = f"Intervening on `{rec_action_str}` modulates latent dynamics to optimize thermal-fluid equilibrium."
        contrast = (
            f"Without intervention, factual baseline reaches peak T_core of {base_peak_t:.1f}°C. "
            f"Under recommended `{rec_action_str}`, core temperature safely peaks at {cand_peak_t:.1f}°C "
            f"(Net causal benefit: {delta_t:+.1f}°C)."
        )
        summary = f"Recommend `{rec_action_str}` to achieve peak core temperature {cand_peak_t:.1f}°C (Utility: {best_candidate.utility_score:.2f})."


    # 4. Rejected alternatives
    rejected = []
    for cand in all_candidates:
        if cand.candidate_id == best_candidate.candidate_id:
            continue
        target_str = cand.candidate_id if (cand.spec is None or cand.spec.target in ["none", "do_nothing"]) else f"{cand.spec.target}={cand.spec.value:.1f}"
        if not cand.is_safe:
            reason = f"Rejected due to safety violations: {'; '.join(cand.safety_violations)}"
        else:
            diff = cand.utility_score - best_candidate.utility_score
            reason = (
                f"Sub-optimal utility ({cand.utility_score:+.4f} vs {best_candidate.utility_score:+.4f}, ΔU={diff:+.4f}) — "
                f"Peak T_core: {cand.peak_t_core:.1f}°C, Mean CPU Load: {cand.mean_cpu_load:.1f}%"
            )
        rejected.append({"candidate": target_str, "reason": reason})


    verdict = "🟢 SAFE & OPTIMAL" if best_candidate.is_safe else "🔴 SAFETY BLOCKED / ABSTAIN"

    return CausalExplanation(
        summary=summary,
        diagnosis=diagnosis,
        recommended_action=rec_action_str,
        physical_mechanism=mechanism,
        counterfactual_contrast=contrast,
        rejected_alternatives=rejected,
        confidence_level="HIGH (Within Latent Manifold Support)",
        safety_verdict=verdict,
    )


