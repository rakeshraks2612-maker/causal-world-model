"""Decision & Intervention Planner Subsystem (Phase 4).

Generates Pareto-optimal, safety-constrained causal intervention plans:
1. Evaluates candidate interventions across multi-horizon rollouts.
2. Applies strict safety boundary filtering (T_core, P_sys, F_cool limits).
3. Applies epistemic uncertainty and OOD latent novelty gating.
4. Selects optimal intervention maximizing multi-objective utility.
5. Emits structured, verifiable causal explanations.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch import Tensor

from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.intervention.spec import InterventionSpec, InterventionType
from prism.intervention.simulator import LearnedInterventionSimulator, LearnedInterventionResult
from prism.uncertainty.estimator import EpistemicUncertaintyEstimator, LatentManifoldDensityEstimator
from prism.planning.objectives import (
    PlanningObjective,
    SafetyConstraints,
    UtilityWeights,
    CandidateEvaluation,
)
from prism.planning.explanation import CausalExplanation, generate_causal_explanation


DEFAULT_CANDIDATE_GRID: List[Tuple[str, float, InterventionType]] = [
    ("A_valve", 50.0, InterventionType.ACTION_CONTROL),
    ("A_valve", 70.0, InterventionType.ACTION_CONTROL),
    ("A_valve", 85.0, InterventionType.ACTION_CONTROL),
    ("A_valve", 100.0, InterventionType.ACTION_CONTROL),
    ("A_throttle", 20.0, InterventionType.ACTION_CONTROL),
    ("A_throttle", 50.0, InterventionType.ACTION_CONTROL),
    ("A_throttle", 80.0, InterventionType.ACTION_CONTROL),
    ("A_throttle", 100.0, InterventionType.ACTION_CONTROL),
    ("A_pump", 2.0, InterventionType.ACTION_CONTROL),
    ("A_pump", 4.0, InterventionType.ACTION_CONTROL),
]


@dataclass
class PlanRecommendation:
    """Complete output of the PRISM Intervention Planner."""

    recommended_candidate: Optional[CandidateEvaluation]
    should_abstain: bool
    abstention_reason: Optional[str]
    all_evaluated_candidates: List[CandidateEvaluation]
    explanation: CausalExplanation
    baseline_simulation: LearnedInterventionResult
    planning_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_candidate": self.recommended_candidate.to_dict() if self.recommended_candidate else None,
            "should_abstain": self.should_abstain,
            "abstention_reason": self.abstention_reason,
            "all_evaluated_candidates": [c.to_dict() for c in self.all_evaluated_candidates],
            "explanation": self.explanation.to_dict(),
            "planning_time_ms": self.planning_time_ms,
        }


class InterventionPlanner:
    """Safety-aware, causal decision intelligence planner."""

    def __init__(
        self,
        world_model: CausalWorldModel,
        normalizer: ObservationNormalizer,
        objective: Optional[PlanningObjective] = None,
        uncertainty_estimator: Optional[EpistemicUncertaintyEstimator] = None,
        candidate_grid: Optional[List[Tuple[str, float, InterventionType]]] = None,
    ) -> None:
        self.world_model = world_model
        self.normalizer = normalizer
        self.simulator = LearnedInterventionSimulator(world_model, normalizer)
        self.objective = objective or PlanningObjective()
        self.uncertainty_estimator = uncertainty_estimator or EpistemicUncertaintyEstimator(world_model, normalizer)
        self.candidate_grid = candidate_grid or DEFAULT_CANDIDATE_GRID

    def plan_intervention(
        self,
        historical_observations: np.ndarray | Tensor,
        historical_actions: np.ndarray | Tensor,
        future_actions: np.ndarray | Tensor,
        custom_candidates: Optional[List[InterventionSpec]] = None,
        intervention_time: Optional[int] = None,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
    ) -> PlanRecommendation:
        """Find optimal safety-constrained intervention given historical context."""
        import time
        start_t = time.perf_counter()

        t_pre = historical_observations.shape[0] if isinstance(historical_observations, np.ndarray) else historical_observations.shape[1]
        t_star = intervention_time if intervention_time is not None else (t_pre - 1)

        # 1. Simulate Factual Baseline (No-op intervention)
        baseline_sim = self.simulator.simulate(
            pre_observations=historical_observations,
            pre_actions=historical_actions,
            future_actions=future_actions,
            intervention=None,
            intervention_time=t_star,
            horizons=horizons,
            deterministic=True,
        )

        # 2. Build candidate specs
        if custom_candidates is not None:
            candidates = custom_candidates
        else:
            candidates = [
                InterventionSpec(
                    target=tgt,
                    value=val,
                    intervention_time=t_star,
                    intervention_type=itype,
                )
                for tgt, val, itype in self.candidate_grid
            ]

        # 3. Simulate and evaluate all candidate interventions
        evaluations: List[CandidateEvaluation] = []

        for cand_spec in candidates:
            sim_res = self.simulator.simulate(
                pre_observations=historical_observations,
                pre_actions=historical_actions,
                future_actions=future_actions,
                intervention=cand_spec,
                intervention_time=t_star,
                horizons=horizons,
                deterministic=True,
            )

            # Compute latent novelty of candidate rollout
            novelty = float(np.mean(
                self.uncertainty_estimator.density_estimator.compute_mahalanobis_distance(
                    sim_res.intervened_latent_mean
                )
            ))

            cand_eval = self.objective.evaluate_candidate(
                spec=cand_spec,
                sim_result=sim_res,
                latent_novelty=novelty,
            )
            evaluations.append(cand_eval)

        # 4. Filter safe candidates and rank by utility score
        safe_candidates = [c for c in evaluations if c.is_safe]

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if safe_candidates:
            # Sort by utility descending
            safe_candidates.sort(key=lambda c: c.utility_score, reverse=True)
            best_cand = safe_candidates[0]

            explanation = generate_causal_explanation(
                best_candidate=best_cand,
                all_candidates=evaluations,
                baseline_result=baseline_sim,
            )

            return PlanRecommendation(
                recommended_candidate=best_cand,
                should_abstain=False,
                abstention_reason=None,
                all_evaluated_candidates=evaluations,
                explanation=explanation,
                baseline_simulation=baseline_sim,
                planning_time_ms=elapsed_ms,
            )
        else:
            # All candidates violated safety or OOD constraints -> Abstain
            evaluations.sort(key=lambda c: c.utility_score, reverse=True)
            fallback_cand = evaluations[0]
            reason = "No candidate intervention satisfied all physical safety and latent support constraints."

            explanation = generate_causal_explanation(
                best_candidate=fallback_cand,
                all_candidates=evaluations,
                baseline_result=baseline_sim,
            )

            return PlanRecommendation(
                recommended_candidate=fallback_cand,
                should_abstain=True,
                abstention_reason=reason,
                all_evaluated_candidates=evaluations,
                explanation=explanation,
                baseline_simulation=baseline_sim,
                planning_time_ms=elapsed_ms,
            )
