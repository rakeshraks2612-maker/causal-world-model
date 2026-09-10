"""Tests for Decision & Intervention Planner Subsystem (Phase 4)."""

from __future__ import annotations
import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.planning.objectives import PlanningObjective, SafetyConstraints, UtilityWeights
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.planning.explanation import CausalExplanation


@pytest.fixture(scope="module")
def planner_system():
    """Load the trained baseline_003 world model and initialize InterventionPlanner."""
    b3_dir = Path("artifacts/baseline_003")
    assert b3_dir.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(b3_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    planner = InterventionPlanner(model, norm)
    return planner, model, norm


def test_intervention_planner_search_and_recommendation(planner_system):
    """Test multi-candidate evaluation, safety filtering, and optimal selection."""
    planner, model, norm = planner_system
    
    ep = generate_single_episode(SplitType.TEST, index=5, regime="moderate_load", length=100)
    t_star = 40
    
    plan_rec = planner.plan_intervention(
        historical_observations=ep.observations[: t_star + 1],
        historical_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention_time=t_star,
    )
    
    assert isinstance(plan_rec, PlanRecommendation)
    assert plan_rec.recommended_candidate is not None
    assert len(plan_rec.all_evaluated_candidates) == 10
    assert isinstance(plan_rec.explanation, CausalExplanation)
    assert plan_rec.planning_time_ms > 0.0
    
    # Check that explanation has all sections
    exp_md = plan_rec.explanation.format_markdown()
    assert "PRISM Decision Intelligence Explanation" in exp_md
    assert "System Diagnosis" in exp_md
    assert "Causal Mechanism" in exp_md
    assert "Counterfactual Contrast" in exp_md


def test_intervention_planner_abstention_on_extreme_constraints(planner_system):
    """Test that planner triggers abstention when safety constraints are impossible to satisfy."""
    planner, model, norm = planner_system
    
    # Impossible safety constraint: T_core must be < 30°C in a running server
    impossible_constraints = SafetyConstraints(t_core_max=30.0)
    obj = PlanningObjective(constraints=impossible_constraints)
    strict_planner = InterventionPlanner(model, norm, objective=obj)
    
    ep = generate_single_episode(SplitType.TEST, index=5, regime="nominal", length=100)
    t_star = 40
    
    plan_rec = strict_planner.plan_intervention(
        historical_observations=ep.observations[: t_star + 1],
        historical_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention_time=t_star,
    )
    
    assert plan_rec.should_abstain is True
    assert plan_rec.abstention_reason is not None
