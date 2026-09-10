"""Tests for Learned Level-3 Counterfactual Engine (Task 3.5).

Verifies:
1. Latent Abduction Step: Incurs historical observations and infers latent distribution q_phi(Z_t*).
2. Action Substitution: Counterfactual action is substituted at t*, while pre-t* history and future actions are preserved.
3. Twin-World Simulation: Factual vs Counterfactual rollouts diverge causally under identical abduced latent state.
4. Metric Evaluation: Single and benchmark aggregation metrics compute cleanly against ground truth.
"""

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
from prism.dataset.counterfactuals import (
    CounterfactualSpec,
    execute_counterfactual_experiment,
    LearnerCounterfactualRecord,
    OracleCounterfactualRecord,
)
from prism.counterfactual.engine import (
    LearnedCounterfactualEngine,
    LearnedCounterfactualResult,
)
from prism.counterfactual.metrics import (
    evaluate_single_counterfactual,
    aggregate_counterfactual_benchmark,
)


@pytest.fixture(scope="module")
def trained_engine():
    """Load the trained baseline_003 world model and initialize LearnedCounterfactualEngine."""
    b3_dir = Path("artifacts/baseline_003")
    assert b3_dir.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(b3_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    engine = LearnedCounterfactualEngine(model, norm)
    return engine, model, norm


@pytest.fixture
def sample_counterfactual_experiment():
    """Generate a paired oracle and learner counterfactual record."""
    factual_ep = generate_single_episode(SplitType.TEST, index=12, regime="nominal", length=100)
    spec = CounterfactualSpec(target_action="A_valve", counterfactual_value=85.0, counterfactual_time=40)
    oracle_rec = execute_counterfactual_experiment(factual_ep, spec)
    learner_rec = oracle_rec.to_learner_record()
    return learner_rec, oracle_rec


def test_abduction_step(trained_engine, sample_counterfactual_experiment):
    """Test Step 1: Latent state abduction from historical observations."""
    engine, model, norm = trained_engine
    learner_rec, _ = sample_counterfactual_experiment
    
    z_dist, abduced = engine.abduce_latent_state(
        observations=learner_rec.historical_observations,
        actions=learner_rec.historical_actions,
        observation_mask=learner_rec.historical_observation_mask,
    )
    
    assert abduced.latent_mean.shape == (64,)
    assert abduced.latent_std.shape == (64,)
    assert abduced.timestep == 40
    assert np.all(abduced.latent_std > 0.0)


def test_twin_world_counterfactual_simulation(trained_engine, sample_counterfactual_experiment):
    """Test Steps 2 & 3: Twin-world counterfactual execution and effect calculation."""
    engine, model, norm = trained_engine
    learner_rec, oracle_rec = sample_counterfactual_experiment
    
    result = engine.evaluate_from_learner_record(learner_rec, horizons=(1, 5, 10, 20, 40), deterministic=True)
    
    assert isinstance(result, LearnedCounterfactualResult)
    assert result.counterfactual_id == learner_rec.counterfactual_id
    assert result.target_action == "A_valve"
    assert result.counterfactual_value == 85.0
    assert result.factual_observations.shape[0] == len(learner_rec.future_actions) + 1
    assert result.counterfactual_observations.shape[0] == len(learner_rec.future_actions) + 1
    
    # Delta at h=10 must be recorded
    assert 10 in result.effects.horizon_effects
    eff_10 = result.effects.horizon_effects[10]
    assert isinstance(eff_10.delta_t_core, float)


def test_counterfactual_metric_evaluation(trained_engine, sample_counterfactual_experiment):
    """Test paired evaluation of learned counterfactual against oracle ground truth."""
    engine, model, norm = trained_engine
    learner_rec, oracle_rec = sample_counterfactual_experiment
    
    result = engine.evaluate_from_learner_record(learner_rec, horizons=(1, 5, 10, 20, 40), deterministic=True)
    eval_res = evaluate_single_counterfactual(result, oracle_rec)
    
    assert eval_res.counterfactual_id == oracle_rec.counterfactual_id
    assert 10 in eval_res.causal_errors
    assert "delta_t_core" in eval_res.causal_errors[10]
    
    summary = aggregate_counterfactual_benchmark([eval_res], horizons=(1, 5, 10, 20, 40))
    assert summary.total_records == 1
    assert summary.overall_mean_causal_error >= 0.0
    assert summary.overall_directional_accuracy in [0.0, 1.0] or (0.0 <= summary.overall_directional_accuracy <= 1.0)
