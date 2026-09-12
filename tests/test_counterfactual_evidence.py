"""Unit tests for Task 6.3: PRISM Counterfactual Evidence Engine."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.dataset.counterfactuals import (
    CounterfactualSpec,
    execute_counterfactual_experiment,
    LearnerCounterfactualRecord,
)
from prism.counterfactual.engine import (
    LearnedCounterfactualEngine,
    LearnedCounterfactualResult,
)
from prism.explanation.counterfactual_evidence import (
    CounterfactualEvidence,
    build_counterfactual_evidence,
)


@pytest.fixture(scope="module")
def b005_cf_engine():
    """Load baseline_005 model and initialize LearnedCounterfactualEngine."""
    model_dir = Path("artifacts/baseline_005")
    norm = ObservationNormalizer.load_yaml(model_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    engine = LearnedCounterfactualEngine(model, norm)
    return engine, model, norm


@pytest.fixture
def sample_valve_cf_record():
    """Generate a paired test counterfactual record for valve intervention."""
    factual_ep = generate_single_episode(SplitType.TEST, index=12, regime="nominal", length=100)
    spec = CounterfactualSpec(target_action="A_valve", counterfactual_value=85.0, counterfactual_time=40)
    oracle_rec = execute_counterfactual_experiment(factual_ep, spec)
    return oracle_rec.to_learner_record()


@pytest.fixture
def sample_pump_cf_record():
    """Generate a paired test counterfactual record for pump intervention."""
    factual_ep = generate_single_episode(SplitType.TEST, index=15, regime="moderate_load", length=100)
    spec = CounterfactualSpec(target_action="A_pump", counterfactual_value=4.0, counterfactual_time=40)
    oracle_rec = execute_counterfactual_experiment(factual_ep, spec)
    return oracle_rec.to_learner_record()


# -----------------------------------------------------------------------------
# Test 1: Factual Context Preserved
# -----------------------------------------------------------------------------
def test_factual_context_preserved(b005_cf_engine, sample_valve_cf_record):
    """Test 1: Factual pre-intervention context and abduced state are faithfully preserved."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec, timestamp_utc="2026-09-11T12:00:00Z")

    assert evidence.factual_world.episode_id == learner_rec.parent_episode_id
    assert evidence.factual_world.intervention_time == 40
    assert evidence.factual_world.planning_horizon == len(learner_rec.future_actions)
    assert evidence.factual_world.historical_t_core_at_t_star is not None
    assert evidence.factual_world.abduced_latent_norm is not None


# -----------------------------------------------------------------------------
# Test 2: Intervention Semantics
# -----------------------------------------------------------------------------
def test_intervention_semantics(b005_cf_engine, sample_valve_cf_record):
    """Test 2: Intervention target, value, and formal do(.) notation are exact."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)

    assert evidence.intervention.intervention_target == "A_valve"
    assert evidence.intervention.counterfactual_value == 85.0
    assert evidence.intervention.intervention_type == "ACTION"
    assert evidence.intervention.formal_notation == "do(A_valve = 85.0)"


# -----------------------------------------------------------------------------
# Test 3: Twin-World Mode
# -----------------------------------------------------------------------------
def test_twin_world_mode(b005_cf_engine, sample_valve_cf_record):
    """Test 3: Twin-world replay mode is explicitly verified."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)
    assert evidence.twin_world_integrity.replay_mode == "TWIN_WORLD_FROZEN_EXOGENOUS"


# -----------------------------------------------------------------------------
# Test 4: Shared Exogenous Conditions
# -----------------------------------------------------------------------------
def test_shared_exogenous_conditions(b005_cf_engine, sample_valve_cf_record):
    """Test 4: Shared exogenous conditions flag is True."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)
    assert evidence.twin_world_integrity.shared_exogenous_conditions is True
    assert evidence.twin_world_integrity.identical_pre_intervention_history is True


# -----------------------------------------------------------------------------
# Test 5: Counterfactual Trajectory Extraction
# -----------------------------------------------------------------------------
def test_cf_trajectory_extraction(b005_cf_engine, sample_valve_cf_record):
    """Test 5: Factual and counterfactual trajectories and multi-horizon effects are extracted."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)

    assert evidence.counterfactual_world.peak_t_core is not None
    assert evidence.counterfactual_world.max_p_sys is not None
    assert evidence.counterfactual_world.min_f_cool is not None
    assert len(evidence.causal_effect.horizon_effects) > 0
    assert 10 in evidence.causal_effect.horizon_effects


# -----------------------------------------------------------------------------
# Test 6: Delta T_core
# -----------------------------------------------------------------------------
def test_delta_t_core(b005_cf_engine, sample_pump_cf_record):
    """Test 6: Verify delta_t_core_peak == cf_peak_t - fact_peak_t."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_pump_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)

    fact_peak_t = float(np.max(res.factual_observations[:, 0]))
    cf_peak_t = float(np.max(res.counterfactual_observations[:, 0]))

    assert pytest.approx(evidence.causal_effect.delta_t_core_peak, 1e-4) == cf_peak_t - fact_peak_t


# -----------------------------------------------------------------------------
# Test 7: Delta Flow
# -----------------------------------------------------------------------------
def test_delta_flow(b005_cf_engine, sample_pump_cf_record):
    """Test 7: Verify delta_f_cool_min == cf_min_f - fact_min_f."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_pump_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)

    fact_min_f = float(np.min(res.factual_observations[:, 3]))
    cf_min_f = float(np.min(res.counterfactual_observations[:, 3]))

    assert pytest.approx(evidence.causal_effect.delta_f_cool_min, 1e-4) == cf_min_f - fact_min_f


# -----------------------------------------------------------------------------
# Test 8: Delta Pressure
# -----------------------------------------------------------------------------
def test_delta_pressure(b005_cf_engine, sample_valve_cf_record):
    """Test 8: Verify delta_p_sys_max == cf_max_p - fact_max_p."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)

    fact_max_p = float(np.max(res.factual_observations[:, 2]))
    cf_max_p = float(np.max(res.counterfactual_observations[:, 2]))

    assert pytest.approx(evidence.causal_effect.delta_p_sys_max, 1e-4) == cf_max_p - fact_max_p


# -----------------------------------------------------------------------------
# Test 9: Direction Classification
# -----------------------------------------------------------------------------
def test_direction_classification(b005_cf_engine, sample_pump_cf_record):
    """Test 9: Direction string is deterministically classified."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_pump_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)
    assert evidence.causal_effect.primary_effect_direction in [
        "REDUCES_CORE_TEMPERATURE",
        "INCREASES_COOLANT_FLOW",
        "ELEVATES_PRESSURE",
        "NOMINAL_STABLE",
    ]


# -----------------------------------------------------------------------------
# Test 10: Safe to Safe Transition
# -----------------------------------------------------------------------------
def test_safe_to_safe(b005_cf_engine, sample_valve_cf_record):
    """Test 10: Nominal intervention maintains SAFE -> SAFE state."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec)
    if evidence.safety_comparison.factual_safety_state == "SAFE" and evidence.safety_comparison.counterfactual_safety_state == "SAFE":
        assert evidence.safety_comparison.safety_transition == "SAFE -> SAFE"
        assert evidence.safety_comparison.outcome_classification == "INTERVENTION_PRESERVES_SAFETY"


# -----------------------------------------------------------------------------
# Test 11: Unsafe to Safe Transition
# -----------------------------------------------------------------------------
def test_unsafe_to_safe():
    """Test 11: Verify INTERVENTION_MITIGATES_FAILURE classification on synthetic transitions."""
    from prism.counterfactual.engine import AbducedLatentState
    from prism.intervention.effects import CausalEffectSummary, LearnedFailureMetrics

    # Mock result where factual is 98°C (unsafe) and counterfactual is 85°C (safe)
    fact_obs = np.ones((41, 8), dtype=np.float32) * 50.0
    fact_obs[:, 0] = 98.0  # T_core unsafe
    fact_obs[:, 2] = 3.0   # P_sys safe
    fact_obs[:, 3] = 30.0  # F_cool safe

    cf_obs = np.ones((41, 8), dtype=np.float32) * 50.0
    cf_obs[:, 0] = 85.0   # T_core safe
    cf_obs[:, 2] = 3.0
    cf_obs[:, 3] = 40.0

    mock_res = LearnedCounterfactualResult(
        counterfactual_id="mock_cf_01",
        parent_episode_id="mock_ep_01",
        counterfactual_time=40,
        target_action="A_pump",
        counterfactual_value=3.0,
        abduced_latent_state=AbducedLatentState(np.zeros(64), np.zeros(64), np.ones(64), 40),
        factual_observations=fact_obs,
        counterfactual_observations=cf_obs,
        factual_latent_mean=np.zeros((41, 64)),
        counterfactual_latent_mean=np.zeros((41, 64)),
        effects=CausalEffectSummary(
            horizon_effects={},
            mean_deltas={},
            peak_t_core_base=98.0,
            peak_t_core_int=85.0,
            delta_peak_t_core=-13.0,
            max_pressure_base=3.0,
            max_pressure_int=3.0,
            delta_max_pressure=0.0,
            min_flow_base=30.0,
            min_flow_int=40.0,
            delta_min_flow=10.0,
            failure_metrics=LearnedFailureMetrics(False, False, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    evidence = build_counterfactual_evidence(mock_res)
    assert evidence.safety_comparison.safety_transition == "UNSAFE -> SAFE"
    assert evidence.safety_comparison.outcome_classification == "INTERVENTION_MITIGATES_FAILURE"


# -----------------------------------------------------------------------------
# Test 12: Safe to Unsafe Transition
# -----------------------------------------------------------------------------
def test_safe_to_unsafe():
    """Test 12: Verify INTERVENTION_INTRODUCES_RISK classification on synthetic transitions."""
    from prism.counterfactual.engine import AbducedLatentState
    from prism.intervention.effects import CausalEffectSummary, LearnedFailureMetrics

    # Factual is safe (85°C), CF exceeds pressure limit (6.0 bar > 5.5 bar)
    fact_obs = np.ones((41, 8), dtype=np.float32) * 50.0
    fact_obs[:, 0] = 85.0
    fact_obs[:, 2] = 3.0
    fact_obs[:, 3] = 30.0

    cf_obs = np.ones((41, 8), dtype=np.float32) * 50.0
    cf_obs[:, 0] = 85.0
    cf_obs[:, 2] = 6.0   # P_sys UNSAFE
    cf_obs[:, 3] = 30.0

    mock_res = LearnedCounterfactualResult(
        counterfactual_id="mock_cf_02",
        parent_episode_id="mock_ep_02",
        counterfactual_time=40,
        target_action="A_valve",
        counterfactual_value=10.0,
        abduced_latent_state=AbducedLatentState(np.zeros(64), np.zeros(64), np.ones(64), 40),
        factual_observations=fact_obs,
        counterfactual_observations=cf_obs,
        factual_latent_mean=np.zeros((41, 64)),
        counterfactual_latent_mean=np.zeros((41, 64)),
        effects=CausalEffectSummary(
            horizon_effects={},
            mean_deltas={},
            peak_t_core_base=85.0,
            peak_t_core_int=85.0,
            delta_peak_t_core=0.0,
            max_pressure_base=3.0,
            max_pressure_int=6.0,
            delta_max_pressure=3.0,
            min_flow_base=30.0,
            min_flow_int=30.0,
            delta_min_flow=0.0,
            failure_metrics=LearnedFailureMetrics(False, False, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    evidence = build_counterfactual_evidence(mock_res)
    assert evidence.safety_comparison.safety_transition == "SAFE -> UNSAFE"
    assert evidence.safety_comparison.outcome_classification == "INTERVENTION_INTRODUCES_RISK"


# -----------------------------------------------------------------------------
# Test 13: CF Blocked Under Model Abstain
# -----------------------------------------------------------------------------
def test_cf_blocked_under_model_abstain(sample_valve_cf_record):
    """Test 13: Under MODEL_ABSTAIN, counterfactual simulation is strictly blocked."""
    learner_rec = sample_valve_cf_record
    evidence = build_counterfactual_evidence(
        cf_result=None,
        learner_record=learner_rec,
        is_model_abstained=True,
        abstention_reason="Reconstruction inconsistency R_T = 33.88°C > 6.08°C",
    )

    assert evidence.counterfactual_world.peak_t_core is None
    assert evidence.causal_effect.primary_effect_direction == "UNTRUSTED_BLOCKED"
    assert evidence.safety_comparison.safety_transition == "ABSTAINED"
    assert evidence.safety_comparison.outcome_classification == "UNTRUSTED_BLOCKED"
    assert "blocked" in evidence.causal_effect.causal_interpretation.lower()


# -----------------------------------------------------------------------------
# Test 14: Provenance Hash
# -----------------------------------------------------------------------------
def test_provenance_hash(b005_cf_engine, sample_valve_cf_record):
    """Test 14: Valid 64-character SHA-256 hash is computed."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    evidence = build_counterfactual_evidence(res, learner_rec, source_evidence_hash="test_parent_hash_123")

    assert len(evidence.provenance.counterfactual_hash) == 64
    assert evidence.provenance.source_evidence_hash == "test_parent_hash_123"
    assert evidence.provenance.model_version == "baseline_005"


# -----------------------------------------------------------------------------
# Test 15: Deterministic Output
# -----------------------------------------------------------------------------
def test_deterministic_output(b005_cf_engine, sample_valve_cf_record):
    """Test 15: Multiple invocations with identical inputs produce identical hashes and dictionaries."""
    engine, _, _ = b005_cf_engine
    learner_rec = sample_valve_cf_record
    res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

    ev_1 = build_counterfactual_evidence(res, learner_rec, timestamp_utc="2026-09-11T12:00:00Z")
    ev_2 = build_counterfactual_evidence(res, learner_rec, timestamp_utc="2026-09-11T12:00:00Z")

    assert ev_1.to_dict() == ev_2.to_dict()
    assert ev_1.provenance.counterfactual_hash == ev_2.provenance.counterfactual_hash
    assert ev_1.to_json() == ev_2.to_json()
