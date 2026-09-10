"""Integration Tests for Learned Intervention Simulator with Frozen Baseline Model."""

from pathlib import Path
import numpy as np
import torch
import pytest

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.intervention.spec import state_clamp
from prism.dataset.interventions import LearnerInterventionRecord


@pytest.fixture(scope="module")
def loaded_simulator():
    """Load the canonical frozen baseline_002 model and normalizer."""
    art_dir = Path("artifacts/baseline_002")
    if not (art_dir / "best.pt").exists() or not (art_dir / "normalization.yaml").exists():
        pytest.skip("baseline_002 checkpoint not found")

    normalizer = ObservationNormalizer.load_yaml(art_dir / "normalization.yaml")
    config = WorldModelConfig.from_yaml(art_dir / "config.yaml")
    model = CausalWorldModel(config)

    checkpoint = torch.load(art_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sim = LearnedInterventionSimulator(model, normalizer)
    return sim


def test_learned_intervention_simulator_shapes_and_execution(loaded_simulator):
    """Test full simulation run producing valid shapes and causal metrics."""
    sim = loaded_simulator

    # Create dummy 21-step historical context (t* = 20)
    pre_obs = np.ones((21, 8), dtype=np.float32) * 50.0
    pre_act = np.ones((21, 4), dtype=np.float32) * 50.0
    pre_act[:, 2] = 2 # A_pump = 2
    pre_act[:, 3] = 0 # A_flush = 0

    # 40-step future actions
    fut_act = np.copy(pre_act[-1:, :]).repeat(40, axis=0)

    spec = state_clamp("V_pos", 85.0, intervention_time=20)

    # 1. Deterministic mode
    res_det = sim.simulate(
        pre_observations=pre_obs,
        pre_actions=pre_act,
        future_actions=fut_act,
        intervention=spec,
        horizons=(1, 5, 10, 20, 40),
        deterministic=True,
    )

    assert res_det.baseline_observations.shape == (41, 8)
    assert res_det.intervened_observations.shape == (41, 8)
    assert res_det.inferred_z_t_star.shape == (64,)
    assert 10 in res_det.effects.horizon_effects

    # 2. Monte Carlo mode
    res_mc = sim.simulate(
        pre_observations=pre_obs,
        pre_actions=pre_act,
        future_actions=fut_act,
        intervention=spec,
        horizons=(1, 5, 10, 20, 40),
        deterministic=False,
        num_particles=10,
    )
    assert res_mc.baseline_observations.shape == (41, 8)
    assert res_mc.intervened_observations.shape == (41, 8)


def test_simulate_from_learner_record(loaded_simulator):
    """Test simulation directly from a LearnerInterventionRecord file."""
    sim = loaded_simulator
    learner_files = list(Path("data/pilot/learner/intervention").glob("*.npz"))
    if not learner_files:
        pytest.skip("No learner intervention files found")

    rec = LearnerInterventionRecord.load_npz(learner_files[0])
    res = sim.simulate_from_learner_record(rec)

    assert res.target == rec.target
    assert res.value == rec.value
    assert res.intervention_time == rec.intervention_time
    assert len(res.effects.horizon_effects) > 0
