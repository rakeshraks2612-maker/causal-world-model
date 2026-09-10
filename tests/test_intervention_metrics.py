"""Unit Tests for Intervention Evaluation Metrics and Benchmarking."""

from pathlib import Path
import numpy as np
import pytest

from prism.evaluation.intervention_metrics import (
    PairedInterventionEvaluation,
    InterventionBenchmarkSummary,
    evaluate_single_intervention,
    aggregate_intervention_benchmark,
)
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.dataset.interventions import LearnerInterventionRecord
from prism.training.normalization import ObservationNormalizer
from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
import torch


@pytest.fixture(scope="module")
def simulator():
    art_dir = Path("artifacts/baseline_002")
    if not (art_dir / "best.pt").exists():
        pytest.skip("baseline_002 checkpoint not found")

    normalizer = ObservationNormalizer.load_yaml(art_dir / "normalization.yaml")
    config = WorldModelConfig.from_yaml(art_dir / "config.yaml")
    model = CausalWorldModel(config)
    checkpoint = torch.load(art_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LearnedInterventionSimulator(model, normalizer)


def test_evaluate_single_and_aggregate_metrics(simulator):
    """Test evaluate_single_intervention and aggregate_intervention_benchmark."""
    oracle_files = sorted(list(Path("data/pilot/oracle/intervention").glob("*.npz")))
    learner_files = sorted(list(Path("data/pilot/learner/intervention").glob("*.npz")))

    if not oracle_files or not learner_files:
        pytest.skip("No pilot intervention files found")

    evaluations = []
    # Test on first 3 records
    for i in range(min(3, len(oracle_files))):
        orc_file = oracle_files[i]
        learn_file = Path("data/pilot/learner/intervention") / orc_file.name
        learn_rec = LearnerInterventionRecord.load_npz(learn_file)

        res = simulator.simulate_from_learner_record(learn_rec)
        eval_item = evaluate_single_intervention(res, orc_file)

        assert eval_item.intervention_id is not None
        assert 10 in eval_item.causal_errors
        evaluations.append(eval_item)

    summary = aggregate_intervention_benchmark(evaluations, horizons=(1, 5, 10, 20, 40))
    assert summary.total_records == len(evaluations)
    assert summary.overall_mean_causal_error >= 0.0
    assert 0.0 <= summary.overall_directional_accuracy <= 1.0

    table_str = summary.format_table(horizon=10)
    assert "T_core" in table_str
    assert "F_cool" in table_str

    dict_repr = summary.to_dict()
    assert "mean_causal_error_by_horizon" in dict_repr
    assert "breakdown_by_target" in dict_repr
