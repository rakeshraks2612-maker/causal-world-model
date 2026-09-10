"""Tests for Metrics Calculation, Baseline Models, and Table Formatting."""

import pytest
import numpy as np
import torch
from pathlib import Path

from prism.training.metrics import (
    PersistenceBaseline,
    MLPDynamicsBaseline,
    compute_regression_metrics,
    format_benchmark_table,
    export_metrics_json,
    EvaluationSummary,
)


def test_persistence_baseline_prediction() -> None:
    """Test Persistence baseline: O_hat_{t+1} == O_t."""
    obs = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ])  # [1, 3, 2]
    # Pad to 8 channels
    obs_8 = torch.zeros(1, 3, 8)
    obs_8[:, :, :2] = obs

    baseline = PersistenceBaseline()
    pred = baseline.predict_one_step(obs_8)

    # Pred at t=1 is obs at t=0
    np.testing.assert_array_equal(pred[:, 0].numpy(), obs_8[:, 0].numpy())
    np.testing.assert_array_equal(pred[:, 1].numpy(), obs_8[:, 1].numpy())


def test_mlp_dynamics_baseline_forward() -> None:
    """Test MLP dynamics baseline forward pass."""
    mlp = MLPDynamicsBaseline(obs_dim=8, action_dim=4, hidden_dim=32)
    obs = torch.randn(4, 8)
    act = torch.randn(4, 4)

    next_obs = mlp(obs, act)
    assert next_obs.shape == (4, 8)


def test_compute_regression_metrics() -> None:
    """Test MAE, RMSE, and NLL computation per channel."""
    preds = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])
    targets = torch.tensor([[12.0, 19.0, 30.0, 44.0, 50.0, 60.0, 70.0, 80.0]])
    mask = torch.ones(1, 8)

    summary = compute_regression_metrics(preds, targets, mask, model_name="TestModel")
    assert summary.model_name == "TestModel"
    assert summary.per_channel["T_core"].mae == 2.0
    assert summary.per_channel["T_cool"].mae == 1.0
    assert summary.per_channel["P_sys"].mae == 0.0
    assert summary.per_channel["F_cool"].mae == 4.0
    assert summary.overall_mae > 0.0


def test_benchmark_table_formatting_and_json_export(tmp_path) -> None:
    """Test formatting comparison table and saving machine-readable JSON metrics."""
    s1 = EvaluationSummary(model_name="Persistence", overall_mae=1.5, overall_rmse=2.0, overall_nll=None)
    s2 = EvaluationSummary(model_name="MLP_Dynamics", overall_mae=1.1, overall_rmse=1.6, overall_nll=None)
    s3 = EvaluationSummary(model_name="PRISM_WorldModel", overall_mae=0.8, overall_rmse=1.2, overall_nll=1.4)

    table_str = format_benchmark_table([s1, s2, s3])
    assert "Persistence" in table_str
    assert "MLP_Dynamics" in table_str
    assert "PRISM_WorldModel" in table_str

    json_path = tmp_path / "metrics.json"
    export_metrics_json([s1, s2, s3], json_path, metadata={"seed": 42})
    assert json_path.exists()
