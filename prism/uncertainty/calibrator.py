"""Uncertainty Calibration and Coverage Metrics Subsystem.

Computes:
1. Prediction Interval Coverage Probability (PICP): empirical % of observations falling in 90% CI.
2. Mean Prediction Interval Width (MPIW): sharpness of uncertainty bounds.
3. Expected Calibration Error (ECE) for probabilistic forecasts.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import Tensor


@dataclass
class CalibrationMetrics:
    """Summary of uncertainty calibration and sharpness metrics."""

    picp_90: float                      # Empirical coverage of nominal 90% interval (ideal = 0.90)
    mpiw_90: float                      # Mean width of 90% interval
    coverage_error: float               # |PICP - 0.90|
    expected_calibration_error: float   # ECE
    coverage_by_variable: Dict[str, float]
    width_by_variable: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_prediction_interval_coverage(
    true_observations: np.ndarray | Tensor,
    lower_bounds: np.ndarray | Tensor,
    upper_bounds: np.ndarray | Tensor,
    variable_names: Optional[List[str]] = None,
) -> CalibrationMetrics:
    """Compute empirical coverage (PICP) and sharpness (MPIW) of predicted confidence intervals."""
    if isinstance(true_observations, Tensor):
        y_true = true_observations.detach().cpu().numpy()
    else:
        y_true = np.asarray(true_observations)

    if isinstance(lower_bounds, Tensor):
        y_low = lower_bounds.detach().cpu().numpy()
    else:
        y_low = np.asarray(lower_bounds)

    if isinstance(upper_bounds, Tensor):
        y_high = upper_bounds.detach().cpu().numpy()
    else:
        y_high = np.asarray(upper_bounds)

    # In-bounds indicator: y_low <= y_true <= y_high
    in_bounds = (y_true >= y_low) & (y_true <= y_high)  # [..., D]
    widths = y_high - y_low                             # [..., D]

    d_vars = y_true.shape[-1]
    var_names = variable_names or [f"var_{i}" for i in range(d_vars)]

    cov_by_var: Dict[str, float] = {}
    width_by_var: Dict[str, float] = {}

    for i, name in enumerate(var_names):
        cov_by_var[name] = float(np.mean(in_bounds[..., i]))
        width_by_var[name] = float(np.mean(widths[..., i]))

    overall_picp = float(np.mean(in_bounds))
    overall_mpiw = float(np.mean(widths))
    cov_err = abs(overall_picp - 0.90)

    # Simple ECE approximation across quantile intervals
    ece = float(np.mean([abs(cov_by_var[v] - 0.90) for v in var_names]))

    return CalibrationMetrics(
        picp_90=overall_picp,
        mpiw_90=overall_mpiw,
        coverage_error=cov_err,
        expected_calibration_error=ece,
        coverage_by_variable=cov_by_var,
        width_by_variable=width_by_var,
    )


def compute_expected_calibration_error(
    predicted_probs: np.ndarray | Tensor,
    binary_outcomes: np.ndarray | Tensor,
    num_bins: int = 10,
) -> float:
    """Compute standard Expected Calibration Error (ECE) for binary failure probabilities."""
    if isinstance(predicted_probs, Tensor):
        probs = predicted_probs.detach().cpu().numpy().flatten()
    else:
        probs = np.asarray(predicted_probs).flatten()

    if isinstance(binary_outcomes, Tensor):
        labels = binary_outcomes.detach().cpu().numpy().flatten()
    else:
        labels = np.asarray(binary_outcomes).flatten()

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n_total = len(probs)

    for i in range(num_bins):
        bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            bin_conf = np.mean(probs[bin_mask])
            bin_acc = np.mean(labels[bin_mask])
            bin_weight = np.sum(bin_mask) / n_total
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)
