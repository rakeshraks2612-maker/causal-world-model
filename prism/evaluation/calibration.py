"""Statistical Calibration, Coverage Error, and Uncertainty Correlation Utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import spearmanr


@dataclass
class CalibrationStatistics:
    """Statistical summary of prediction interval calibration."""

    nominal_coverage: float
    empirical_coverage: float
    coverage_error: float
    mean_interval_width: float
    median_interval_width: float
    pearson_error_correlation: float
    spearman_error_correlation: float


def compute_interval_calibration(
    targets: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    predictive_stds: np.ndarray,
    point_predictions: np.ndarray,
    nominal_coverage: float = 0.90,
) -> CalibrationStatistics:
    """Compute empirical coverage, interval width, and error-uncertainty correlation.
    
    Args:
        targets: Array of true values [N]
        lower_bounds: Array of lower CI bounds [N]
        upper_bounds: Array of upper CI bounds [N]
        predictive_stds: Array of predictive standard deviations [N]
        point_predictions: Array of point predictions (mean) [N]
        nominal_coverage: Target confidence level (e.g., 0.90)
        
    Returns:
        CalibrationStatistics
    """
    inside = (targets >= lower_bounds) & (targets <= upper_bounds)
    emp_cov = float(np.mean(inside))
    cov_err = float(np.abs(emp_cov - nominal_coverage))

    widths = upper_bounds - lower_bounds
    mean_w = float(np.mean(widths))
    median_w = float(np.median(widths))

    errors = np.abs(point_predictions - targets)
    
    if np.std(predictive_stds) > 1e-7 and np.std(errors) > 1e-7:
        p_corr = float(np.corrcoef(errors, predictive_stds)[0, 1])
        if np.isnan(p_corr):
            p_corr = 0.0
        res = spearmanr(errors, predictive_stds)
        s_corr = float(res.statistic if hasattr(res, "statistic") else res[0])
        if np.isnan(s_corr):
            s_corr = 0.0
    else:
        p_corr = 0.0
        s_corr = 0.0

    return CalibrationStatistics(
        nominal_coverage=nominal_coverage,
        empirical_coverage=emp_cov,
        coverage_error=cov_err,
        mean_interval_width=mean_w,
        median_interval_width=median_w,
        pearson_error_correlation=p_corr,
        spearman_error_correlation=s_corr,
    )
