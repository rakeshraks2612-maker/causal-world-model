"""Tests for Statistical Calibration and Prediction Interval Coverage."""

from __future__ import annotations
import pytest
import numpy as np
from prism.evaluation.calibration import compute_interval_calibration


def test_synthetic_gaussian_calibration_near_90():
    """Verify that a true calibrated Gaussian N(mu, sigma^2) exhibits ~90% empirical coverage under +/- 1.645 sigma."""
    np.random.seed(42)
    N = 10000
    mu = np.random.uniform(20.0, 50.0, size=N)
    sigma = np.random.uniform(1.0, 5.0, size=N)
    targets = np.random.normal(mu, sigma)

    low_90 = mu - 1.645 * sigma
    high_90 = mu + 1.645 * sigma

    stats = compute_interval_calibration(
        targets=targets,
        lower_bounds=low_90,
        upper_bounds=high_90,
        predictive_stds=sigma,
        point_predictions=mu,
        nominal_coverage=0.90,
    )

    # Empirical coverage on 10,000 samples should be within 1.5% of 90% (88.5% to 91.5%)
    assert 0.885 <= stats.empirical_coverage <= 0.915
    assert stats.coverage_error <= 0.015
    # Absolute error and sigma should have positive correlation
    assert stats.pearson_error_correlation > 0.35


def test_wide_vs_narrow_interval_distinction():
    """Verify that wide intervals increase coverage at the cost of interval width."""
    targets = np.array([10.0, 20.0, 30.0, 40.0])
    preds = np.array([10.0, 20.0, 30.0, 40.0])
    stds = np.array([1.0, 1.0, 1.0, 1.0])

    # Narrow: width 2.0
    narrow_stats = compute_interval_calibration(targets, preds - 1.0, preds + 1.0, stds, preds)
    # Wide: width 20.0
    wide_stats = compute_interval_calibration(targets, preds - 10.0, preds + 10.0, stds, preds)

    assert narrow_stats.mean_interval_width == 2.0
    assert wide_stats.mean_interval_width == 20.0
