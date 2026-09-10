"""Uncertainty Quantification, Decomposition, and Calibration Contracts."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from prism.world_model.latent_state import LatentDistribution, ObservationDistribution


@dataclass
class UncertaintyMetrics:
    """Evaluation summary for uncertainty calibration across observable channels."""

    channel_name: str
    empirical_coverage: float   # Fraction of true values within CI (target: e.g. 0.90)
    expected_confidence: float  # Nominal confidence level (e.g. 0.90)
    calibration_error: float    # |empirical_coverage - expected_confidence|
    mean_interval_width: float  # Average width of predictive interval
    mean_nll: float             # Mean Negative Log-Likelihood


def compute_prediction_intervals(
    dist: ObservationDistribution,
    confidence: float = 0.90,
) -> Tuple[Tensor, Tensor]:
    """Compute symmetric predictive intervals [lower, upper] from an observation distribution."""
    return dist.confidence_interval(confidence=confidence)


def evaluate_calibration(
    predictions: ObservationDistribution,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    confidence_levels: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Dict[float, List[UncertaintyMetrics]]:
    """Evaluate empirical coverage and calibration error across all 8 observable channels.
    
    Args:
        predictions: Predicted ObservationDistribution with mean and variance
        targets: Ground-truth observed sensor measurements [B, (T), 8]
        mask: Optional observation mask [B, (T), 8]
        confidence_levels: Nominal confidence levels to evaluate
        
    Returns:
        Dict mapping nominal confidence -> List of UncertaintyMetrics per channel
    """
    channel_names = [
        "T_core", "T_cool", "P_sys", "F_cool",
        "L_cpu", "V_pos", "Vib_pump", "P_elec",
    ]
    results: Dict[float, List[UncertaintyMetrics]] = {}

    target_t = targets.detach()
    mask_t = mask.detach().bool() if mask is not None else torch.ones_like(target_t, dtype=torch.bool)

    for conf in confidence_levels:
        lower, upper = predictions.confidence_interval(conf)
        conf_metrics: List[UncertaintyMetrics] = []

        for ch_idx, ch_name in enumerate(channel_names):
            ch_targets = target_t[..., ch_idx]
            ch_lower = lower[..., ch_idx]
            ch_upper = upper[..., ch_idx]
            ch_mask = mask_t[..., ch_idx]

            valid_targets = ch_targets[ch_mask]
            valid_lower = ch_lower[ch_mask]
            valid_upper = ch_upper[ch_mask]

            if valid_targets.numel() > 0:
                in_interval = (valid_targets >= valid_lower) & (valid_targets <= valid_upper)
                emp_cov = float(in_interval.float().mean().item())
                width = float((valid_upper - valid_lower).mean().item())
                cal_err = abs(emp_cov - conf)

                # Channel-wise NLL
                ch_mean = predictions.mean[..., ch_idx][ch_mask]
                ch_var = predictions.variance[..., ch_idx][ch_mask]
                nll = 0.5 * (torch.log(2.0 * np.pi * ch_var) + (valid_targets - ch_mean) ** 2 / ch_var).mean().item()

                conf_metrics.append(UncertaintyMetrics(
                    channel_name=ch_name,
                    empirical_coverage=emp_cov,
                    expected_confidence=conf,
                    calibration_error=cal_err,
                    mean_interval_width=width,
                    mean_nll=float(nll),
                ))

        results[conf] = conf_metrics

    return results


def compute_ood_uncertainty_score(
    latent_dist: LatentDistribution,
    prior: Optional[LatentDistribution] = None,
) -> Tensor:
    """Compute an epistemic OOD anomaly score based on latent KL divergence from standard prior."""
    # High KL from prior N(0, I) indicates latent state is far from training distribution
    kl = latent_dist.kl_divergence(prior)
    return kl
