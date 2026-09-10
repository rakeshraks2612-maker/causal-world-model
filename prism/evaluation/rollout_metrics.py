"""Rollout Metrics and Multi-Step Evaluation Data Structures."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.training.normalization import ObservationNormalizer
from prism.training.metrics import ChannelMetrics
from prism.diagnosis.domain_metrics import DOMAINS, DomainSummary, PHYSICAL_UNITS


@dataclass
class HorizonChannelMetric:
    """Evaluation metrics for a specific channel at a specific rollout horizon."""

    channel_name: str
    horizon: int
    mae: float
    rmse: float
    nll: Optional[float] = None
    mean_error: Optional[float] = None


@dataclass
class HorizonEvaluationSummary:
    """Summary of all channels and domains at a specific rollout horizon."""

    horizon: int
    train_normalized_aggregate_mae: float
    raw_overall_mae: float
    overall_rmse: float
    overall_nll: Optional[float] = None
    per_channel: Dict[str, HorizonChannelMetric] = field(default_factory=dict)
    domains: Dict[str, DomainSummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "train_normalized_aggregate_mae": self.train_normalized_aggregate_mae,
            "raw_overall_mae": self.raw_overall_mae,
            "overall_rmse": self.overall_rmse,
            "overall_nll": self.overall_nll,
            "domains": {k: asdict(v) for k, v in self.domains.items()},
            "per_channel": {k: asdict(v) for k, v in self.per_channel.items()},
        }


@dataclass
class MultiStepEvaluationReport:
    """Comprehensive multi-step forecasting evaluation across multiple horizons."""

    model_name: str
    mode: str  # "open_loop" or "teacher_forced"
    horizons: List[int]
    by_horizon: Dict[int, HorizonEvaluationSummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "horizons": self.horizons,
            "by_horizon": {str(h): v.to_dict() for h, v in self.by_horizon.items()},
        }


def compute_horizon_metrics(
    predictions_phys: Tensor,          # [N, 8] unnormalized physical units
    targets_phys: Tensor,              # [N, 8] unnormalized physical units
    mask: Tensor,                      # [N, 8] binary mask
    normalizer: ObservationNormalizer, # Train-set fitted normalizer
    horizon: int,
    pred_logvar_phys: Optional[Tensor] = None, # [N, 8]
) -> HorizonEvaluationSummary:
    """Compute per-channel, domain, and standardized metrics for a specific horizon."""
    ch_names = list(OBSERVABLE_VARIABLES)
    mask_bool = mask.bool()

    train_mean = normalizer._mean_tensor.to(predictions_phys.device)
    train_std = normalizer._std_tensor.to(predictions_phys.device)

    norm_preds = (predictions_phys - train_mean) / train_std
    norm_targets = (targets_phys - train_mean) / train_std

    norm_diff = torch.abs(norm_preds - norm_targets)
    valid_norm_diff = norm_diff[mask_bool]
    train_norm_mae = float(torch.mean(valid_norm_diff).item()) if valid_norm_diff.numel() > 0 else 0.0

    per_channel: Dict[str, HorizonChannelMetric] = {}
    channel_norm_maes: Dict[str, float] = {}

    all_raw_errors = []
    all_raw_sq_errors = []
    all_nlls = []

    for ch_idx, ch_name in enumerate(ch_names):
        ch_m = mask_bool[:, ch_idx]
        ch_p = predictions_phys[:, ch_idx][ch_m]
        ch_t = targets_phys[:, ch_idx][ch_m]

        if ch_t.numel() > 0:
            diff = torch.abs(ch_p - ch_t)
            sq_diff = (ch_p - ch_t) ** 2
            mae = float(torch.mean(diff).item())
            rmse = float(torch.sqrt(torch.mean(sq_diff)).item())
            mean_err = float(torch.mean(ch_p - ch_t).item())

            all_raw_errors.append(diff)
            all_raw_sq_errors.append(sq_diff)

            n_diff = norm_diff[:, ch_idx][ch_m]
            channel_norm_maes[ch_name] = float(torch.mean(n_diff).item())

            nll_val = None
            if pred_logvar_phys is not None:
                ch_lv = pred_logvar_phys[:, ch_idx][ch_m]
                ch_var = torch.exp(ch_lv)
                nll_t = 0.5 * (np.log(2.0 * np.pi) + ch_lv + sq_diff / (ch_var + 1e-8))
                nll_val = float(torch.mean(nll_t).item())
                all_nlls.append(nll_t)

            per_channel[ch_name] = HorizonChannelMetric(
                channel_name=ch_name,
                horizon=horizon,
                mae=mae,
                rmse=rmse,
                nll=nll_val,
                mean_error=mean_err,
            )

    raw_overall_mae = float(torch.mean(torch.cat(all_raw_errors)).item()) if all_raw_errors else 0.0
    overall_rmse = float(torch.sqrt(torch.mean(torch.cat(all_raw_sq_errors))).item()) if all_raw_sq_errors else 0.0
    overall_nll = float(torch.mean(torch.cat(all_nlls)).item()) if all_nlls else None

    domains: Dict[str, DomainSummary] = {}
    for dom_name, dom_vars in DOMAINS.items():
        dom_maes = [per_channel[v].mae for v in dom_vars if v in per_channel]
        dom_rmses = [per_channel[v].rmse for v in dom_vars if v in per_channel]
        dom_norm_maes = [channel_norm_maes[v] for v in dom_vars if v in channel_norm_maes]

        domains[dom_name] = DomainSummary(
            domain_name=dom_name,
            variables=dom_vars,
            mae=float(np.mean(dom_maes)) if dom_maes else 0.0,
            rmse=float(np.mean(dom_rmses)) if dom_rmses else 0.0,
            normalized_mae=float(np.mean(dom_norm_maes)) if dom_norm_maes else 0.0,
        )

    return HorizonEvaluationSummary(
        horizon=horizon,
        train_normalized_aggregate_mae=train_norm_mae,
        raw_overall_mae=raw_overall_mae,
        overall_rmse=overall_rmse,
        overall_nll=overall_nll,
        per_channel=per_channel,
        domains=domains,
    )
