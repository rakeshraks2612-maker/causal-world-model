"""Domain-Grouped Metrics, Physical Units, and Train-Standardized Aggregate Metrics."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.training.normalization import ObservationNormalizer
from prism.training.metrics import ChannelMetrics, EvaluationSummary


DOMAINS = {
    "thermal": ["T_core", "T_cool"],
    "hydraulic": ["P_sys", "F_cool"],
    "compute": ["L_cpu", "P_elec"],
    "actuation": ["V_pos"],
    "vibration": ["Vib_pump"],
}

PHYSICAL_UNITS = {
    "T_core": "°C",
    "T_cool": "°C",
    "P_sys": "bar",
    "F_cool": "L/min",
    "L_cpu": "%",
    "V_pos": "%",
    "Vib_pump": "mm/s",
    "P_elec": "kW",
}


@dataclass
class DomainSummary:
    """Aggregated metrics across a physical functional domain."""
    
    domain_name: str
    variables: List[str]
    mae: float
    rmse: float
    normalized_mae: float


@dataclass
class ComprehensiveEvaluationReport:
    """Comprehensive evaluation containing physical per-channel, domain-grouped, and train-normalized aggregate metrics."""

    model_name: str
    per_channel: Dict[str, ChannelMetrics]
    domains: Dict[str, DomainSummary]
    train_normalized_aggregate_mae: float
    raw_overall_mae: float
    overall_rmse: float
    overall_nll: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "train_normalized_aggregate_mae": self.train_normalized_aggregate_mae,
            "raw_overall_mae": self.raw_overall_mae,
            "overall_rmse": self.overall_rmse,
            "overall_nll": self.overall_nll,
            "domains": {k: asdict(v) for k, v in self.domains.items()},
            "per_channel": {k: asdict(v) for k, v in self.per_channel.items()},
        }


def compute_comprehensive_metrics(
    predictions_phys: Tensor,         # [N, 8] unnormalized physical units
    targets_phys: Tensor,             # [N, 8] unnormalized physical units
    mask: Tensor,                     # [N, 8] binary mask
    normalizer: ObservationNormalizer,# Train-set fitted normalizer
    pred_logvar_phys: Optional[Tensor] = None, # [N, 8]
    model_name: str = "Model",
) -> ComprehensiveEvaluationReport:
    """Compute physical metrics, domain groupings, and train-normalized aggregate metrics."""
    ch_names = list(OBSERVABLE_VARIABLES)
    mask_bool = mask.bool()
    
    # 1. Compute Standardized Errors using Training-Set-Only Statistics
    train_mean = normalizer._mean_tensor.to(predictions_phys.device)
    train_std = normalizer._std_tensor.to(predictions_phys.device)
    
    norm_preds = (predictions_phys - train_mean) / train_std
    norm_targets = (targets_phys - train_mean) / train_std
    
    norm_diff = torch.abs(norm_preds - norm_targets)
    valid_norm_diff = norm_diff[mask_bool]
    train_normalized_aggregate_mae = float(torch.mean(valid_norm_diff).item()) if valid_norm_diff.numel() > 0 else 0.0

    # 2. Per-Channel Physical Metrics
    per_channel: Dict[str, ChannelMetrics] = {}
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
            
            all_raw_errors.append(diff)
            all_raw_sq_errors.append(sq_diff)
            
            # Normalized channel MAE
            n_diff = norm_diff[:, ch_idx][ch_m]
            channel_norm_maes[ch_name] = float(torch.mean(n_diff).item())
            
            nll_val = None
            if pred_logvar_phys is not None:
                ch_lv = pred_logvar_phys[:, ch_idx][ch_m]
                ch_var = torch.exp(ch_lv)
                nll_t = 0.5 * (np.log(2.0 * np.pi) + ch_lv + sq_diff / (ch_var + 1e-8))
                nll_val = float(torch.mean(nll_t).item())
                all_nlls.append(nll_t)
                
            per_channel[ch_name] = ChannelMetrics(
                channel_name=ch_name,
                mae=mae,
                rmse=rmse,
                nll=nll_val,
            )
            
    raw_overall_mae = float(torch.mean(torch.cat(all_raw_errors)).item()) if all_raw_errors else 0.0
    overall_rmse = float(torch.sqrt(torch.mean(torch.cat(all_raw_sq_errors))).item()) if all_raw_sq_errors else 0.0
    overall_nll = float(torch.mean(torch.cat(all_nlls)).item()) if all_nlls else None

    # 3. Domain Grouped Metrics
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

    return ComprehensiveEvaluationReport(
        model_name=model_name,
        per_channel=per_channel,
        domains=domains,
        train_normalized_aggregate_mae=train_normalized_aggregate_mae,
        raw_overall_mae=raw_overall_mae,
        overall_rmse=overall_rmse,
        overall_nll=overall_nll,
    )
