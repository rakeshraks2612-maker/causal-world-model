"""Evaluation Metrics, Baselines (Persistence & MLP), and Benchmarking Tools.

Provides:
- Baseline A: Persistence Model (O_hat_{t+1} = O_t)
- Baseline B: Supervised MLP Dynamics Model ([O_t, A_t] -> MLP -> O_hat_{t+1})
- Evaluation metrics: MAE, RMSE, NLL, per-channel breakdown, and metrics.json exporter.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


@dataclass
class ChannelMetrics:
    """Evaluation metrics for a single observable channel."""

    channel_name: str
    mae: float
    rmse: float
    nll: Optional[float] = None


@dataclass
class EvaluationSummary:
    """Summary of one-step prediction metrics for a model."""

    model_name: str
    overall_mae: float
    overall_rmse: float
    overall_nll: Optional[float]
    per_channel: Dict[str, ChannelMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "overall_mae": self.overall_mae,
            "overall_rmse": self.overall_rmse,
            "overall_nll": self.overall_nll,
            "per_channel": {k: asdict(v) for k, v in self.per_channel.items()},
        }


class PersistenceBaseline:
    """Baseline A: Predicts the next observation is identical to the current observation: O_hat_{t+1} = O_t."""

    def predict_one_step(self, observations: Tensor) -> Tensor:
        """Predict next step by shifting current observation forward.
        
        Args:
            observations: Tensor [B, T, 8]
            
        Returns:
            Predicted next observations for t=1..T-1: [B, T-1, 8]
        """
        # Pred at t is observation at t-1: pred[:, 0] = obs[:, 0], ..., pred[:, T-2] = obs[:, T-2]
        return observations[:, :-1]


class MLPDynamicsBaseline(nn.Module):
    """Baseline B: Simple supervised feedforward dynamics model mapping [O_t, A_t] -> O_{t+1}."""

    def __init__(self, obs_dim: int = 8, action_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Predict next observation from current observation and action."""
        x = torch.cat([obs, action], dim=-1)
        delta = self.net(x)
        return obs + delta  # Residual connection


def compute_regression_metrics(
    predictions: Tensor,   # [N, 8]
    targets: Tensor,       # [N, 8]
    mask: Optional[Tensor] = None, # [N, 8]
    pred_logvar: Optional[Tensor] = None, # [N, 8]
    model_name: str = "Model",
) -> EvaluationSummary:
    """Compute per-channel and aggregate MAE, RMSE, and Gaussian NLL."""
    ch_names = list(OBSERVABLE_VARIABLES)
    per_channel: Dict[str, ChannelMetrics] = {}

    mask_bool = mask.bool() if mask is not None else torch.ones_like(targets, dtype=torch.bool)
    all_valid_errors = []
    all_valid_sq_errors = []
    all_valid_nlls = []

    for ch_idx, ch_name in enumerate(ch_names):
        ch_p = predictions[:, ch_idx]
        ch_t = targets[:, ch_idx]
        ch_m = mask_bool[:, ch_idx]

        v_p = ch_p[ch_m]
        v_t = ch_t[ch_m]

        if v_t.numel() > 0:
            diff = torch.abs(v_p - v_t)
            mae_val = float(torch.mean(diff).item())
            rmse_val = float(torch.sqrt(torch.mean((v_p - v_t) ** 2)).item())

            all_valid_errors.append(diff)
            all_valid_sq_errors.append((v_p - v_t) ** 2)

            nll_val = None
            if pred_logvar is not None:
                ch_lv = pred_logvar[:, ch_idx][ch_m]
                ch_var = torch.exp(ch_lv)
                nll_t = 0.5 * (np.log(2.0 * np.pi) + ch_lv + ((v_p - v_t) ** 2) / (ch_var + 1e-8))
                nll_val = float(torch.mean(nll_t).item())
                all_valid_nlls.append(nll_t)

            per_channel[ch_name] = ChannelMetrics(
                channel_name=ch_name,
                mae=mae_val,
                rmse=rmse_val,
                nll=nll_val,
            )

    overall_mae = float(torch.mean(torch.cat(all_valid_errors)).item()) if all_valid_errors else 0.0
    overall_rmse = float(torch.sqrt(torch.mean(torch.cat(all_valid_sq_errors))).item()) if all_valid_sq_errors else 0.0
    overall_nll = float(torch.mean(torch.cat(all_valid_nlls)).item()) if all_valid_nlls else None

    return EvaluationSummary(
        model_name=model_name,
        overall_mae=overall_mae,
        overall_rmse=overall_rmse,
        overall_nll=overall_nll,
        per_channel=per_channel,
    )


def format_benchmark_table(eval_summaries: List[EvaluationSummary]) -> str:
    """Format evaluation comparison table for audit reporting."""
    lines = []
    lines.append("=" * 115)
    lines.append(f"{'Model':<16} | {'T_core MAE':<12} | {'T_cool MAE':<12} | {'P_sys MAE':<12} | {'F_cool MAE':<12} | {'L_cpu MAE':<12} | {'Overall MAE':<12}")
    lines.append("-" * 115)

    for s in eval_summaries:
        t_core = s.per_channel.get("T_core", ChannelMetrics("T_core", 0, 0)).mae
        t_cool = s.per_channel.get("T_cool", ChannelMetrics("T_cool", 0, 0)).mae
        p_sys = s.per_channel.get("P_sys", ChannelMetrics("P_sys", 0, 0)).mae
        f_cool = s.per_channel.get("F_cool", ChannelMetrics("F_cool", 0, 0)).mae
        l_cpu = s.per_channel.get("L_cpu", ChannelMetrics("L_cpu", 0, 0)).mae
        ov = s.overall_mae

        lines.append(
            f"{s.model_name:<16} | {t_core:>10.3f}°C | {t_cool:>10.3f}°C | {p_sys:>10.3f} bar| {f_cool:>9.3f} L/m | {l_cpu:>10.3f}%  | {ov:>10.3f}"
        )
    lines.append("=" * 115)
    return "\n".join(lines)


def export_metrics_json(
    eval_summaries: List[EvaluationSummary],
    file_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export benchmark evaluation results to machine-readable JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "metadata": metadata or {},
        "models": {s.model_name: s.to_dict() for s in eval_summaries},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
