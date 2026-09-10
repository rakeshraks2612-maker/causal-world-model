"""Monte Carlo Uncertainty Engine and Interval Prediction Contracts."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prism.dataset.schema import LearnerEpisode
from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import LatentDistribution
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


@dataclass
class ChannelUncertaintySummary:
    """Predictive uncertainty and calibration metrics for a single channel at horizon h."""

    channel_name: str
    horizon: int
    predictive_mean_mae: float
    predictive_mean_rmse: float
    mean_predictive_std: float
    mean_interval_width_90: float
    empirical_coverage_90: float
    coverage_error: float
    error_uncertainty_correlation: float
    spearman_correlation: float


@dataclass
class HorizonUncertaintySummary:
    """Aggregated uncertainty metrics across all channels for a specific horizon."""

    horizon: int
    mean_predictive_std_norm: float
    mean_interval_width_norm: float
    overall_empirical_coverage_90: float
    overall_coverage_error: float
    per_channel: Dict[str, ChannelUncertaintySummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "mean_predictive_std_norm": self.mean_predictive_std_norm,
            "mean_interval_width_norm": self.mean_interval_width_norm,
            "overall_empirical_coverage_90": self.overall_empirical_coverage_90,
            "overall_coverage_error": self.overall_coverage_error,
            "per_channel": {k: asdict(v) for k, v in self.per_channel.items()},
        }


@dataclass
class MonteCarloUncertaintyReport:
    """Comprehensive Monte Carlo uncertainty and calibration report."""

    model_name: str
    num_particles: int
    horizons: List[int]
    by_horizon: Dict[int, HorizonUncertaintySummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_particles": self.num_particles,
            "horizons": self.horizons,
            "by_horizon": {str(h): v.to_dict() for h, v in self.by_horizon.items()},
        }


def evaluate_monte_carlo_uncertainty(
    model: CausalWorldModel,
    episodes: List[LearnerEpisode],
    normalizer: ObservationNormalizer,
    horizons: List[int] = [1, 5, 10, 20, 40],
    num_particles: int = 50,
    context_length: int = 40,
    stride: int = 5,
    device: Optional[torch.device] = None,
) -> MonteCarloUncertaintyReport:
    """Execute Monte Carlo 50-particle ensemble rollouts and calculate predictive intervals & calibration."""
    from scipy.stats import spearmanr

    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    max_horizon = max(horizons)
    ch_names = list(OBSERVABLE_VARIABLES)
    std_t = normalizer._std_tensor.to(dev)
    mean_t = normalizer._mean_tensor.to(dev)

    # Per horizon storage:
    # dict[h -> dict of arrays]: "mean_phys", "std_phys", "target_phys", "mask", "low90", "high90", "std_norm"
    storage_by_h: Dict[int, Dict[str, List[np.ndarray]]] = {
        h: {"mean_phys": [], "std_phys": [], "target_phys": [], "mask": [], "low90": [], "high90": [], "std_norm": []}
        for h in horizons
    }

    with torch.no_grad():
        for ep in episodes:
            T_ep = len(ep.timestamps)
            if T_ep < context_length + max_horizon:
                continue

            for start_idx in range(0, T_ep - context_length - max_horizon + 1, stride):
                raw_ctx_obs = ep.observations[start_idx : start_idx + context_length]
                ctx_mask = ep.observation_mask[start_idx : start_idx + context_length]
                ctx_act = ep.actions[start_idx : start_idx + context_length]

                norm_ctx_obs = normalizer.normalize(torch.tensor(raw_ctx_obs, dtype=torch.float32)).unsqueeze(0).to(dev)
                ctx_mask_t = torch.tensor(ctx_mask, dtype=torch.float32).unsqueeze(0).to(dev)
                ctx_act_t = torch.tensor(ctx_act, dtype=torch.float32).unsqueeze(0).to(dev)

                ctx_inputs = ModelInputs(observations=norm_ctx_obs, observation_mask=ctx_mask_t, actions=ctx_act_t)

                fut_act_np = ep.actions[start_idx + context_length - 1 : start_idx + context_length + max_horizon - 1]
                fut_act_t = torch.tensor(fut_act_np, dtype=torch.float32).unsqueeze(0).to(dev) # [1, max_horizon, 4]

                fut_raw_obs = ep.observations[start_idx + context_length : start_idx + context_length + max_horizon]
                fut_mask = ep.observation_mask[start_idx + context_length : start_idx + context_length + max_horizon]

                # 1. Infer latent distribution at context boundary
                post_latents, _ = model.encode(ctx_inputs)
                z_dist = LatentDistribution(
                    mean=post_latents.mean[:, -1],
                    logvar=post_latents.logvar[:, -1],
                    min_std=post_latents.min_std,
                    max_std=post_latents.max_std,
                )

                # 2. Monte Carlo particle rollout with N particles
                # Propagates particles through stochastic transition and decoder
                mc_traj = model.rollout_manager.rollout_monte_carlo(
                    initial_z_dist=z_dist,
                    actions=fut_act_t,
                    num_samples=num_particles,
                )

                # mc_traj.observations.mean: [1, max_horizon, 8] (empirical mean in normalized space)
                # mc_traj.observations.variance: [1, max_horizon, 8] (empirical variance in normalized space)
                obs_mean_norm = mc_traj.observations.mean.squeeze(0)     # [max_horizon, 8]
                obs_var_norm = mc_traj.observations.variance.squeeze(0)   # [max_horizon, 8]
                obs_std_norm = torch.sqrt(obs_var_norm + 1e-8)

                # Convert to physical units
                obs_mean_phys = normalizer.denormalize(obs_mean_norm.cpu()).to(dev)
                obs_std_phys = obs_std_norm * std_t

                # Analytical / Gaussian 90% Prediction Interval: CI90 = [mu - 1.645*sigma, mu + 1.645*sigma] (unclipped)
                low_90_phys = obs_mean_phys - 1.645 * obs_std_phys
                high_90_phys = obs_mean_phys + 1.645 * obs_std_phys

                for h in horizons:
                    step_idx = h - 1
                    storage_by_h[h]["mean_phys"].append(obs_mean_phys[step_idx : step_idx + 1].cpu().numpy())
                    storage_by_h[h]["std_phys"].append(obs_std_phys[step_idx : step_idx + 1].cpu().numpy())
                    storage_by_h[h]["target_phys"].append(fut_raw_obs[step_idx : step_idx + 1])
                    storage_by_h[h]["mask"].append(fut_mask[step_idx : step_idx + 1])
                    storage_by_h[h]["low90"].append(low_90_phys[step_idx : step_idx + 1].cpu().numpy())
                    storage_by_h[h]["high90"].append(high_90_phys[step_idx : step_idx + 1].cpu().numpy())
                    storage_by_h[h]["std_norm"].append(obs_std_norm[step_idx : step_idx + 1].cpu().numpy())

    # Aggregate summaries per horizon
    by_horizon: Dict[int, HorizonUncertaintySummary] = {}

    for h in horizons:
        means_p = np.concatenate(storage_by_h[h]["mean_phys"], axis=0)      # [M, 8]
        stds_p = np.concatenate(storage_by_h[h]["std_phys"], axis=0)        # [M, 8]
        targets_p = np.concatenate(storage_by_h[h]["target_phys"], axis=0)  # [M, 8]
        masks_b = np.concatenate(storage_by_h[h]["mask"], axis=0).astype(bool) # [M, 8]
        lows_p = np.concatenate(storage_by_h[h]["low90"], axis=0)          # [M, 8]
        highs_p = np.concatenate(storage_by_h[h]["high90"], axis=0)        # [M, 8]
        stds_n = np.concatenate(storage_by_h[h]["std_norm"], axis=0)        # [M, 8]

        per_ch: Dict[str, ChannelUncertaintySummary] = {}
        all_coverages = []
        all_cov_errors = []

        for ch_idx, ch_name in enumerate(ch_names):
            m_valid = masks_b[:, ch_idx]
            ch_mu = means_p[:, ch_idx][m_valid]
            ch_std = stds_p[:, ch_idx][m_valid]
            ch_y = targets_p[:, ch_idx][m_valid]
            ch_low = lows_p[:, ch_idx][m_valid]
            ch_high = highs_p[:, ch_idx][m_valid]

            if len(ch_y) > 0:
                errors = np.abs(ch_mu - ch_y)
                mae = float(np.mean(errors))
                rmse = float(np.sqrt(np.mean((ch_mu - ch_y) ** 2)))
                mean_std = float(np.mean(ch_std))
                mean_width = float(np.mean(ch_high - ch_low))

                # Empirical 90% Coverage: fraction of true y inside [low, high]
                inside = (ch_y >= ch_low) & (ch_y <= ch_high)
                cov = float(np.mean(inside))
                cov_err = float(np.abs(cov - 0.90))

                all_coverages.append(cov)
                all_cov_errors.append(cov_err)

                # Pearson and Spearman correlation between |error| and predictive sigma
                if np.std(ch_std) > 1e-7 and np.std(errors) > 1e-7:
                    pearson_corr = float(np.corrcoef(errors, ch_std)[0, 1])
                    if np.isnan(pearson_corr):
                        pearson_corr = 0.0
                    res_sp = spearmanr(errors, ch_std)
                    sp_corr = float(res_sp.statistic if hasattr(res_sp, "statistic") else res_sp[0])
                    if np.isnan(sp_corr):
                        sp_corr = 0.0
                else:
                    pearson_corr = 0.0
                    sp_corr = 0.0

                per_ch[ch_name] = ChannelUncertaintySummary(
                    channel_name=ch_name,
                    horizon=h,
                    predictive_mean_mae=mae,
                    predictive_mean_rmse=rmse,
                    mean_predictive_std=mean_std,
                    mean_interval_width_90=mean_width,
                    empirical_coverage_90=cov,
                    coverage_error=cov_err,
                    error_uncertainty_correlation=pearson_corr,
                    spearman_correlation=sp_corr,
                )

        by_horizon[h] = HorizonUncertaintySummary(
            horizon=h,
            mean_predictive_std_norm=float(np.mean(stds_n)),
            mean_interval_width_norm=float(np.mean(stds_n) * 3.29),
            overall_empirical_coverage_90=float(np.mean(all_coverages)) if all_coverages else 0.0,
            overall_coverage_error=float(np.mean(all_cov_errors)) if all_cov_errors else 0.0,
            per_channel=per_ch,
        )

    return MonteCarloUncertaintyReport(
        model_name="PRISM_MonteCarlo_50",
        num_particles=num_particles,
        horizons=horizons,
        by_horizon=by_horizon,
    )
