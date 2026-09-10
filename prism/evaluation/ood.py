"""OOD Evaluation and Uncertainty Generalization Engine."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from prism.dataset.schema import LearnerEpisode
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.forecasting import evaluate_prism_multistep
from prism.evaluation.uncertainty_metrics import evaluate_monte_carlo_uncertainty, MonteCarloUncertaintyReport


@dataclass
class RegimeUncertaintySummary:
    """Summary of predictive error and uncertainty behavior for a specific regime (Test or OOD)."""

    regime_name: str
    num_episodes: int
    train_normalized_mae_h1: float
    train_normalized_mae_h40: float
    raw_mae_h1: float
    raw_mae_h40: float
    mean_uncertainty_std_h1: float
    mean_uncertainty_std_h40: float
    empirical_coverage_90_h1: float
    empirical_coverage_90_h40: float
    error_uncertainty_correlation_h1: float
    error_uncertainty_correlation_h40: float


@dataclass
class OODEvaluationReport:
    """Comprehensive comparison of In-Distribution Test vs OOD-1..5 regimes."""

    model_name: str
    regimes: Dict[str, RegimeUncertaintySummary] = field(default_factory=dict)
    detailed_reports: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "regimes": {k: asdict(v) for k, v in self.regimes.items()},
            "detailed_reports": self.detailed_reports,
        }


def evaluate_ood_regimes(
    model: CausalWorldModel,
    test_episodes: List[LearnerEpisode],
    ood_root_dir: str | Path,
    normalizer: ObservationNormalizer,
    horizons: List[int] = [1, 5, 10, 20, 40],
    num_particles: int = 50,
    device: Optional[torch.device] = None,
) -> OODEvaluationReport:
    """Evaluate frozen world model on In-Distribution Test vs OOD-1 through OOD-5 regimes without retraining."""
    dev = device or torch.device("cpu")
    ood_path = Path(ood_root_dir)

    regimes_to_eval = [
        ("Test_InDistribution", test_episodes),
    ]

    ood_dirs = sorted([d for d in ood_path.iterdir() if d.is_dir()])
    for d in ood_dirs:
        eps = [LearnerEpisode.load_npz(f) for f in sorted(d.glob("*.npz"))]
        if eps:
            regimes_to_eval.append((d.name, eps))

    summaries: Dict[str, RegimeUncertaintySummary] = {}
    detailed: Dict[str, Any] = {}

    for regime_name, eps in regimes_to_eval:
        # 1. Multi-Step Open-Loop Forecasting Report
        fc_rep = evaluate_prism_multistep(
            model=model,
            episodes=eps,
            normalizer=normalizer,
            horizons=horizons,
            context_length=40,
            stride=5,
            mode="open_loop",
            deterministic=True,
            device=dev,
        )

        # 2. Monte Carlo Uncertainty Report
        unc_rep = evaluate_monte_carlo_uncertainty(
            model=model,
            episodes=eps,
            normalizer=normalizer,
            horizons=horizons,
            num_particles=num_particles,
            context_length=40,
            stride=5,
            device=dev,
        )

        # Extract min and max horizon metrics
        h_min = min(horizons)
        h_max = max(horizons)

        h_min_fc = fc_rep.by_horizon[h_min]
        h_max_fc = fc_rep.by_horizon[h_max]

        h_min_unc = unc_rep.by_horizon[h_min]
        h_max_unc = unc_rep.by_horizon[h_max]

        # Aggregate error-uncertainty correlation across all channels at h_min and h_max
        corr_h_min = float(np.mean([ch.error_uncertainty_correlation for ch in h_min_unc.per_channel.values()]))
        corr_h_max = float(np.mean([ch.error_uncertainty_correlation for ch in h_max_unc.per_channel.values()]))

        summaries[regime_name] = RegimeUncertaintySummary(
            regime_name=regime_name,
            num_episodes=len(eps),
            train_normalized_mae_h1=h_min_fc.train_normalized_aggregate_mae,
            train_normalized_mae_h40=h_max_fc.train_normalized_aggregate_mae,
            raw_mae_h1=h_min_fc.raw_overall_mae,
            raw_mae_h40=h_max_fc.raw_overall_mae,
            mean_uncertainty_std_h1=h_min_unc.mean_predictive_std_norm,
            mean_uncertainty_std_h40=h_max_unc.mean_predictive_std_norm,
            empirical_coverage_90_h1=h_min_unc.overall_empirical_coverage_90,
            empirical_coverage_90_h40=h_max_unc.overall_empirical_coverage_90,
            error_uncertainty_correlation_h1=corr_h_min,
            error_uncertainty_correlation_h40=corr_h_max,
        )

        detailed[regime_name] = {
            "forecasting": fc_rep.to_dict(),
            "uncertainty": unc_rep.to_dict(),
        }

    return OODEvaluationReport(
        model_name="PRISM_OOD_Generalization",
        regimes=summaries,
        detailed_reports=detailed,
    )
