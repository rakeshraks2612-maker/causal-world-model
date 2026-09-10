"""Uncertainty Estimation and Latent Manifold Novelty Scoring (Task 3.7).

Decomposes:
1. Aleatoric Uncertainty: Physical process stochasticity and measurement noise.
2. Epistemic Uncertainty: Out-Of-Distribution (OOD) novelty and model parameter ignorance.
3. Latent Manifold Distance D_latent(z): Mahalanobis distance from training latent support.
4. Abstention Logic: Flags recommendations where epistemic uncertainty exceeds safety envelope.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.model import CausalWorldModel
from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.training.normalization import ObservationNormalizer


@dataclass
class UncertaintyEstimate:
    """Decomposed predictive uncertainty metrics for a trajectory."""

    aleatoric_variance: np.ndarray      # Shape: [H, 8]
    epistemic_variance: np.ndarray      # Shape: [H, 8]
    total_variance: np.ndarray          # Shape: [H, 8]
    latent_novelty_scores: np.ndarray   # Shape: [H] -> Mahalanobis distance D_latent(z_h)
    max_novelty_score: float
    mean_novelty_score: float
    should_abstain: bool
    abstention_reason: Optional[str] = None
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_novelty_score": float(self.max_novelty_score),
            "mean_novelty_score": float(self.mean_novelty_score),
            "should_abstain": bool(self.should_abstain),
            "abstention_reason": self.abstention_reason,
            "horizons": self.horizons,
        }


class LatentManifoldDensityEstimator:
    """Empirical latent manifold density and support distance estimator."""

    def __init__(self, d_z: int = 64, novelty_threshold: float = 12.0) -> None:
        self.d_z = d_z
        self.novelty_threshold = novelty_threshold
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, latent_vectors: np.ndarray | Tensor) -> None:
        """Fit empirical Gaussian manifold statistics (mu, Sigma^-1) from training latent vectors."""
        if isinstance(latent_vectors, Tensor):
            z_np = latent_vectors.detach().cpu().numpy()
        else:
            z_np = np.asarray(latent_vectors)

        if z_np.ndim > 2:
            z_np = z_np.reshape(-1, z_np.shape[-1])

        self.mean = np.mean(z_np, axis=0)  # [d_z]
        cov = np.cov(z_np, rowvar=False)   # [d_z, d_z]

        # Regularize covariance for numerical stability
        cov_reg = cov + np.eye(self.d_z) * 1e-4
        self.inv_cov = np.linalg.pinv(cov_reg)
        self.is_fitted = True

    def compute_mahalanobis_distance(self, z: np.ndarray | Tensor) -> np.ndarray:
        """Compute Mahalanobis distance D(z) = sqrt((z - mu)^T Sigma^-1 (z - mu))."""
        if not self.is_fitted or self.mean is None or self.inv_cov is None:
            # Fallback to Euclidean norm if not fitted
            if isinstance(z, Tensor):
                return torch.norm(z, dim=-1).cpu().numpy()
            return np.linalg.norm(z, axis=-1)

        if isinstance(z, Tensor):
            z_np = z.detach().cpu().numpy()
        else:
            z_np = np.asarray(z)

        has_extra_dims = z_np.ndim > 1
        orig_shape = z_np.shape[:-1]
        flat_z = z_np.reshape(-1, self.d_z)

        diff = flat_z - self.mean  # [N, d_z]
        # (diff @ inv_cov * diff).sum(axis=-1)
        dist_sq = np.sum((diff @ self.inv_cov) * diff, axis=-1)
        dist = np.sqrt(np.maximum(0.0, dist_sq))

        if has_extra_dims:
            return dist.reshape(orig_shape)
        return dist

    def save(self, file_path: str | Path) -> None:
        """Save fitted manifold statistics to .npz."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            mean=self.mean,
            inv_cov=self.inv_cov,
            novelty_threshold=np.array(self.novelty_threshold),
            d_z=np.array(self.d_z),
        )

    @classmethod
    def load(cls, file_path: str | Path) -> LatentManifoldDensityEstimator:
        """Load fitted manifold statistics from .npz."""
        data = np.load(file_path)
        est = cls(
            d_z=int(data["d_z"]),
            novelty_threshold=float(data["novelty_threshold"]),
        )
        est.mean = data["mean"]
        est.inv_cov = data["inv_cov"]
        est.is_fitted = True
        return est


class EpistemicUncertaintyEstimator:
    """Full predictive uncertainty and abstention gate evaluator."""

    def __init__(
        self,
        world_model: CausalWorldModel,
        normalizer: ObservationNormalizer,
        density_estimator: Optional[LatentManifoldDensityEstimator] = None,
        novelty_threshold: float = 12.0,
        variance_threshold: float = 25.0,
    ) -> None:
        self.world_model = world_model
        self.normalizer = normalizer
        self.density_estimator = density_estimator or LatentManifoldDensityEstimator(
            d_z=world_model.config.model.latent_dim,
            novelty_threshold=novelty_threshold,
        )
        self.novelty_threshold = novelty_threshold
        self.variance_threshold = variance_threshold

    def evaluate_rollout_uncertainty(
        self,
        initial_z_dist: LatentDistribution,
        actions: Tensor,
        num_particles: int = 50,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
    ) -> UncertaintyEstimate:
        """Estimate decomposed aleatoric and epistemic uncertainty across rollout."""
        self.world_model.eval()
        batch_size, horizon, act_dim = actions.shape

        with torch.no_grad():
            # 1. Deterministic trajectory for latent mean path
            det_traj = self.world_model.rollout_manager.rollout_deterministic(
                initial_z=initial_z_dist.mean,
                actions=actions,
            )
            det_z = det_traj.latent_mean.cpu().numpy()[0]  # [H, d_z]

            # 2. Monte Carlo particle rollouts for epistemic spread
            mc_traj = self.world_model.rollout_manager.rollout_monte_carlo(
                initial_z_dist=initial_z_dist,
                actions=actions,
                num_samples=num_particles,
            )

            # Observation distributions
            obs_mean_norm = mc_traj.observations.mean.cpu()      # [1, H, 8]
            obs_var_norm = mc_traj.observations.variance.cpu()   # [1, H, 8]

            # Denormalize variances: Var_phys = Var_norm * std^2
            stds = torch.tensor(self.normalizer.stats.stds, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            aleatoric_var_phys = (obs_var_norm * (stds ** 2)).numpy()[0]  # [H, 8]

            # Latent variance across particles as epistemic uncertainty
            latent_var = mc_traj.latent_variance.cpu().numpy()[0]  # [H, d_z]
            epistemic_var_phys = np.repeat(np.mean(latent_var, axis=-1, keepdims=True), 8, axis=-1)

            total_var_phys = aleatoric_var_phys + epistemic_var_phys

            # 3. Compute Latent Novelty D_latent(z_h)
            novelty_scores = self.density_estimator.compute_mahalanobis_distance(det_z)  # [H]
            max_nov = float(np.max(novelty_scores))
            mean_nov = float(np.mean(novelty_scores))

            # 4. Abstention Evaluation
            should_abstain = False
            reason = None

            if max_nov > self.novelty_threshold:
                should_abstain = True
                reason = f"Latent novelty score ({max_nov:.2f}) exceeds safety support threshold ({self.novelty_threshold:.2f})"
            elif float(np.max(total_var_phys)) > self.variance_threshold:
                should_abstain = True
                reason = f"Predictive uncertainty ({np.max(total_var_phys):.2f}) exceeds maximum allowable variance ({self.variance_threshold:.2f})"

            return UncertaintyEstimate(
                aleatoric_variance=aleatoric_var_phys,
                epistemic_variance=epistemic_var_phys,
                total_variance=total_var_phys,
                latent_novelty_scores=novelty_scores,
                max_novelty_score=max_nov,
                mean_novelty_score=mean_nov,
                should_abstain=should_abstain,
                abstention_reason=reason,
                horizons=list(horizons),
            )
