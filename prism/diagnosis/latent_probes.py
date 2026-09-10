"""Diagnostic Linear Probes for Evaluating Latent Space Information Content."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch import Tensor

from prism.simulator.state import LATENT_VARIABLES
from prism.dataset.schema import OracleEpisode
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


@dataclass
class ProbeMetric:
    """Evaluation metrics for a linear probe on a latent target variable."""

    variable_name: str
    r2_score: float
    mae: float
    rmse: float
    target_mean: float
    target_std: float


@dataclass
class LatentProbeReport:
    """Comprehensive report on hidden-state linear probe decodability."""

    model_name: str
    latent_dim: int
    probe_metrics: Dict[str, ProbeMetric]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "latent_dim": self.latent_dim,
            "probes": {k: asdict(v) for k, v in self.probe_metrics.items()},
        }


def fit_and_evaluate_latent_probes(
    model: CausalWorldModel,
    train_oracle_episodes: List[OracleEpisode],
    test_oracle_episodes: List[OracleEpisode],
    normalizer: ObservationNormalizer,
    ridge_alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> LatentProbeReport:
    """Extract latent representations Z_t from frozen encoder and train linear probes to predict unobserved physical variables.
    
    Variables evaluated: T_amb, W_wear, Q_internal, xi_leak.
    
    Args:
        model: Trained CausalWorldModel
        train_oracle_episodes: Oracle episodes from train split (contains ground_truth_states)
        test_oracle_episodes: Oracle episodes from test split
        normalizer: Train-set fitted ObservationNormalizer
        ridge_alpha: L2 regularization for ridge regression
        device: Torch device
        
    Returns:
        LatentProbeReport with R^2, MAE, and RMSE per unobserved latent variable.
    """
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    def extract_latents_and_targets(episodes: List[OracleEpisode]) -> Tuple[np.ndarray, np.ndarray]:
        all_z = []
        all_y = []

        with torch.no_grad():
            for ep in episodes:
                # Observable inputs (normalized)
                norm_obs = normalizer.normalize(torch.tensor(ep.observations, dtype=torch.float32)).unsqueeze(0).to(dev)
                mask = torch.tensor(ep.observation_mask, dtype=torch.float32).unsqueeze(0).to(dev)
                act = torch.tensor(ep.actions, dtype=torch.float32).unsqueeze(0).to(dev)

                inputs = ModelInputs(observations=norm_obs, observation_mask=mask, actions=act)
                post_latents, _ = model.encode(inputs)
                z_mean = post_latents.mean.squeeze(0).cpu().numpy() # [T, d_z]

                # Latent targets from ground_truth_states[:, 8:12]
                y_latents = ep.ground_truth_states[:, 8:12] # [T, 4]

                all_z.append(z_mean)
                all_y.append(y_latents)

        return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)

    z_train, y_train = extract_latents_and_targets(train_oracle_episodes)
    z_test, y_test = extract_latents_and_targets(test_oracle_episodes)

    # Standardize features
    z_mean = np.mean(z_train, axis=0, keepdims=True)
    z_std = np.std(z_train, axis=0, keepdims=True) + 1e-8
    
    phi_train = (z_train - z_mean) / z_std
    phi_test = (z_test - z_mean) / z_std

    # Add bias term column
    N_tr = phi_train.shape[0]
    N_te = phi_test.shape[0]
    phi_train_bias = np.concatenate([phi_train, np.ones((N_tr, 1))], axis=1) # [N_tr, d_z + 1]
    phi_test_bias = np.concatenate([phi_test, np.ones((N_te, 1))], axis=1)   # [N_te, d_z + 1]

    # Analytical Ridge Regression: W = (Phi^T Phi + alpha * I)^{-1} Phi^T Y
    p_dim = phi_train_bias.shape[1]
    reg_matrix = ridge_alpha * np.eye(p_dim)
    reg_matrix[-1, -1] = 0.0 # Don't penalize bias

    w = np.linalg.solve(phi_train_bias.T @ phi_train_bias + reg_matrix, phi_train_bias.T @ y_train) # [d_z+1, 4]
    y_test_pred = phi_test_bias @ w # [N_te, 4]

    probe_metrics: Dict[str, ProbeMetric] = {}
    for idx, var_name in enumerate(LATENT_VARIABLES):
        y_true = y_test[:, idx]
        y_hat = y_test_pred[:, idx]

        mae = float(np.mean(np.abs(y_hat - y_true)))
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        
        # R^2 = 1 - SS_res / SS_tot
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        probe_metrics[var_name] = ProbeMetric(
            variable_name=var_name,
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            target_mean=float(np.mean(y_true)),
            target_std=float(np.std(y_true)),
        )

    return LatentProbeReport(
        model_name="PRISM_LatentProbes",
        latent_dim=model.config.model.latent_dim,
        probe_metrics=probe_metrics,
    )
