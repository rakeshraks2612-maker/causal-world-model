"""Model Trust, Anomaly Detection, and Abstention Decision Subsystem (Task 5.10E).

Defines explicit model confidence states:
- MODEL_TRUSTED: In-distribution, low reconstruction residual, confident predictions.
- MODEL_UNCERTAIN: Marginal residual or elevated epistemic uncertainty; planner applies risk penalty.
- MODEL_ABSTAIN: Significant latent distribution shift OR observation inconsistency;
  PRISM rigorously abstains and refrains from trust-violating intervention.

Semantics:
IF D_latent(z_t*) > tau_novelty OR R_T(t*) > tau_residual:
    ABSTAIN (do not trust model-based forecast)
ELSE:
    MODEL_TRUSTED -> proceed to authoritative safety constraints & planner
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


class ModelTrustState(str, Enum):
    """Discrete trust states for PRISM decision intelligence."""
    MODEL_TRUSTED = "model_trusted"
    MODEL_UNCERTAIN = "model_uncertain"
    MODEL_ABSTAIN = "model_abstain"


@dataclass
class TrustDiagnostics:
    """Quantitative trust and anomaly metrics computed at decision point t*."""
    state: ModelTrustState
    residual_t_core: float            # Physical |T_core - T_core_hat| (°C)
    residual_8d_norm: float           # Normalized ||O_norm - O_norm_hat||_2
    latent_mahalanobis_d: float       # d_M(z_t*; mu_ID, Sigma_ID)
    epistemic_sigma: float            # Ensemble / transition variance
    aleatoric_sigma: float            # Posterior decoder / latent logvar
    is_novel_latent: bool
    is_reconstruction_inconsistent: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "residual_t_core": self.residual_t_core,
            "residual_8d_norm": self.residual_8d_norm,
            "latent_mahalanobis_d": self.latent_mahalanobis_d,
            "epistemic_sigma": self.epistemic_sigma,
            "aleatoric_sigma": self.aleatoric_sigma,
            "is_novel_latent": self.is_novel_latent,
            "is_reconstruction_inconsistent": self.is_reconstruction_inconsistent,
            "rationale": self.rationale,
        }


class ModelTrustEvaluator:
    """Evaluates online model trust and anomaly signals at decision points."""

    def __init__(
        self,
        model: CausalWorldModel,
        normalizer: ObservationNormalizer,
        mu_id: np.ndarray,
        inv_cov_id: np.ndarray,
        tau_residual_t: float = 13.86,   # Calibrated P98 residual threshold (°C)
        tau_residual_8d: float = 1.80,   # Calibrated P98 8D residual threshold
        tau_novelty: float = 15.0,       # Latent Mahalanobis novelty threshold
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.mu_id = mu_id
        self.inv_cov_id = inv_cov_id
        self.tau_residual_t = tau_residual_t
        self.tau_residual_8d = tau_residual_8d
        self.tau_novelty = tau_novelty

    def evaluate_trust(
        self,
        historical_obs: np.ndarray,          # [T, 8] raw physical observations
        historical_mask: np.ndarray,         # [T, 8] boolean mask
        historical_act: np.ndarray,          # [T, 4] historical actions
        t_star: int,
    ) -> TrustDiagnostics:
        """Compute decision-point trust state and anomaly residuals."""
        self.model.eval()
        
        obs_norm = self.normalizer.normalize(historical_obs)
        if not isinstance(obs_norm, torch.Tensor):
            obs_norm = torch.from_numpy(obs_norm)
            
        t_obs = obs_norm.float().unsqueeze(0)
        t_mask = torch.from_numpy(historical_mask).bool().unsqueeze(0)
        t_act = torch.from_numpy(historical_act).float().unsqueeze(0)
        
        inputs = ModelInputs(t_obs, t_mask, t_act)
        
        with torch.no_grad():
            lat_dist, _ = self.model.encode(inputs)
            z_seq = lat_dist.mean.squeeze(0)             # [T, 64]
            logvar_seq = lat_dist.logvar.squeeze(0)       # [T, 64]
            
            rec_dist = self.model.decode(lat_dist.mean)
            rec_norm = rec_dist.mean.squeeze(0)          # [T, 8]
            rec_raw_t = self.normalizer.denormalize(rec_norm)
            rec_raw = rec_raw_t.cpu().numpy() if isinstance(rec_raw_t, torch.Tensor) else np.asarray(rec_raw_t)
            
        z_star = z_seq[t_star].cpu().numpy()
        
        # 1. Latent Mahalanobis Distance
        diff = z_star - self.mu_id
        d2 = float(diff @ self.inv_cov_id @ diff)
        d_mah = float(np.sqrt(max(0.0, d2)))
        
        # 2. Online Reconstruction Residual
        r_T = float(np.abs(historical_obs[t_star, 0] - rec_raw[t_star, 0]))
        r_8d = float(np.linalg.norm(obs_norm.numpy()[t_star] - rec_norm.cpu().numpy()[t_star]))
        
        # 3. Uncertainties
        aleatoric = float(torch.exp(0.5 * logvar_seq[t_star]).mean().item())
        epistemic = float(np.std(z_star)) # Latent dispersion proxy
        
        # 4. Gating Decisions
        is_novel = bool(d_mah > self.tau_novelty)
        is_inconsistent = bool(r_T > self.tau_residual_t or r_8d > self.tau_residual_8d)
        
        if is_novel and is_inconsistent:
            state = ModelTrustState.MODEL_ABSTAIN
            rationale = (
                f"Dual failure: Severe latent novelty (d_M = {d_mah:.2f} > {self.tau_novelty:.1f}) "
                f"AND extreme reconstruction residual (R_T = {r_T:.2f}°C > {self.tau_residual_t:.1f}°C). "
                "System state is severely unmodeled; model predictions cannot be trusted."
            )
        elif is_inconsistent:
            state = ModelTrustState.MODEL_ABSTAIN
            if r_T > self.tau_residual_t and r_8d > self.tau_residual_8d:
                sub_detail = f"R_T = {r_T:.2f}°C > {self.tau_residual_t:.1f}°C, R_8D = {r_8d:.2f} > {self.tau_residual_8d:.2f}"
            elif r_T > self.tau_residual_t:
                sub_detail = f"R_T = {r_T:.2f}°C > {self.tau_residual_t:.1f}°C (R_8D = {r_8d:.2f} <= {self.tau_residual_8d:.2f})"
            else:
                sub_detail = f"R_8D = {r_8d:.2f} > {self.tau_residual_8d:.2f} (R_T = {r_T:.2f}°C <= {self.tau_residual_t:.1f}°C)"
            rationale = (
                f"Reconstruction inconsistency: Telemetry contradicts internal world-model dynamics ({sub_detail}). "
                "Acute sensor/physics mismatch detected; triggering hard safety abstention."
            )
        elif is_novel:
            state = ModelTrustState.MODEL_ABSTAIN
            rationale = (
                f"Latent novelty: Inferred state is OOD (d_M = {d_mah:.2f} > {self.tau_novelty:.1f}). "
                "Forecast variance exceeds safe bounds; triggering safety abstention."
            )
        elif r_T > 0.5 * self.tau_residual_t or d_mah > 0.7 * self.tau_novelty:
            state = ModelTrustState.MODEL_UNCERTAIN
            rationale = "Marginal residual or elevated uncertainty. Model predictions should carry higher risk penalty."
        else:
            state = ModelTrustState.MODEL_TRUSTED
            rationale = "Nominal in-distribution telemetry. Model predictions validated for planning."
            
        return TrustDiagnostics(
            state=state,
            residual_t_core=r_T,
            residual_8d_norm=r_8d,
            latent_mahalanobis_d=d_mah,
            epistemic_sigma=epistemic,
            aleatoric_sigma=aleatoric,
            is_novel_latent=is_novel,
            is_reconstruction_inconsistent=is_inconsistent,
            rationale=rationale,
        )
