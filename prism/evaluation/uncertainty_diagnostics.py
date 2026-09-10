"""Diagnostic Subsystem for Investigating OOD Uncertainty and Latent Distribution Shifts.

Calculates:
1. Latent posterior statistics: mean norm, std norm, per-dimension variance.
2. Latent distribution shift metrics relative to In-Distribution (ID) reference:
   - Euclidean distance of latent means ||mu_z - mu_ID||_2
   - Mahalanobis distance d_M(mu_z; mu_ID, Sigma_ID)
3. Particle disagreement in latent space Var(Z_{t+h}) and observable space Var(O_{t+h}).
4. Error vs Latent Distance correlation r(|e|, d_M).
5. Error vs Particle Disagreement correlation r(|e|, sigma_particles).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from scipy.stats import spearmanr

from prism.dataset.schema import LearnerEpisode
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.world_model.latent_state import LatentDistribution
from prism.training.normalization import ObservationNormalizer


@dataclass
class RegimeDiagnosticSummary:
    """Detailed diagnostic metrics for a specific dataset regime."""

    regime_name: str
    num_episodes: int
    forecast_error_h40: float          # Train-normalized MAE at h=40
    raw_error_h40: float               # Raw physical MAE at h=40
    latent_mean_norm: float            # Mean ||mu_z||_2
    latent_std_norm: float             # Mean ||sigma_z||_2
    latent_euclidean_distance: float   # Mean Euclidean distance from ID center
    latent_mahalanobis_distance: float # Mean Mahalanobis distance from ID distribution
    particle_latent_variance_h1: float # Mean Var(Z) across 50 particles at h=1
    particle_latent_variance_h40: float# Mean Var(Z) across 50 particles at h=40
    particle_obs_sigma_h1: float       # Mean observable predictive sigma at h=1
    particle_obs_sigma_h40: float      # Mean observable predictive sigma at h=40
    error_latent_distance_corr: float  # Pearson r(|error_h40|, d_Mahalanobis)
    error_particle_sigma_corr: float   # Pearson r(|error_h40|, sigma_obs_h40)


@dataclass
class UncertaintyDiagnosticReport:
    """Comprehensive diagnostic report investigating OOD latent shifts and uncertainty failure modes."""

    model_name: str
    num_particles: int
    id_reference_regime: str
    regime_diagnostics: Dict[str, RegimeDiagnosticSummary] = field(default_factory=dict)
    diagnosis_verdict: str = ""
    diagnosis_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_particles": self.num_particles,
            "id_reference_regime": self.id_reference_regime,
            "regime_diagnostics": {k: asdict(v) for k, v in self.regime_diagnostics.items()},
            "diagnosis_verdict": self.diagnosis_verdict,
            "diagnosis_rationale": self.diagnosis_rationale,
        }


def compute_uncertainty_diagnostics(
    model: CausalWorldModel,
    train_episodes: List[LearnerEpisode],
    test_episodes: List[LearnerEpisode],
    ood_regimes: Dict[str, List[LearnerEpisode]],
    normalizer: ObservationNormalizer,
    num_particles: int = 50,
    context_length: int = 40,
    stride: int = 5,
    horizon: int = 40,
    device: Optional[torch.device] = None,
) -> UncertaintyDiagnosticReport:
    """Run comprehensive latent distribution and uncertainty diagnostics across ID and OOD regimes."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    std_t = normalizer._std_tensor.to(dev)
    mean_t = normalizer._mean_tensor.to(dev)

    # 1. Extract Training Set Latent Representation to Compute ID Reference Statistics
    train_latents = []
    with torch.no_grad():
        for ep in train_episodes:
            T_ep = len(ep.timestamps)
            if T_ep < context_length:
                continue
            for s_idx in range(0, T_ep - context_length + 1, stride):
                raw_obs = ep.observations[s_idx : s_idx + context_length]
                mask = ep.observation_mask[s_idx : s_idx + context_length]
                act = ep.actions[s_idx : s_idx + context_length]

                norm_obs = normalizer.normalize(torch.tensor(raw_obs, dtype=torch.float32)).unsqueeze(0).to(dev)
                mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(dev)
                act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(dev)

                inputs = ModelInputs(norm_obs, mask_t, act_t)
                post, _ = model.encode(inputs)
                train_latents.append(post.mean[:, -1].cpu().numpy())

    z_train_arr = np.concatenate(train_latents, axis=0) # [N_train, d_z]
    mu_id = np.mean(z_train_arr, axis=0)                # [d_z]
    cov_id = np.cov(z_train_arr, rowvar=False)          # [d_z, d_z]

    # Regularized inverse covariance for Mahalanobis distance
    d_z = z_train_arr.shape[1]
    reg_cov = cov_id + 1e-4 * np.eye(d_z)
    inv_cov_id = np.linalg.inv(reg_cov)

    # 2. Evaluate Diagnostics for Each Regime
    all_regimes = {"Test_InDistribution": test_episodes}
    all_regimes.update(ood_regimes)

    regime_summaries: Dict[str, RegimeDiagnosticSummary] = {}

    for reg_name, eps in all_regimes.items():
        all_z_means = []
        all_z_stds = []
        all_d_euc = []
        all_d_mah = []
        all_errors_norm_h40 = []
        all_errors_raw_h40 = []
        all_z_var_particles_h1 = []
        all_z_var_particles_h40 = []
        all_obs_sigma_h1 = []
        all_obs_sigma_h40 = []

        with torch.no_grad():
            for ep in eps:
                T_ep = len(ep.timestamps)
                if T_ep < context_length + horizon:
                    continue

                for s_idx in range(0, T_ep - context_length - horizon + 1, stride):
                    raw_ctx = ep.observations[s_idx : s_idx + context_length]
                    mask_ctx = ep.observation_mask[s_idx : s_idx + context_length]
                    act_ctx = ep.actions[s_idx : s_idx + context_length]

                    norm_ctx = normalizer.normalize(torch.tensor(raw_ctx, dtype=torch.float32)).unsqueeze(0).to(dev)
                    m_t = torch.tensor(mask_ctx, dtype=torch.float32).unsqueeze(0).to(dev)
                    a_t = torch.tensor(act_ctx, dtype=torch.float32).unsqueeze(0).to(dev)

                    ctx_in = ModelInputs(norm_ctx, m_t, a_t)
                    post, _ = model.encode(ctx_in)

                    z_mean_step = post.mean[:, -1].cpu().numpy().squeeze(0) # [d_z]
                    z_std_step = post.std[:, -1].cpu().numpy().squeeze(0)   # [d_z]

                    all_z_means.append(np.linalg.norm(z_mean_step))
                    all_z_stds.append(np.linalg.norm(z_std_step))

                    # Distance from ID Center
                    diff_id = z_mean_step - mu_id
                    d_euc = float(np.linalg.norm(diff_id))
                    d_mah = float(np.sqrt(np.dot(np.dot(diff_id, inv_cov_id), diff_id)))

                    all_d_euc.append(d_euc)
                    all_d_mah.append(d_mah)

                    # Future action sequence [1, H, 4]
                    fut_act_np = ep.actions[s_idx + context_length - 1 : s_idx + context_length + horizon - 1]
                    fut_act_t = torch.tensor(fut_act_np, dtype=torch.float32).unsqueeze(0).to(dev)

                    fut_raw_obs = ep.observations[s_idx + context_length : s_idx + context_length + horizon]
                    fut_mask = ep.observation_mask[s_idx + context_length : s_idx + context_length + horizon]

                    # Monte Carlo Particle Rollout
                    z_dist = LatentDistribution(
                        mean=post.mean[:, -1],
                        logvar=post.logvar[:, -1],
                        min_std=post.min_std,
                        max_std=post.max_std,
                    )

                    mc_traj = model.rollout_manager.rollout_monte_carlo(
                        initial_z_dist=z_dist,
                        actions=fut_act_t,
                        num_samples=num_particles,
                    )

                    # mc_traj.latent_variance: [1, H, d_z]
                    z_var_h1 = float(torch.mean(mc_traj.latent_variance[0, 0]).item())
                    z_var_h40 = float(torch.mean(mc_traj.latent_variance[0, -1]).item())
                    all_z_var_particles_h1.append(z_var_h1)
                    all_z_var_particles_h40.append(z_var_h40)

                    # Observable predictive standard deviation in normalized space
                    obs_var_norm = mc_traj.observations.variance.squeeze(0) # [H, 8]
                    obs_std_norm = torch.sqrt(obs_var_norm + 1e-8)
                    all_obs_sigma_h1.append(float(torch.mean(obs_std_norm[0]).item()))
                    all_obs_sigma_h40.append(float(torch.mean(obs_std_norm[-1]).item()))

                    # Point Forecast Error at h=40
                    pred_obs_norm_h40 = mc_traj.observations.mean[0, -1] # [8]
                    pred_obs_phys_h40 = normalizer.denormalize(pred_obs_norm_h40.cpu()).to(dev)

                    target_phys_h40 = torch.tensor(fut_raw_obs[-1], dtype=torch.float32).to(dev)
                    mask_h40 = torch.tensor(fut_mask[-1], dtype=torch.float32).to(dev).bool()

                    # Train-normalized error
                    norm_target_h40 = (target_phys_h40 - mean_t) / std_t
                    norm_diff = torch.abs(pred_obs_norm_h40 - norm_target_h40)[mask_h40]
                    norm_mae = float(torch.mean(norm_diff).item()) if norm_diff.numel() > 0 else 0.0

                    raw_diff = torch.abs(pred_obs_phys_h40 - target_phys_h40)[mask_h40]
                    raw_mae = float(torch.mean(raw_diff).item()) if raw_diff.numel() > 0 else 0.0

                    all_errors_norm_h40.append(norm_mae)
                    all_errors_raw_h40.append(raw_mae)

        err_arr = np.array(all_errors_norm_h40)
        mah_arr = np.array(all_d_mah)
        sig_arr = np.array(all_obs_sigma_h40)

        corr_mah = float(np.corrcoef(err_arr, mah_arr)[0, 1]) if len(err_arr) > 1 and np.std(err_arr) > 1e-7 and np.std(mah_arr) > 1e-7 else 0.0
        if np.isnan(corr_mah):
            corr_mah = 0.0

        corr_sig = float(np.corrcoef(err_arr, sig_arr)[0, 1]) if len(err_arr) > 1 and np.std(err_arr) > 1e-7 and np.std(sig_arr) > 1e-7 else 0.0
        if np.isnan(corr_sig):
            corr_sig = 0.0

        regime_summaries[reg_name] = RegimeDiagnosticSummary(
            regime_name=reg_name,
            num_episodes=len(eps),
            forecast_error_h40=float(np.mean(all_errors_norm_h40)),
            raw_error_h40=float(np.mean(all_errors_raw_h40)),
            latent_mean_norm=float(np.mean(all_z_means)),
            latent_std_norm=float(np.mean(all_z_stds)),
            latent_euclidean_distance=float(np.mean(all_d_euc)),
            latent_mahalanobis_distance=float(np.mean(all_d_mah)),
            particle_latent_variance_h1=float(np.mean(all_z_var_particles_h1)),
            particle_latent_variance_h40=float(np.mean(all_z_var_particles_h40)),
            particle_obs_sigma_h1=float(np.mean(all_obs_sigma_h1)),
            particle_obs_sigma_h40=float(np.mean(all_obs_sigma_h40)),
            error_latent_distance_corr=corr_mah,
            error_particle_sigma_corr=corr_sig,
        )

    # Determine Diagnostic Verdict
    id_mah = regime_summaries["Test_InDistribution"].latent_mahalanobis_distance
    ood1_mah = regime_summaries.get("ood_1_extreme_ambient", regime_summaries["Test_InDistribution"]).latent_mahalanobis_distance
    ood3_mah = regime_summaries.get("ood_3_combined_stress", regime_summaries["Test_InDistribution"]).latent_mahalanobis_distance

    id_sig = regime_summaries["Test_InDistribution"].particle_obs_sigma_h40
    ood1_sig = regime_summaries.get("ood_1_extreme_ambient", regime_summaries["Test_InDistribution"]).particle_obs_sigma_h40

    latent_shifted = (ood1_mah > id_mah * 1.3) or (ood3_mah > id_mah * 1.3)
    sigma_flat = abs(ood1_sig - id_sig) < 0.1

    if latent_shifted and sigma_flat:
        verdict = "Case A: Latent Representation Shifts Strongly, but Single-Model Transition/Decoder Predictive Variance Remains Flat"
        rationale = (
            "Under severe OOD regimes (OOD-1, OOD-3), the latent state mean shifts significantly in Mahalanobis space, "
            "proving the encoder detects the domain shift. However, because a single deterministic/Gaussian transition network "
            "outputs learned aleatoric noise rather than epistemic parameter uncertainty, predictive particle sigma does not expand."
        )
    elif not latent_shifted:
        verdict = "Case B: Latent Representation Barely Shifts"
        rationale = "The encoder maps OOD inputs to standard in-distribution latent regions."
    else:
        verdict = "Case D: Single Stochastic Model Mechanism Inadequacy"
        rationale = "Predictive variance is purely transition-stochastic and lacks ensemble/epistemic distance mechanisms."

    return UncertaintyDiagnosticReport(
        model_name="PRISM_Uncertainty_Diagnostics",
        num_particles=num_particles,
        id_reference_regime="Test_InDistribution",
        regime_diagnostics=regime_summaries,
        diagnosis_verdict=verdict,
        diagnosis_rationale=rationale,
    )
