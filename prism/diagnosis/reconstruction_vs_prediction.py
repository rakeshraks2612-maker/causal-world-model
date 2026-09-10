"""Diagnostic Tool for Isolating Autoencoder Reconstruction vs Latent Transition Prediction."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.diagnosis.domain_metrics import ComprehensiveEvaluationReport, compute_comprehensive_metrics


def evaluate_reconstruction_vs_prediction(
    model: CausalWorldModel,
    dataloader: DataLoader,
    normalizer: ObservationNormalizer,
    deterministic: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[ComprehensiveEvaluationReport, ComprehensiveEvaluationReport]:
    """Evaluate both pure reconstruction (Z_t -> O_t) and transition prediction (Z_t + A_t -> Z_{t+1} -> O_{t+1}).
    
    Args:
        model: Trained CausalWorldModel
        dataloader: Test or Validation DataLoader
        normalizer: Train-set fitted ObservationNormalizer
        deterministic: If True, use mean latent states; if False, sample stochastically
        device: Target execution device
        
    Returns:
        (reconstruction_report, prediction_report)
    """
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    # Collectors for pure reconstruction (t = 0..T-1)
    recon_preds_raw = []
    recon_targets_raw = []
    recon_masks = []
    recon_logvar_raw = []

    # Collectors for transition prediction (t = 1..T-1 from state at t-1)
    pred_preds_raw = []
    pred_targets_raw = []
    pred_masks = []
    pred_logvar_raw = []

    with torch.no_grad():
        for batch in dataloader:
            inputs: ModelInputs = batch["inputs"]
            raw_obs = batch["raw_observations"].to(dev) # [B, T, 8]
            raw_mask = batch["raw_mask"].to(dev)        # [B, T, 8]

            obs = inputs.observations.to(dev)
            mask = inputs.observation_mask.to(dev)
            act = inputs.actions.to(dev)
            dev_inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

            # 1. Variational Posterior over history: q(Z_t | O_<=t, A_<=t) for all t=0..T-1
            post_latents, _ = model.encode(dev_inputs)
            
            if deterministic:
                z_samples = post_latents.mean
            else:
                z_samples = post_latents.sample(deterministic=False)

            # --- PART A: Pure Autoencoding Reconstruction (Z_t -> O_t) ---
            recon_dist = model.decode(z_samples) # [B, T, 8]
            recon_denorm = normalizer.denormalize(recon_dist.mean.cpu()).to(dev)
            
            std_t = normalizer._std_tensor.to(dev)
            recon_var_denorm = recon_dist.variance * (std_t ** 2)
            recon_logvar_denorm = torch.log(recon_var_denorm + 1e-8)

            recon_preds_raw.append(recon_denorm.reshape(-1, 8))
            recon_targets_raw.append(raw_obs.reshape(-1, 8))
            recon_masks.append(raw_mask.reshape(-1, 8))
            recon_logvar_raw.append(recon_logvar_denorm.reshape(-1, 8))

            # --- PART B: Latent Transition Prediction (Z_t + A_t -> Z_{t+1} -> O_{t+1}) ---
            z_curr = z_samples[:, :-1]     # [B, T-1, d_z]
            act_curr = act[:, :-1]        # [B, T-1, 4]
            prior_trans = model.transition_step(z_curr, act_curr)

            if deterministic:
                z_next = prior_trans.mean
            else:
                z_next = prior_trans.sample(deterministic=False)

            pred_dist = model.decode(z_next) # [B, T-1, 8]
            pred_denorm = normalizer.denormalize(pred_dist.mean.cpu()).to(dev)

            pred_var_denorm = pred_dist.variance * (std_t ** 2)
            pred_logvar_denorm = torch.log(pred_var_denorm + 1e-8)

            pred_preds_raw.append(pred_denorm.reshape(-1, 8))
            pred_targets_raw.append(raw_obs[:, 1:].reshape(-1, 8))
            pred_masks.append(raw_mask[:, 1:].reshape(-1, 8))
            pred_logvar_raw.append(pred_logvar_denorm.reshape(-1, 8))

    # Compute Comprehensive Reports
    mode_str = "Deterministic" if deterministic else "Stochastic"
    
    recon_report = compute_comprehensive_metrics(
        predictions_phys=torch.cat(recon_preds_raw, dim=0).cpu(),
        targets_phys=torch.cat(recon_targets_raw, dim=0).cpu(),
        mask=torch.cat(recon_masks, dim=0).cpu(),
        normalizer=normalizer,
        pred_logvar_phys=torch.cat(recon_logvar_raw, dim=0).cpu(),
        model_name=f"PRISM_Reconstruction_{mode_str}",
    )

    pred_report = compute_comprehensive_metrics(
        predictions_phys=torch.cat(pred_preds_raw, dim=0).cpu(),
        targets_phys=torch.cat(pred_targets_raw, dim=0).cpu(),
        mask=torch.cat(pred_masks, dim=0).cpu(),
        normalizer=normalizer,
        pred_logvar_phys=torch.cat(pred_logvar_raw, dim=0).cpu(),
        model_name=f"PRISM_1StepPrediction_{mode_str}",
    )

    return recon_report, pred_report
