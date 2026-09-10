"""Multi-Step Forecasting Engines: Open-Loop and Teacher-Forced Rollouts."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from prism.dataset.schema import LearnerEpisode
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.training.metrics import PersistenceBaseline, MLPDynamicsBaseline
from prism.evaluation.rollout_metrics import (
    MultiStepEvaluationReport,
    HorizonEvaluationSummary,
    compute_horizon_metrics,
)


def evaluate_prism_multistep(
    model: CausalWorldModel,
    episodes: List[LearnerEpisode],
    normalizer: ObservationNormalizer,
    horizons: List[int] = [1, 5, 10, 20, 40],
    context_length: int = 40,
    stride: int = 5,
    mode: str = "open_loop",
    deterministic: bool = True,
    device: Optional[torch.device] = None,
) -> MultiStepEvaluationReport:
    """Evaluate PRISM World Model on multi-step forecasting across episodes."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    max_horizon = max(horizons)
    # Collectors per horizon: dict[h -> list of tensors]
    preds_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    targets_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    masks_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    logvars_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}

    std_t = normalizer._std_tensor.to(dev)

    with torch.no_grad():
        for ep in episodes:
            T_ep = len(ep.timestamps)
            if T_ep < context_length + max_horizon:
                continue

            for start_idx in range(0, T_ep - context_length - max_horizon + 1, stride):
                # 1. Historical Context [start_idx : start_idx + context_length]
                raw_ctx_obs = ep.observations[start_idx : start_idx + context_length]
                ctx_mask = ep.observation_mask[start_idx : start_idx + context_length]
                ctx_act = ep.actions[start_idx : start_idx + context_length]

                norm_ctx_obs = normalizer.normalize(torch.tensor(raw_ctx_obs, dtype=torch.float32)).unsqueeze(0).to(dev)
                ctx_mask_t = torch.tensor(ctx_mask, dtype=torch.float32).unsqueeze(0).to(dev)
                ctx_act_t = torch.tensor(ctx_act, dtype=torch.float32).unsqueeze(0).to(dev)

                ctx_inputs = ModelInputs(observations=norm_ctx_obs, observation_mask=ctx_mask_t, actions=ctx_act_t)

                # Future Actions [start_idx + context_length - 1 : start_idx + context_length + max_horizon - 1]
                # Action executed at step t drives state at step t+1
                fut_act_np = ep.actions[start_idx + context_length - 1 : start_idx + context_length + max_horizon - 1]
                fut_act_t = torch.tensor(fut_act_np, dtype=torch.float32).unsqueeze(0).to(dev) # [1, max_horizon, 4]

                # Ground Truth Targets & Masks for all future steps [1 .. max_horizon]
                fut_raw_obs = ep.observations[start_idx + context_length : start_idx + context_length + max_horizon]
                fut_mask = ep.observation_mask[start_idx + context_length : start_idx + context_length + max_horizon]

                if mode == "open_loop":
                    # Encode initial history up to t_star = start_idx + context_length - 1
                    post_latents, _ = model.encode(ctx_inputs)
                    initial_z = post_latents.mean[:, -1] # [1, d_z]

                    # Autoregressive Rollout across future actions
                    rollout_traj = model.rollout_manager.rollout_deterministic(
                        initial_z=initial_z,
                        actions=fut_act_t,
                    )
                    # rollout_traj.observations.mean has shape [1, max_horizon, 8]
                    pred_obs_norm = rollout_traj.observations.mean.squeeze(0) # [max_horizon, 8]
                    pred_var_norm = rollout_traj.observations.variance.squeeze(0)

                    pred_obs_phys = normalizer.denormalize(pred_obs_norm.cpu()).to(dev)
                    pred_var_phys = pred_var_norm * (std_t ** 2)
                    pred_logvar_phys = torch.log(pred_var_phys + 1e-8)

                    for h in horizons:
                        step_idx = h - 1
                        preds_by_h[h].append(pred_obs_phys[step_idx : step_idx + 1])
                        targets_by_h[h].append(torch.tensor(fut_raw_obs[step_idx : step_idx + 1], dtype=torch.float32).to(dev))
                        masks_by_h[h].append(torch.tensor(fut_mask[step_idx : step_idx + 1], dtype=torch.float32).to(dev))
                        logvars_by_h[h].append(pred_logvar_phys[step_idx : step_idx + 1])

                elif mode == "teacher_forced":
                    # In teacher-forced mode, we evaluate each horizon step h by encoding full history up to that step
                    for h in horizons:
                        tf_len = context_length + h - 1
                        tf_raw_obs = ep.observations[start_idx : start_idx + tf_len]
                        tf_mask = ep.observation_mask[start_idx : start_idx + tf_len]
                        tf_act = ep.actions[start_idx : start_idx + tf_len]

                        tf_norm_obs = normalizer.normalize(torch.tensor(tf_raw_obs, dtype=torch.float32)).unsqueeze(0).to(dev)
                        tf_mask_t = torch.tensor(tf_mask, dtype=torch.float32).unsqueeze(0).to(dev)
                        tf_act_t = torch.tensor(tf_act, dtype=torch.float32).unsqueeze(0).to(dev)

                        tf_inputs = ModelInputs(observations=tf_norm_obs, observation_mask=tf_mask_t, actions=tf_act_t)
                        post_latents, _ = model.encode(tf_inputs)

                        # Transition 1 step from last encoded step
                        last_z = post_latents.mean[:, -1:]
                        last_act = tf_act_t[:, -1:]
                        prior_trans = model.transition_step(last_z, last_act)
                        pred_dist = model.decode(prior_trans.mean.squeeze(1))

                        pred_phys = normalizer.denormalize(pred_dist.mean.cpu()).to(dev)
                        pred_var_phys = pred_dist.variance * (std_t ** 2)
                        pred_lv = torch.log(pred_var_phys + 1e-8)

                        step_idx = h - 1
                        preds_by_h[h].append(pred_phys)
                        targets_by_h[h].append(torch.tensor(fut_raw_obs[step_idx : step_idx + 1], dtype=torch.float32).to(dev))
                        masks_by_h[h].append(torch.tensor(fut_mask[step_idx : step_idx + 1], dtype=torch.float32).to(dev))
                        logvars_by_h[h].append(pred_lv)

    # Compute Horizon Summaries
    by_horizon: Dict[int, HorizonEvaluationSummary] = {}
    for h in horizons:
        p_cat = torch.cat(preds_by_h[h], dim=0).cpu()
        t_cat = torch.cat(targets_by_h[h], dim=0).cpu()
        m_cat = torch.cat(masks_by_h[h], dim=0).cpu()
        lv_cat = torch.cat(logvars_by_h[h], dim=0).cpu()

        by_horizon[h] = compute_horizon_metrics(
            predictions_phys=p_cat,
            targets_phys=t_cat,
            mask=m_cat,
            normalizer=normalizer,
            horizon=h,
            pred_logvar_phys=lv_cat,
        )

    return MultiStepEvaluationReport(
        model_name=f"PRISM_{mode}",
        mode=mode,
        horizons=horizons,
        by_horizon=by_horizon,
    )


def evaluate_mlp_multistep(
    mlp_model: MLPDynamicsBaseline,
    episodes: List[LearnerEpisode],
    normalizer: ObservationNormalizer,
    horizons: List[int] = [1, 5, 10, 20, 40],
    context_length: int = 40,
    stride: int = 5,
    device: Optional[torch.device] = None,
) -> MultiStepEvaluationReport:
    """Evaluate Supervised MLP Dynamics baseline autoregressively over multi-step horizons."""
    dev = device or torch.device("cpu")
    mlp_model.to(dev)
    mlp_model.eval()

    max_horizon = max(horizons)
    preds_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    targets_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    masks_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}

    with torch.no_grad():
        for ep in episodes:
            T_ep = len(ep.timestamps)
            if T_ep < context_length + max_horizon:
                continue

            for start_idx in range(0, T_ep - context_length - max_horizon + 1, stride):
                # Last observed observation at context boundary
                last_raw_obs = ep.observations[start_idx + context_length - 1]
                last_norm_obs = normalizer.normalize(torch.tensor(last_raw_obs, dtype=torch.float32)).unsqueeze(0).to(dev) # [1, 8]

                fut_act_np = ep.actions[start_idx + context_length - 1 : start_idx + context_length + max_horizon - 1]
                fut_raw_obs = ep.observations[start_idx + context_length : start_idx + context_length + max_horizon]
                fut_mask = ep.observation_mask[start_idx + context_length : start_idx + context_length + max_horizon]

                curr_obs = last_norm_obs
                for step in range(max_horizon):
                    act_step = torch.tensor(fut_act_np[step : step + 1], dtype=torch.float32).to(dev)
                    next_norm_obs = mlp_model(curr_obs, act_step)
                    curr_obs = next_norm_obs # Autoregressive chaining

                    h = step + 1
                    if h in horizons:
                        pred_phys = normalizer.denormalize(next_norm_obs.cpu()).to(dev)
                        preds_by_h[h].append(pred_phys)
                        targets_by_h[h].append(torch.tensor(fut_raw_obs[step : step + 1], dtype=torch.float32).to(dev))
                        masks_by_h[h].append(torch.tensor(fut_mask[step : step + 1], dtype=torch.float32).to(dev))

    by_horizon: Dict[int, HorizonEvaluationSummary] = {}
    for h in horizons:
        p_cat = torch.cat(preds_by_h[h], dim=0).cpu()
        t_cat = torch.cat(targets_by_h[h], dim=0).cpu()
        m_cat = torch.cat(masks_by_h[h], dim=0).cpu()

        by_horizon[h] = compute_horizon_metrics(
            predictions_phys=p_cat,
            targets_phys=t_cat,
            mask=m_cat,
            normalizer=normalizer,
            horizon=h,
        )

    return MultiStepEvaluationReport(
        model_name="MLP_Dynamics_open_loop",
        mode="open_loop",
        horizons=horizons,
        by_horizon=by_horizon,
    )


def evaluate_persistence_multistep(
    episodes: List[LearnerEpisode],
    normalizer: ObservationNormalizer,
    horizons: List[int] = [1, 5, 10, 20, 40],
    context_length: int = 40,
    stride: int = 5,
) -> MultiStepEvaluationReport:
    """Evaluate Persistence Baseline (O_hat_{t+h} = O_t) over multi-step horizons."""
    max_horizon = max(horizons)
    preds_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    targets_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}
    masks_by_h: Dict[int, List[Tensor]] = {h: [] for h in horizons}

    for ep in episodes:
        T_ep = len(ep.timestamps)
        if T_ep < context_length + max_horizon:
            continue

        for start_idx in range(0, T_ep - context_length - max_horizon + 1, stride):
            last_raw_obs = ep.observations[start_idx + context_length - 1]
            fut_raw_obs = ep.observations[start_idx + context_length : start_idx + context_length + max_horizon]
            fut_mask = ep.observation_mask[start_idx + context_length : start_idx + context_length + max_horizon]

            for h in horizons:
                step_idx = h - 1
                preds_by_h[h].append(torch.tensor(last_raw_obs, dtype=torch.float32).unsqueeze(0))
                targets_by_h[h].append(torch.tensor(fut_raw_obs[step_idx : step_idx + 1], dtype=torch.float32))
                masks_by_h[h].append(torch.tensor(fut_mask[step_idx : step_idx + 1], dtype=torch.float32))

    by_horizon: Dict[int, HorizonEvaluationSummary] = {}
    for h in horizons:
        p_cat = torch.cat(preds_by_h[h], dim=0)
        t_cat = torch.cat(targets_by_h[h], dim=0)
        m_cat = torch.cat(masks_by_h[h], dim=0)

        by_horizon[h] = compute_horizon_metrics(
            predictions_phys=p_cat,
            targets_phys=t_cat,
            mask=m_cat,
            normalizer=normalizer,
            horizon=h,
        )

    return MultiStepEvaluationReport(
        model_name="Persistence_open_loop",
        mode="open_loop",
        horizons=horizons,
        by_horizon=by_horizon,
    )
