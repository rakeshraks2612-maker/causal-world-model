"""Learned Intervention Simulator for PRISM Causal World Models.

Implements Pearl's Level 2 do-calculus on learned latent representations:
1. Incurs historical observations (O_<=t*, A_<=t*)
2. Infers hidden physical state Z_t* ~ q_phi(Z_t* | O_<=t*, A_<=t*)
3. Executes factual baseline rollout: Z_t* -> Y_base
4. Executes counterfactual/intervened rollout: Z_t* + do(X=x) -> Y_int
5. Computes multi-horizon causal effect deltas Delta Y and safety metrics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.model import CausalWorldModel
from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import LatentDistribution, ObservationDistribution
from prism.world_model.rollout import RolloutTrajectory
from prism.training.normalization import ObservationNormalizer
from prism.intervention.spec import InterventionSpec
from prism.intervention.operator import InterventionOperator
from prism.intervention.effects import CausalEffectSummary, compute_causal_effects
from prism.dataset.interventions import LearnerInterventionRecord


@dataclass
class LearnedInterventionResult:
    """Complete result of a learned intervention simulation."""

    intervention_id: str
    target: str
    value: float
    intervention_time: int
    duration: Optional[int]
    baseline_observations: np.ndarray       # Shape: [H, 8] in physical units
    intervened_observations: np.ndarray     # Shape: [H, 8] in physical units
    baseline_latent_mean: np.ndarray        # Shape: [H, d_z]
    intervened_latent_mean: np.ndarray      # Shape: [H, d_z]
    inferred_z_t_star: np.ndarray           # Shape: [d_z]
    inferred_z_logvar: np.ndarray           # Shape: [d_z]
    effects: CausalEffectSummary
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "target": self.target,
            "value": self.value,
            "intervention_time": self.intervention_time,
            "duration": self.duration,
            "effects": self.effects.to_dict(),
            "horizons": self.horizons,
            "metadata": self.metadata,
        }


class LearnedInterventionSimulator:
    """Simulator that executes causal interventions on top of a trained PRISM world model."""

    def __init__(
        self,
        world_model: CausalWorldModel,
        normalizer: ObservationNormalizer,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.world_model = world_model.to(self.device)
        self.world_model.eval()
        self.normalizer = normalizer

    def simulate(
        self,
        pre_observations: np.ndarray | Tensor,
        pre_actions: np.ndarray | Tensor,
        future_actions: np.ndarray | Tensor,
        intervention: Optional[Union[InterventionSpec, List[InterventionSpec]]] = None,
        intervention_time: Optional[int] = None,
        observation_mask: Optional[np.ndarray | Tensor] = None,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
        deterministic: bool = True,
        num_particles: int = 50,
        intervention_id: Optional[str] = None,
    ) -> LearnedInterventionResult:
        """Run paired factual baseline and counterfactual intervention simulations from identical latent state.
        
        Args:
            pre_observations: Historical observations [t*+1, 8] in physical units
            pre_actions: Historical actions [t*+1, 4]
            future_actions: Future actions [H_fut, 4] starting at step t*+1
            intervention: InterventionSpec or list of specs, or None (no-op)
            intervention_time: Timestep t* (defaults to len(pre_observations) - 1)
            observation_mask: Optional boolean mask [t*+1, 8]
            horizons: Evaluation horizons
            deterministic: If True, propagate mean states deterministically
            num_particles: Particle count for Monte Carlo rollouts
            intervention_id: Identifier string
            
        Returns:
            LearnedInterventionResult containing both trajectories and computed causal effects
        """
        # 1. Prepare history tensors
        if isinstance(pre_observations, np.ndarray):
            pre_obs_t = torch.tensor(pre_observations, dtype=torch.float32)
        else:
            pre_obs_t = pre_observations.clone().float()

        if isinstance(pre_actions, np.ndarray):
            pre_act_t = torch.tensor(pre_actions, dtype=torch.float32)
        else:
            pre_act_t = pre_actions.clone().float()

        t_pre = pre_obs_t.shape[0] if pre_obs_t.ndim == 2 else pre_obs_t.shape[1]
        t_star = intervention_time if intervention_time is not None else (t_pre - 1)

        if observation_mask is None:
            mask_t = ~torch.isnan(pre_obs_t)
        elif isinstance(observation_mask, np.ndarray):
            mask_t = torch.tensor(observation_mask, dtype=torch.bool)
        else:
            mask_t = observation_mask.clone().bool()

        if pre_obs_t.ndim == 2:
            pre_obs_t = pre_obs_t.unsqueeze(0)
            pre_act_t = pre_act_t.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0)

        # Normalize historical observations
        norm_pre_obs = self.normalizer.normalize(pre_obs_t).to(self.device)
        mask_t = mask_t.to(self.device).float()
        pre_act_t = pre_act_t.to(self.device)

        # 2. Build rollout action sequence: Action at t* + future actions
        if isinstance(future_actions, np.ndarray):
            fut_act_t = torch.tensor(future_actions, dtype=torch.float32)
        else:
            fut_act_t = future_actions.clone().float()

        if fut_act_t.ndim == 2:
            fut_act_t = fut_act_t.unsqueeze(0)  # [1, H_fut, 4]

        # Action at t* from historical actions
        act_at_t_star = pre_act_t[:, -1:, :]  # [1, 1, 4]
        full_rollout_actions = torch.cat([act_at_t_star, fut_act_t.to(self.device)], dim=1)  # [1, 1 + H_fut, 4]
        horizon_len = full_rollout_actions.shape[1]

        # 3. Infer latent state Z_t* from observable history
        history_inputs = ModelInputs(
            observations=norm_pre_obs,
            observation_mask=mask_t,
            actions=pre_act_t,
        )

        with torch.no_grad():
            post_latents, _ = self.world_model.encode(history_inputs)
            # Latent distribution at t*
            z_t_star_mean = post_latents.mean[:, -1]      # [1, d_z]
            z_t_star_logvar = post_latents.logvar[:, -1]  # [1, d_z]
            z_t_star_dist = LatentDistribution(
                mean=z_t_star_mean,
                logvar=z_t_star_logvar,
                min_std=self.world_model.config.model.min_std,
                max_std=self.world_model.config.model.max_std,
            )

            # 4. Factual Baseline Rollout (unintervened)
            if deterministic:
                base_traj = self.world_model.rollout_manager.rollout_deterministic(
                    initial_z=z_t_star_mean,
                    actions=full_rollout_actions,
                )
                base_obs_norm = base_traj.observations.mean  # [1, H, 8]
            else:
                base_traj = self.world_model.rollout_manager.rollout_monte_carlo(
                    initial_z_dist=z_t_star_dist,
                    actions=full_rollout_actions,
                    num_samples=num_particles,
                )
                base_obs_norm = base_traj.observations.mean  # [1, H, 8]

            base_obs_phys = self.normalizer.denormalize(base_obs_norm.cpu()).numpy()[0]  # [H, 8]
            base_z_mean = base_traj.latent_mean.cpu().numpy()[0]                        # [H, d_z]

            # 5. Counterfactual Intervened Rollout
            specs_list: List[InterventionSpec] = []
            if intervention is not None:
                if isinstance(intervention, InterventionSpec):
                    specs_list = [intervention]
                else:
                    specs_list = list(intervention)

            operator = InterventionOperator(specs_list)

            # Apply action modifications starting at t*
            intervened_actions = operator.get_modified_actions(full_rollout_actions, t_star=t_star).to(self.device)

            if deterministic:
                int_traj = self.world_model.rollout_manager.rollout_deterministic(
                    initial_z=z_t_star_mean,  # IDENTICAL INITIAL LATENT STATE!
                    actions=intervened_actions,
                )
                int_obs_norm = int_traj.observations.mean  # [1, H, 8]
            else:
                int_traj = self.world_model.rollout_manager.rollout_monte_carlo(
                    initial_z_dist=z_t_star_dist,  # IDENTICAL INITIAL LATENT STATE!
                    actions=intervened_actions,
                    num_samples=num_particles,
                )
                int_obs_norm = int_traj.observations.mean  # [1, H, 8]

            int_obs_phys_t = self.normalizer.denormalize(int_obs_norm.cpu())  # [1, H, 8]

            # Apply state clamps in physical observation space
            int_obs_phys_t = operator.apply_observation_clamps(
                int_obs_phys_t,
                t_star=t_star + 1,  # Rollout step 0 corresponds to t* + 1
                normalizer=None,
                is_normalized=False,
            )
            int_obs_phys = int_obs_phys_t.numpy()[0]  # [H, 8]
            int_z_mean = int_traj.latent_mean.cpu().numpy()[0]

        # 6. Compute Causal Effect Deltas
        effects = compute_causal_effects(
            baseline_obs=base_obs_phys,
            intervened_obs=int_obs_phys,
            horizons=horizons,
            t_star=t_star,
        )

        inv_id = intervention_id or (
            f"inv_t{t_star}_{specs_list[0].target}_{int(specs_list[0].value)}"
            if specs_list else f"noop_t{t_star}"
        )
        target_name = specs_list[0].target if specs_list else "none"
        target_val = specs_list[0].value if specs_list else 0.0
        dur = specs_list[0].duration if specs_list else None

        return LearnedInterventionResult(
            intervention_id=inv_id,
            target=target_name,
            value=target_val,
            intervention_time=t_star,
            duration=dur,
            baseline_observations=base_obs_phys,
            intervened_observations=int_obs_phys,
            baseline_latent_mean=base_z_mean,
            intervened_latent_mean=int_z_mean,
            inferred_z_t_star=z_t_star_mean.cpu().numpy()[0],
            inferred_z_logvar=z_t_star_logvar.cpu().numpy()[0],
            effects=effects,
            horizons=list(horizons),
        )

    def simulate_from_learner_record(
        self,
        record: LearnerInterventionRecord,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
        deterministic: bool = True,
        num_particles: int = 50,
    ) -> LearnedInterventionResult:
        """Execute intervention simulation directly from a LearnerInterventionRecord."""
        spec = InterventionSpec(
            target=record.target,
            value=record.value,
            intervention_time=record.intervention_time,
            metadata={"parent_episode_id": record.parent_episode_id},
        )

        return self.simulate(
            pre_observations=record.pre_intervention_observations,
            pre_actions=record.pre_intervention_actions,
            future_actions=record.future_actions,
            intervention=spec,
            intervention_time=record.intervention_time,
            horizons=horizons,
            deterministic=deterministic,
            num_particles=num_particles,
            intervention_id=record.intervention_id,
        )
