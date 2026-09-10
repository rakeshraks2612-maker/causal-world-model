"""Learned Counterfactual Engine for PRISM Causal World Models.

Implements Pearl's Level 3 Counterfactual Reasoning (Twin-World Structural Causal Model):
1. Step 1: ABDUCTION
   - Incurs historical observations (O_{<=t*}, A_{<=t*})
   - Infers posterior distribution q_phi(Z_t* | O_{<=t*}, A_{<=t*}) over the latent physical manifold
2. Step 2: ACTION SUBSTITUTION / INTERVENTION
   - Replaces factual action at t* with counterfactual action A'_{t*} (or state intervention)
   - Preserves identical pre-t* history and identical planned post-t* future actions
3. Step 3: TWIN-WORLD PREDICTION & REPLAY
   - Executes factual rollout from inferred Z_t* under factual action sequence
   - Executes counterfactual rollout from identical inferred Z_t* under counterfactual action sequence
   - Computes multi-horizon counterfactual effect deltas Delta Y^CF(h) and failure risk transitions
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
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
from prism.intervention.effects import (
    HorizonCausalDelta,
    LearnedFailureMetrics,
    CausalEffectSummary,
    compute_causal_effects,
    SAFETY_THRESHOLDS,
)
from prism.dataset.counterfactuals import (
    CounterfactualSpec,
    LearnerCounterfactualRecord,
    CounterfactualHorizonEffect,
    CounterfactualFailureMetrics,
)
from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES


@dataclass
class AbducedLatentState:
    """Inferred latent physical state and uncertainty at intervention time t*."""

    latent_mean: np.ndarray        # Shape: [d_z]
    latent_logvar: np.ndarray      # Shape: [d_z]
    latent_std: np.ndarray         # Shape: [d_z]
    timestep: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_mean": self.latent_mean.tolist(),
            "latent_logvar": self.latent_logvar.tolist(),
            "latent_std": self.latent_std.tolist(),
            "timestep": self.timestep,
        }


@dataclass
class LearnedCounterfactualResult:
    """Complete result of a learned Level-3 counterfactual simulation."""

    counterfactual_id: str
    parent_episode_id: str
    counterfactual_time: int
    target_action: str
    counterfactual_value: float
    abduced_latent_state: AbducedLatentState
    factual_observations: np.ndarray           # Shape: [H, 8] in physical units
    counterfactual_observations: np.ndarray    # Shape: [H, 8] in physical units
    factual_latent_mean: np.ndarray            # Shape: [H, d_z]
    counterfactual_latent_mean: np.ndarray     # Shape: [H, d_z]
    effects: CausalEffectSummary
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "parent_episode_id": self.parent_episode_id,
            "counterfactual_time": self.counterfactual_time,
            "target_action": self.target_action,
            "counterfactual_value": self.counterfactual_value,
            "abduced_latent_state": self.abduced_latent_state.to_dict(),
            "effects": self.effects.to_dict(),
            "horizons": self.horizons,
            "metadata": self.metadata,
        }


class LearnedCounterfactualEngine:
    """Twin-World Counterfactual Simulator operating on learned latent representations."""

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

    def abduce_latent_state(
        self,
        observations: np.ndarray | Tensor,
        actions: np.ndarray | Tensor,
        observation_mask: Optional[np.ndarray | Tensor] = None,
    ) -> Tuple[LatentDistribution, AbducedLatentState]:
        """Step 1 (Abduction): Infer latent physical state distribution q_phi(Z_t* | O_<=t*, A_<=t*)."""
        if isinstance(observations, np.ndarray):
            obs_t = torch.tensor(observations, dtype=torch.float32)
        else:
            obs_t = observations.clone().float()

        if isinstance(actions, np.ndarray):
            act_t = torch.tensor(actions, dtype=torch.float32)
        else:
            act_t = actions.clone().float()

        if observation_mask is None:
            mask_t = ~torch.isnan(obs_t)
        elif isinstance(observation_mask, np.ndarray):
            mask_t = torch.tensor(observation_mask, dtype=torch.bool)
        else:
            mask_t = observation_mask.clone().bool()

        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)
            act_t = act_t.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0)

        norm_obs = self.normalizer.normalize(obs_t).to(self.device)
        mask_t = mask_t.to(self.device).float()
        act_t = act_t.to(self.device)

        inputs = ModelInputs(
            observations=norm_obs,
            observation_mask=mask_t,
            actions=act_t,
        )

        with torch.no_grad():
            post_latents, _ = self.world_model.encode(inputs)
            z_t_star_mean = post_latents.mean[:, -1]      # [1, d_z]
            z_t_star_logvar = post_latents.logvar[:, -1]  # [1, d_z]
            z_dist = LatentDistribution(
                mean=z_t_star_mean,
                logvar=z_t_star_logvar,
                min_std=self.world_model.config.model.min_std,
                max_std=self.world_model.config.model.max_std,
            )

        t_star = obs_t.shape[1] - 1
        std_np = z_dist.std.cpu().numpy()[0]
        abduced_state = AbducedLatentState(
            latent_mean=z_t_star_mean.cpu().numpy()[0],
            latent_logvar=z_t_star_logvar.cpu().numpy()[0],
            latent_std=std_np,
            timestep=t_star,
        )

        return z_dist, abduced_state

    def evaluate_counterfactual(
        self,
        historical_observations: np.ndarray | Tensor,
        historical_actions: np.ndarray | Tensor,
        factual_action_at_t_star: np.ndarray | Tensor,
        counterfactual_action_at_t_star: np.ndarray | Tensor,
        future_actions: np.ndarray | Tensor,
        counterfactual_time: int,
        target_action: str = "A_valve",
        counterfactual_value: float = 85.0,
        observation_mask: Optional[np.ndarray | Tensor] = None,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
        deterministic: bool = True,
        num_particles: int = 50,
        counterfactual_id: Optional[str] = None,
        parent_episode_id: Optional[str] = None,
    ) -> LearnedCounterfactualResult:
        """Execute full 3-step Pearl Counterfactual Engine evaluation.
        
        Args:
            historical_observations: Observations O_{0:t*} [t*+1, 8]
            historical_actions: Actions A_{0:t*} [t*+1, 4]
            factual_action_at_t_star: Factual action at t* [4]
            counterfactual_action_at_t_star: Counterfactual substituted action at t* [4]
            future_actions: Planned actions for t > t* [H_fut, 4]
            counterfactual_time: Timestep t*
            target_action: Substituted action name
            counterfactual_value: Substituted action value
            observation_mask: Telemetry mask
            horizons: Evaluation horizons
            deterministic: Deterministic vs stochastic Monte Carlo rollouts
            num_particles: Particle count for Monte Carlo
            counterfactual_id: ID string
            parent_episode_id: Parent episode ID
            
        Returns:
            LearnedCounterfactualResult with twin-world rollouts and causal effect metrics
        """
        # Step 1: Abduction
        z_dist, abduced_state = self.abduce_latent_state(
            observations=historical_observations,
            actions=historical_actions,
            observation_mask=observation_mask,
        )
        z_init_mean = z_dist.mean.to(self.device)  # [1, d_z]

        # Step 2: Action Sequence Construction
        # Factual rollout action sequence: [A_{t*}^{fact}, A_{t*+1:T}]
        if isinstance(factual_action_at_t_star, np.ndarray):
            fact_act_t0 = torch.tensor(factual_action_at_t_star, dtype=torch.float32)
        else:
            fact_act_t0 = factual_action_at_t_star.clone().float()

        if isinstance(counterfactual_action_at_t_star, np.ndarray):
            cf_act_t0 = torch.tensor(counterfactual_action_at_t_star, dtype=torch.float32)
        else:
            cf_act_t0 = counterfactual_action_at_t_star.clone().float()

        if isinstance(future_actions, np.ndarray):
            fut_act_t = torch.tensor(future_actions, dtype=torch.float32)
        else:
            fut_act_t = future_actions.clone().float()

        if fact_act_t0.ndim == 1:
            fact_act_t0 = fact_act_t0.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        elif fact_act_t0.ndim == 2:
            fact_act_t0 = fact_act_t0.unsqueeze(0)

        if cf_act_t0.ndim == 1:
            cf_act_t0 = cf_act_t0.unsqueeze(0).unsqueeze(0)      # [1, 1, 4]
        elif cf_act_t0.ndim == 2:
            cf_act_t0 = cf_act_t0.unsqueeze(0)

        if fut_act_t.ndim == 2:
            fut_act_t = fut_act_t.unsqueeze(0)                  # [1, H_fut, 4]

        factual_actions = torch.cat([fact_act_t0, fut_act_t], dim=1).to(self.device)     # [1, 1+H_fut, 4]
        counterfactual_actions = torch.cat([cf_act_t0, fut_act_t], dim=1).to(self.device) # [1, 1+H_fut, 4]

        # Step 3: Twin-World Prediction & Replay
        with torch.no_grad():
            if deterministic:
                fact_traj = self.world_model.rollout_manager.rollout_deterministic(
                    initial_z=z_init_mean,
                    actions=factual_actions,
                )
                cf_traj = self.world_model.rollout_manager.rollout_deterministic(
                    initial_z=z_init_mean,
                    actions=counterfactual_actions,
                )
            else:
                fact_traj = self.world_model.rollout_manager.rollout_monte_carlo(
                    initial_z_dist=z_dist,
                    actions=factual_actions,
                    num_samples=num_particles,
                )
                cf_traj = self.world_model.rollout_manager.rollout_monte_carlo(
                    initial_z_dist=z_dist,
                    actions=counterfactual_actions,
                    num_samples=num_particles,
                )

            fact_obs_norm = fact_traj.observations.mean.cpu()
            cf_obs_norm = cf_traj.observations.mean.cpu()

            fact_obs_phys = self.normalizer.denormalize(fact_obs_norm).numpy()[0]  # [H, 8]
            cf_obs_phys = self.normalizer.denormalize(cf_obs_norm).numpy()[0]      # [H, 8]

            fact_z_mean = fact_traj.latent_mean.cpu().numpy()[0]
            cf_z_mean = cf_traj.latent_mean.cpu().numpy()[0]

        # Step 4: Compute Counterfactual Causal Effects (Y_cf - Y_fact)
        effects = compute_causal_effects(
            baseline_obs=fact_obs_phys,
            intervened_obs=cf_obs_phys,
            horizons=horizons,
            t_star=counterfactual_time,
        )

        cf_id = counterfactual_id or f"cf_t{counterfactual_time}_{target_action}_{int(counterfactual_value)}"
        parent_id = parent_episode_id or "unknown"

        return LearnedCounterfactualResult(
            counterfactual_id=cf_id,
            parent_episode_id=parent_id,
            counterfactual_time=counterfactual_time,
            target_action=target_action,
            counterfactual_value=counterfactual_value,
            abduced_latent_state=abduced_state,
            factual_observations=fact_obs_phys,
            counterfactual_observations=cf_obs_phys,
            factual_latent_mean=fact_z_mean,
            counterfactual_latent_mean=cf_z_mean,
            effects=effects,
            horizons=list(horizons),
        )

    def evaluate_from_learner_record(
        self,
        record: LearnerCounterfactualRecord,
        horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
        deterministic: bool = True,
        num_particles: int = 50,
    ) -> LearnedCounterfactualResult:
        """Run counterfactual evaluation directly from a LearnerCounterfactualRecord."""
        return self.evaluate_counterfactual(
            historical_observations=record.historical_observations,
            historical_actions=record.historical_actions,
            factual_action_at_t_star=record.factual_action,
            counterfactual_action_at_t_star=record.counterfactual_action,
            future_actions=record.future_actions,
            counterfactual_time=record.counterfactual_time,
            target_action=record.target_action,
            counterfactual_value=record.counterfactual_value,
            observation_mask=record.historical_observation_mask,
            horizons=horizons,
            deterministic=deterministic,
            num_particles=num_particles,
            counterfactual_id=record.counterfactual_id,
            parent_episode_id=record.parent_episode_id,
        )
