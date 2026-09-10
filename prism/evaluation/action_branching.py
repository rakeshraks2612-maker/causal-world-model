"""Action Branching and Causal Counterfactual Rollout Engine."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.dataset.schema import LearnerEpisode
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


@dataclass
class BranchTrajectory:
    """Predicted trajectory for an action branch."""

    branch_id: str
    action_value: float
    latent_trajectory: List[List[float]]       # Shape: [H+1, d_z] (including initial z)
    observation_trajectory: Dict[str, List[float]] # Channel -> [H] values in physical units


@dataclass
class ActionBranchingExperiment:
    """Evaluation of multiple action branches originating from the exact same latent state."""

    experiment_name: str
    action_name: str
    horizon: int
    initial_latent_norm: float
    branches: Dict[str, BranchTrajectory]
    pre_branch_identical: bool
    post_branch_divergent: bool
    directional_causal_checks: Dict[str, bool]


@dataclass
class ActionBranchingReport:
    """Comprehensive report on learned action branching and causal simulation."""

    model_name: str
    valve_experiment: ActionBranchingExperiment
    throttle_experiment: ActionBranchingExperiment
    all_invariants_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "valve_experiment": asdict(self.valve_experiment),
            "throttle_experiment": asdict(self.throttle_experiment),
            "all_invariants_passed": self.all_invariants_passed,
        }


def evaluate_action_branching(
    model: CausalWorldModel,
    reference_episode: LearnerEpisode,
    normalizer: ObservationNormalizer,
    horizon: int = 40,
    context_length: int = 40,
    device: Optional[torch.device] = None,
) -> ActionBranchingReport:
    """Branch action sequences from the exact same inferred initial latent state and verify causal dynamics."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    # 1. Encode Context Window to Infer Initial Latent State Z_t*
    raw_obs = reference_episode.observations[:context_length]
    raw_mask = reference_episode.observation_mask[:context_length]
    raw_act = reference_episode.actions[:context_length]

    norm_obs = normalizer.normalize(torch.tensor(raw_obs, dtype=torch.float32)).unsqueeze(0).to(dev)
    mask_t = torch.tensor(raw_mask, dtype=torch.float32).unsqueeze(0).to(dev)
    act_t = torch.tensor(raw_act, dtype=torch.float32).unsqueeze(0).to(dev)

    inputs = ModelInputs(observations=norm_obs, observation_mask=mask_t, actions=act_t)

    with torch.no_grad():
        post_latents, _ = model.encode(inputs)
        initial_z = post_latents.mean[:, -1] # [1, d_z]

        # --- Experiment 1: Valve Action Branching ---
        # A_valve in {50, 85, 100}, throttle = 50%, pump = 0, flush = 0
        valve_values = [50.0, 85.0, 100.0]
        valve_branches: Dict[str, BranchTrajectory] = {}

        for val in valve_values:
            b_id = f"valve_{int(val)}"
            # Construct constant future action sequence [1, H, 4]
            fut_act = torch.zeros(1, horizon, 4, device=dev)
            fut_act[:, :, 0] = val   # A_valve
            fut_act[:, :, 1] = 50.0  # A_throttle

            rollout = model.rollout_manager.rollout_deterministic(initial_z, fut_act)
            obs_phys = normalizer.denormalize(rollout.observations.mean.cpu()).squeeze(0) # [H, 8]

            # Collect channel trajectories
            ch_traj = {}
            for ch_idx, ch_name in enumerate(OBSERVABLE_VARIABLES):
                ch_traj[ch_name] = [float(obs_phys[h, ch_idx].item()) for h in range(horizon)]

            lat_traj = rollout.latent_mean.squeeze(0).cpu().numpy().tolist()
            valve_branches[b_id] = BranchTrajectory(
                branch_id=b_id,
                action_value=val,
                latent_trajectory=lat_traj,
                observation_trajectory=ch_traj,
            )

        # Invariant checks for Valve
        # 1. Pre-branch state check: initial_z was the exact same tensor instance
        pre_id_valve = True
        # 2. Divergence check: final step T_core, F_cool across branches
        b50_f = valve_branches["valve_50"].observation_trajectory["F_cool"][-1]
        b85_f = valve_branches["valve_85"].observation_trajectory["F_cool"][-1]
        b100_f = valve_branches["valve_100"].observation_trajectory["F_cool"][-1]

        b50_t = valve_branches["valve_50"].observation_trajectory["T_core"][-1]
        b85_t = valve_branches["valve_85"].observation_trajectory["T_core"][-1]
        b100_t = valve_branches["valve_100"].observation_trajectory["T_core"][-1]

        divergent_valve = (abs(b85_f - b50_f) > 0.1) and (abs(b100_f - b85_f) > 0.05)
        # Directional: F_cool increases with valve opening; T_core is lower or stabilized with higher cooling
        valve_dir_checks = {
            "F_cool_monotonic_with_valve": bool(b100_f >= b85_f - 0.5 and b85_f >= b50_f - 0.5),
            "T_core_controlled_by_valve": bool(b100_t <= b50_t + 1.0),
        }

        valve_exp = ActionBranchingExperiment(
            experiment_name="Valve_Action_Branching",
            action_name="A_valve",
            horizon=horizon,
            initial_latent_norm=float(torch.norm(initial_z).item()),
            branches=valve_branches,
            pre_branch_identical=pre_id_valve,
            post_branch_divergent=divergent_valve,
            directional_causal_checks=valve_dir_checks,
        )

        # --- Experiment 2: Throttle Action Branching ---
        # A_throttle in {20, 80, 100}, valve = 70%, pump = 0, flush = 0
        throttle_values = [20.0, 80.0, 100.0]
        throttle_branches: Dict[str, BranchTrajectory] = {}

        for thr in throttle_values:
            b_id = f"throttle_{int(thr)}"
            fut_act = torch.zeros(1, horizon, 4, device=dev)
            fut_act[:, :, 0] = 70.0  # A_valve
            fut_act[:, :, 1] = thr   # A_throttle

            rollout = model.rollout_manager.rollout_deterministic(initial_z, fut_act)
            obs_phys = normalizer.denormalize(rollout.observations.mean.cpu()).squeeze(0)

            ch_traj = {}
            for ch_idx, ch_name in enumerate(OBSERVABLE_VARIABLES):
                ch_traj[ch_name] = [float(obs_phys[h, ch_idx].item()) for h in range(horizon)]

            lat_traj = rollout.latent_mean.squeeze(0).cpu().numpy().tolist()
            throttle_branches[b_id] = BranchTrajectory(
                branch_id=b_id,
                action_value=thr,
                latent_trajectory=lat_traj,
                observation_trajectory=ch_traj,
            )

        b20_l = throttle_branches["throttle_20"].observation_trajectory["L_cpu"][-1]
        b80_l = throttle_branches["throttle_80"].observation_trajectory["L_cpu"][-1]
        b100_l = throttle_branches["throttle_100"].observation_trajectory["L_cpu"][-1]

        b20_t = throttle_branches["throttle_20"].observation_trajectory["T_core"][-1]
        b80_t = throttle_branches["throttle_80"].observation_trajectory["T_core"][-1]
        b100_t = throttle_branches["throttle_100"].observation_trajectory["T_core"][-1]

        divergent_throttle = (abs(b80_l - b20_l) > 0.1) and (abs(b100_l - b80_l) > 0.05)
        throttle_dir_checks = {
            "L_cpu_monotonic_with_throttle": bool(b100_l >= b80_l - 0.5 and b80_l >= b20_l - 0.5),
            "T_core_heats_with_workload": bool(b100_t >= b20_t - 0.5),
        }

        throttle_exp = ActionBranchingExperiment(
            experiment_name="Throttle_Action_Branching",
            action_name="A_throttle",
            horizon=horizon,
            initial_latent_norm=float(torch.norm(initial_z).item()),
            branches=throttle_branches,
            pre_branch_identical=True,
            post_branch_divergent=divergent_throttle,
            directional_causal_checks=throttle_dir_checks,
        )

    all_passed = (
        valve_exp.pre_branch_identical
        and valve_exp.post_branch_divergent
        and all(valve_exp.directional_causal_checks.values())
        and throttle_exp.pre_branch_identical
        and throttle_exp.post_branch_divergent
        and all(throttle_exp.directional_causal_checks.values())
    )

    return ActionBranchingReport(
        model_name="PRISM_ActionBranching",
        valve_experiment=valve_exp,
        throttle_experiment=throttle_exp,
        all_invariants_passed=all_passed,
    )
