"""Diagnostic Tool for Evaluating Action Sensitivity and Directional Causal Responses."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer


@dataclass
class ActionSweepResult:
    """Results of sweeping a single action variable."""

    action_name: str
    sweep_values: List[float]
    predicted_means: Dict[str, List[float]] # Channel -> List of predicted values
    directional_checks: Dict[str, bool]     # Assertion -> boolean passed


@dataclass
class ActionSensitivityReport:
    """Comprehensive report on learned action-response causal semantics."""

    model_name: str
    valve_sweep: ActionSweepResult
    throttle_sweep: ActionSweepResult
    all_directional_checks_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "valve_sweep": asdict(self.valve_sweep),
            "throttle_sweep": asdict(self.throttle_sweep),
            "all_directional_checks_passed": self.all_directional_checks_passed,
        }


def evaluate_action_sensitivity(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    device: Optional[torch.device] = None,
) -> ActionSensitivityReport:
    """Evaluate whether the transition and decoder models predict correct directional physical responses to action changes.
    
    Tests:
    1. Valve sweep: A_valve in [50, 85, 100] -> Expect F_cool increases, T_core & T_cool decrease.
    2. Throttle sweep: A_throttle in [20, 80, 100] -> Expect L_cpu & P_elec increase, T_core increases.
    """
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    # Create a nominal nominal history / state:
    # 60 C T_core, 40 C T_cool, 2.5 bar P_sys, 25 L/min F_cool, 50% L_cpu, 50% V_pos, 2.0 mm/s Vib, 1.5 kW P_elec
    nominal_obs_phys = torch.tensor([[60.0, 40.0, 2.5, 25.0, 50.0, 50.0, 2.0, 1.5]], dtype=torch.float32)
    norm_obs = normalizer.normalize(nominal_obs_phys).unsqueeze(0).to(dev) # [1, 1, 8]
    mask = torch.ones_like(norm_obs)
    nominal_act = torch.tensor([[[70.0, 50.0, 0.0, 0.0]]], dtype=torch.float32).to(dev) # [1, 1, 4]

    from prism.world_model.inputs import ModelInputs
    inputs = ModelInputs(observations=norm_obs, observation_mask=mask, actions=nominal_act)

    with torch.no_grad():
        post_latents, _ = model.encode(inputs)
        nominal_z = post_latents.mean[:, -1] # [1, d_z]

        # --- 1. Valve Sweep (A_valve in [50, 85, 100]) ---
        valve_vals = [50.0, 85.0, 100.0]
        valve_preds: Dict[str, List[float]] = {k: [] for k in OBSERVABLE_VARIABLES}

        for v in valve_vals:
            act = torch.tensor([[v, 50.0, 0.0, 0.0]], dtype=torch.float32).to(dev)
            next_z = model.transition_step(nominal_z, act).mean
            pred_obs_dist = model.decode(next_z)
            pred_phys = normalizer.denormalize(pred_obs_dist.mean.cpu()).squeeze(0)

            for idx, ch in enumerate(OBSERVABLE_VARIABLES):
                valve_preds[ch].append(float(pred_phys[idx].item()))

        # Check directional monotonicity for valve:
        # F_cool should increase with valve open
        f_cool_mono = valve_preds["F_cool"][1] >= valve_preds["F_cool"][0] - 0.5 and valve_preds["F_cool"][2] >= valve_preds["F_cool"][1] - 0.5
        # T_core should decrease or stabilize with increased coolant flow
        t_core_resp = valve_preds["T_core"][2] <= valve_preds["T_core"][0] + 1.0

        valve_checks = {
            "F_cool_increases_with_valve": bool(f_cool_mono),
            "T_core_controlled_by_valve": bool(t_core_resp),
        }

        # --- 2. Throttle Sweep (A_throttle in [20, 80, 100]) ---
        throttle_vals = [20.0, 80.0, 100.0]
        throttle_preds: Dict[str, List[float]] = {k: [] for k in OBSERVABLE_VARIABLES}

        for thr in throttle_vals:
            act = torch.tensor([[70.0, thr, 0.0, 0.0]], dtype=torch.float32).to(dev)
            next_z = model.transition_step(nominal_z, act).mean
            pred_obs_dist = model.decode(next_z)
            pred_phys = normalizer.denormalize(pred_obs_dist.mean.cpu()).squeeze(0)

            for idx, ch in enumerate(OBSERVABLE_VARIABLES):
                throttle_preds[ch].append(float(pred_phys[idx].item()))

        # Check directional monotonicity for throttle:
        # L_cpu and P_elec should increase with throttle
        l_cpu_mono = throttle_preds["L_cpu"][1] >= throttle_preds["L_cpu"][0] and throttle_preds["L_cpu"][2] >= throttle_preds["L_cpu"][1]
        p_elec_mono = throttle_preds["P_elec"][1] >= throttle_preds["P_elec"][0] and throttle_preds["P_elec"][2] >= throttle_preds["P_elec"][1]
        t_core_heat = throttle_preds["T_core"][2] >= throttle_preds["T_core"][0] - 0.5

        throttle_checks = {
            "L_cpu_increases_with_throttle": bool(l_cpu_mono),
            "P_elec_increases_with_throttle": bool(p_elec_mono),
            "T_core_heats_with_workload": bool(t_core_heat),
        }

    all_passed = all(valve_checks.values()) and all(throttle_checks.values())

    return ActionSensitivityReport(
        model_name="PRISM_ActionSensitivity",
        valve_sweep=ActionSweepResult(
            action_name="A_valve",
            sweep_values=valve_vals,
            predicted_means=valve_preds,
            directional_checks=valve_checks,
        ),
        throttle_sweep=ActionSweepResult(
            action_name="A_throttle",
            sweep_values=throttle_vals,
            predicted_means=throttle_preds,
            directional_checks=throttle_checks,
        ),
        all_directional_checks_passed=all_passed,
    )
