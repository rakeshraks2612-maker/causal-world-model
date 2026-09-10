"""Intervention Failure Diagnostic Subsystem for PRISM World Models (Task 3.4B).

Diagnoses:
1. Latent vs Decoded state intervention propagation
2. Causal chain breakage: Delta V_pos -> Delta F_cool -> Delta P_sys -> Delta T_cool -> Delta T_core
3. Action control (A=a) vs State clamp (do(X=x)) downstream trajectory differences
4. Latent sensitivity: ||Delta Z_h||_2 under state clamps vs action controls
5. Full oracle vs learned causal effect curves
6. Systematic analysis of False-Safe failure cases
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.intervention.simulator import LearnedInterventionSimulator, LearnedInterventionResult
from prism.intervention.spec import state_clamp, action_control
from prism.dataset.interventions import LearnerInterventionRecord, OracleInterventionRecord
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.simulator.state import OBSERVABLE_VARIABLES


@dataclass
class FalseSafeCaseDetails:
    """Detailed telemetry on a false-safe failure case."""

    intervention_id: str
    target: str
    value: float
    intervention_time: int
    oracle_failure_mode: str
    oracle_failure_time: int
    oracle_time_to_failure: int
    oracle_peak_t_core: float
    oracle_max_p_sys: float
    oracle_min_f_cool: float
    prism_peak_t_core: float
    prism_max_p_sys: float
    prism_min_f_cool: float
    prism_predicted_failed: bool
    failure_category: str  # Thermal Runaway, Overpressure, Cavitation, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatentSensitivityReport:
    """Latent norm differences ||Delta Z_h||_2 across interventions."""

    horizons: List[int]
    norm_delta_z_state_vpos: List[float]
    norm_delta_z_action_valve: List[float]
    norm_delta_z_state_lcpu: List[float]
    norm_delta_z_state_vib: List[float]


def diagnose_latent_sensitivity(
    simulator: LearnedInterventionSimulator,
    pre_obs: np.ndarray,
    pre_act: np.ndarray,
    fut_act: np.ndarray,
    t_star: int = 40,
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> LatentSensitivityReport:
    """Measure latent trajectory divergence ||Delta Z_h||_2 under state clamps vs action controls."""
    # 1. Baseline
    res_base = simulator.simulate(pre_obs, pre_act, fut_act, intervention=None, intervention_time=t_star)
    z_base = res_base.intervened_latent_mean  # [H, d_z]

    # 2. State clamp do(V_pos=85)
    res_do_v = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("V_pos", 85.0, intervention_time=t_star), intervention_time=t_star)
    z_do_v = res_do_v.intervened_latent_mean

    # 3. Action control A_valve=85
    res_act_v = simulator.simulate(pre_obs, pre_act, fut_act, intervention=action_control("A_valve", 85.0, intervention_time=t_star), intervention_time=t_star)
    z_act_v = res_act_v.intervened_latent_mean

    # 4. State clamp do(L_cpu=80)
    res_do_l = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("L_cpu", 80.0, intervention_time=t_star), intervention_time=t_star)
    z_do_l = res_do_l.intervened_latent_mean

    # 5. State clamp do(Vib_pump=5.0)
    res_do_vib = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("Vib_pump", 5.0, intervention_time=t_star), intervention_time=t_star)
    z_do_vib = res_do_vib.intervened_latent_mean

    norm_v_state = [float(np.linalg.norm(z_do_v[h - 1] - z_base[h - 1])) for h in horizons]
    norm_v_action = [float(np.linalg.norm(z_act_v[h - 1] - z_base[h - 1])) for h in horizons]
    norm_l_state = [float(np.linalg.norm(z_do_l[h - 1] - z_base[h - 1])) for h in horizons]
    norm_vib_state = [float(np.linalg.norm(z_do_vib[h - 1] - z_base[h - 1])) for h in horizons]

    return LatentSensitivityReport(
        horizons=list(horizons),
        norm_delta_z_state_vpos=norm_v_state,
        norm_delta_z_action_valve=norm_v_action,
        norm_delta_z_state_lcpu=norm_l_state,
        norm_delta_z_state_vib=norm_vib_state,
    )


def audit_false_safe_cases(
    simulator: LearnedInterventionSimulator,
    oracle_dir: str = "data/pilot/oracle/intervention",
    learner_dir: str = "data/pilot/learner/intervention",
) -> List[FalseSafeCaseDetails]:
    """Inspect all cases where Oracle experienced failure but PRISM predicted safe."""
    oracle_files = sorted(list(Path(oracle_dir).glob("*.npz")))
    false_safe_list: List[FalseSafeCaseDetails] = []

    for orc_file in oracle_files:
        orc_data = np.load(orc_file, allow_pickle=True)
        fail_meta = orc_data.get("failure_metrics")
        if fail_meta is None:
            continue

        f_dict = fail_meta.item() if fail_meta.ndim == 0 else dict(fail_meta)
        orc_failed = bool(f_dict.get("intervened_failed", False))
        if not orc_failed:
            continue  # Only interested in true Oracle failure cases

        # Load corresponding learner record and run PRISM simulation
        learn_file = Path(learner_dir) / orc_file.name
        learn_rec = LearnerInterventionRecord.load_npz(learn_file)
        prism_res = simulator.simulate_from_learner_record(learn_rec)

        prism_failed = prism_res.effects.failure_metrics.intervened_failed

        if not prism_failed:
            # This is a FALSE SAFE case!
            t_star = int(orc_data["intervention_time"])
            int_gt = orc_data["intervened_ground_truth_states"]
            post_gt = int_gt[t_star:]

            orc_peak_t = float(np.max(post_gt[:, 0]))
            orc_max_p = float(np.max(post_gt[:, 2]))
            orc_min_f = float(np.min(post_gt[:, 3]))

            fail_mode = str(f_dict.get("failure_mode_intervention", "unknown"))
            fail_time = int(f_dict.get("intervention_failure_time", 0))

            category = "Thermal Runaway"
            if "pressure" in fail_mode.lower() or orc_max_p >= 6.0:
                category = "Overpressure"
            elif "cavitation" in fail_mode.lower():
                category = "Cavitation"
            elif "temperature" in fail_mode.lower() or orc_peak_t >= 105.0:
                category = "Thermal Runaway"

            case = FalseSafeCaseDetails(
                intervention_id=str(orc_data["intervention_id"]),
                target=str(orc_data["target"]),
                value=float(orc_data["value"]),
                intervention_time=t_star,
                oracle_failure_mode=fail_mode,
                oracle_failure_time=fail_time,
                oracle_time_to_failure=fail_time - t_star,
                oracle_peak_t_core=orc_peak_t,
                oracle_max_p_sys=orc_max_p,
                oracle_min_f_cool=orc_min_f,
                prism_peak_t_core=prism_res.effects.peak_t_core_int,
                prism_max_p_sys=prism_res.effects.max_pressure_int,
                prism_min_f_cool=prism_res.effects.min_flow_int,
                prism_predicted_failed=prism_failed,
                failure_category=category,
            )
            false_safe_list.append(case)

    return false_safe_list
