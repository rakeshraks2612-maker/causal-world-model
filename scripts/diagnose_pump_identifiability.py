"""Diagnostic Script for Task 5.7B: Pump Action Identifiability Diagnostic.

Executes comprehensive 5-step diagnostic:
1. Controlled Action Sensitivity Experiment on Frozen baseline_003 (Pump 1/2/3/4 across horizons 1, 5, 10, 20, 40).
2. Trajectory Divergence & Norm Quantification.
3. Cross-Channel Action Sensitivity Comparison (Valve, Throttle, Pump, Flush vs Oracle).
4. Training Dataset Statistical Audit (Pump stage distribution, transitions, correlations, confounding).
5. Transition Network Functional Jacobian & Gradient Sensitivity Analysis (dZ/dA, dO/dA).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.training.normalization import ObservationNormalizer
from prism.dataset.schema import LearnerEpisode, OracleEpisode
from prism.dataset.generator import generate_single_episode, SplitType
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector, OBSERVABLE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES


def run_controlled_pump_sensitivity_experiment(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> Dict[str, Any]:
    """Step 1: Controlled pump sensitivity experiment holding latent Z_t* fixed."""
    # Generate representative nominal/moderate test episode
    base_ep = generate_single_episode(SplitType.TEST, index=104, regime="moderate_wear", length=100)
    t_star = 40

    obs_slice = base_ep.observations[: t_star + 1]
    act_slice = base_ep.actions[: t_star + 1]

    norm_obs = normalizer.normalize(torch.tensor(obs_slice, dtype=torch.float32)).unsqueeze(0)
    mask = torch.ones_like(norm_obs)
    acts_t = torch.tensor(act_slice, dtype=torch.float32).unsqueeze(0)

    inputs = ModelInputs(observations=norm_obs, observation_mask=mask, actions=acts_t)
    with torch.no_grad():
        post_latents, _ = model.encode(inputs)
        z0 = post_latents.mean[:, -1]  # [1, 64]

    max_h = max(horizons)
    results_by_stage: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for stage in [1.0, 2.0, 3.0, 4.0]:
            fut_acts = torch.zeros(1, max_h, 4)
            fut_acts[:, :, 0] = 85.0  # Valve fixed
            fut_acts[:, :, 1] = 80.0  # Throttle fixed
            fut_acts[:, :, 2] = stage  # Pump varied
            fut_acts[:, :, 3] = 0.0   # Flush fixed

            traj = model.rollout_manager.rollout_deterministic(initial_z=z0, actions=fut_acts)
            obs_phys = normalizer.denormalize(traj.observations.mean.cpu()).detach().numpy()[0]  # [H, 8]
            results_by_stage[int(stage)] = obs_phys

    # Oracle physical simulation from identical physical state
    oracle_by_stage: Dict[int, np.ndarray] = {}
    init_state = StateVector.from_array(base_ep.ground_truth_states[t_star])
    for stage in [1.0, 2.0, 3.0, 4.0]:
        sim = THCSimulator(seed=base_ep.seed + int(stage * 10))
        act_seq = np.tile(np.array([85.0, 80.0, stage, 0.0]), (max_h, 1))
        noise_slice = base_ep.exogenous_noise[t_star : t_star + max_h + 1]
        sim_res = sim.replay_episode(
            recorded_noise=noise_slice,
            action_sequence=act_seq,
            initial_state=init_state,
            episode_id=f"pump_oracle_{int(stage)}",
        )
        oracle_by_stage[int(stage)] = sim_res.observations[1:]  # [max_h, 8]

    # Metrics table at each horizon: compare stage 3 vs stage 1
    p1_pred = results_by_stage[1]
    p3_pred = results_by_stage[3]
    p4_pred = results_by_stage[4]

    p1_orc = oracle_by_stage[1]
    p3_orc = oracle_by_stage[3]
    p4_orc = oracle_by_stage[4]

    table_data = []
    for h in horizons:
        idx = h - 1
        table_data.append({
            "horizon": h,
            "PRISM_F_cool_p1": float(p1_pred[idx, 3]),
            "PRISM_F_cool_p3": float(p3_pred[idx, 3]),
            "PRISM_delta_F": float(p3_pred[idx, 3] - p1_pred[idx, 3]),
            "Oracle_delta_F": float(p3_orc[idx, 3] - p1_orc[idx, 3]),
            "PRISM_P_sys_p1": float(p1_pred[idx, 2]),
            "PRISM_P_sys_p3": float(p3_pred[idx, 2]),
            "PRISM_delta_P": float(p3_pred[idx, 2] - p1_pred[idx, 2]),
            "Oracle_delta_P": float(p3_orc[idx, 2] - p1_orc[idx, 2]),
            "PRISM_T_cool_p1": float(p1_pred[idx, 1]),
            "PRISM_T_cool_p3": float(p3_pred[idx, 1]),
            "PRISM_delta_T_cool": float(p3_pred[idx, 1] - p1_pred[idx, 1]),
            "Oracle_delta_T_cool": float(p3_orc[idx, 1] - p1_orc[idx, 1]),
            "PRISM_T_core_p1": float(p1_pred[idx, 0]),
            "PRISM_T_core_p3": float(p3_pred[idx, 0]),
            "PRISM_delta_T_core": float(p3_pred[idx, 0] - p1_pred[idx, 0]),
            "Oracle_delta_T_core": float(p3_orc[idx, 0] - p1_orc[idx, 0]),
            "PRISM_delta_Vib": float(p3_pred[idx, 6] - p1_pred[idx, 6]),
            "Oracle_delta_Vib": float(p3_orc[idx, 6] - p1_orc[idx, 6]),
        })

    # Trajectory divergence norm
    traj_diff_1_3 = p3_pred - p1_pred
    l2_divergence_1_3 = float(np.linalg.norm(traj_diff_1_3))
    mean_abs_divergence = float(np.mean(np.abs(traj_diff_1_3)))

    orc_diff_1_3 = p3_orc - p1_orc
    orc_l2_divergence_1_3 = float(np.linalg.norm(orc_diff_1_3))

    return {
        "horizon_table": table_data,
        "l2_divergence_PRISM": l2_divergence_1_3,
        "mean_abs_divergence_PRISM": mean_abs_divergence,
        "l2_divergence_Oracle": orc_l2_divergence_1_3,
        "sensitivity_ratio": l2_divergence_1_3 / max(1e-6, orc_l2_divergence_1_3),
    }


def run_cross_channel_sensitivity_audit(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
) -> Dict[str, Any]:
    """Step 3: Compare learned sensitivity vs oracle sensitivity across all 4 action channels."""
    base_ep = generate_single_episode(SplitType.TEST, index=104, regime="moderate_wear", length=100)
    t_star = 40
    H = 40

    norm_obs = normalizer.normalize(torch.tensor(base_ep.observations[: t_star + 1], dtype=torch.float32)).unsqueeze(0)
    acts_t = torch.tensor(base_ep.actions[: t_star + 1], dtype=torch.float32).unsqueeze(0)
    inputs = ModelInputs(observations=norm_obs, observation_mask=torch.ones_like(norm_obs), actions=acts_t)
    with torch.no_grad():
        post_latents, _ = model.encode(inputs)
        z0 = post_latents.mean[:, -1]

    # Baseline action sequence: Valve=50, Throttle=70, Pump=2, Flush=0
    base_act = np.array([50.0, 70.0, 2.0, 0.0])
    base_acts_t = torch.tensor(np.tile(base_act, (H, 1)), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        base_traj = model.rollout_manager.rollout_deterministic(initial_z=z0, actions=base_acts_t)
        base_pred = normalizer.denormalize(base_traj.observations.mean.cpu()).detach().numpy()[0]

    sim = THCSimulator(seed=base_ep.seed)
    init_st = StateVector.from_array(base_ep.ground_truth_states[t_star])
    noise_slice = base_ep.exogenous_noise[t_star : t_star + H + 1]
    base_sim = sim.replay_episode(
        recorded_noise=noise_slice,
        action_sequence=np.tile(base_act, (H, 1)),
        initial_state=init_st,
        episode_id="base_audit",
    )
    base_orc = base_sim.observations[1:]

    channel_tests = [
        ("A_valve", 0, 85.0, 50.0, "Valve opening 50% -> 85%"),
        ("A_throttle", 1, 30.0, 70.0, "CPU Throttle 70% -> 30%"),
        ("A_pump", 2, 4.0, 2.0, "Pump speed 2 -> 4"),
        ("A_flush", 3, 1.0, 0.0, "Emergency purge 0 -> 1"),
    ]

    channel_results = []

    for name, idx, val_int, val_base, desc in channel_tests:
        # Intervened actions
        int_act = np.copy(base_act)
        int_act[idx] = val_int
        int_acts_t = torch.tensor(np.tile(int_act, (H, 1)), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            int_traj = model.rollout_manager.rollout_deterministic(initial_z=z0, actions=int_acts_t)
            int_pred = normalizer.denormalize(int_traj.observations.mean.cpu()).detach().numpy()[0]

        int_sim = sim.replay_episode(
            recorded_noise=noise_slice,
            action_sequence=np.tile(int_act, (H, 1)),
            initial_state=init_st,
            episode_id=f"int_{name}",
        )
        int_orc = int_sim.observations[1:]

        pred_diff = int_pred - base_pred
        orc_diff = int_orc - base_orc

        pred_l2 = float(np.linalg.norm(pred_diff))
        orc_l2 = float(np.linalg.norm(orc_diff))
        ratio = float(pred_l2 / max(1e-6, orc_l2))

        # Channel-specific key delta
        if name == "A_valve":
            p_metric = float(np.mean(pred_diff[:, 3]))  # Delta F_cool
            o_metric = float(np.mean(orc_diff[:, 3]))
            metric_name = "mean_delta_F_cool"
        elif name == "A_throttle":
            p_metric = float(np.mean(pred_diff[:, 4]))  # Delta L_cpu
            o_metric = float(np.mean(orc_diff[:, 4]))
            metric_name = "mean_delta_L_cpu"
        elif name == "A_pump":
            p_metric = float(np.mean(pred_diff[:, 3]))  # Delta F_cool
            o_metric = float(np.mean(orc_diff[:, 3]))
            metric_name = "mean_delta_F_cool"
        else:
            p_metric = float(np.mean(pred_diff[:, 3]))  # Flush flow surge
            o_metric = float(np.mean(orc_diff[:, 3]))
            metric_name = "mean_delta_F_cool"

        channel_results.append({
            "action": name,
            "description": desc,
            "metric_name": metric_name,
            "PRISM_metric_delta": p_metric,
            "Oracle_metric_delta": o_metric,
            "PRISM_L2_response": pred_l2,
            "Oracle_L2_response": orc_l2,
            "learned_to_oracle_ratio": ratio,
        })

    return {"cross_channel_audit": channel_results}


def run_training_dataset_audit(data_dir: str | Path = "data/intervention_aware/learner") -> Dict[str, Any]:
    """Step 2: Statistical audit of pump stage distributions, transitions, and confounding in training data."""
    data_path = Path(data_dir)
    train_files = sorted((data_path / "train").glob("*.npz"))
    val_files = sorted((data_path / "validation").glob("*.npz"))

    all_files = train_files + val_files
    total_timesteps = 0
    pump_stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    pump_transitions = 0
    pump_hold_durations: List[int] = []

    all_pump_acts: List[float] = []
    all_valve_acts: List[float] = []
    all_throttle_acts: List[float] = []
    all_flow_obs: List[float] = []
    all_temp_obs: List[float] = []
    all_pres_obs: List[float] = []

    for f in all_files:
        data = np.load(f)
        acts = data["actions"]        # [L, 4]
        obs = data["observations"]    # [L, 8]
        L = len(acts)
        total_timesteps += L

        pumps = acts[:, 2]
        all_pump_acts.extend(pumps.tolist())
        all_valve_acts.extend(acts[:, 0].tolist())
        all_throttle_acts.extend(acts[:, 1].tolist())
        all_flow_obs.extend(obs[:, 3].tolist())
        all_temp_obs.extend(obs[:, 0].tolist())
        all_pres_obs.extend(obs[:, 2].tolist())

        # Count occurrences
        for val in pumps:
            stage_int = int(round(val))
            if stage_int in pump_stage_counts:
                pump_stage_counts[stage_int] += 1

        # Count transitions and hold lengths
        curr_stage = int(round(pumps[0]))
        curr_len = 1
        for i in range(1, L):
            st = int(round(pumps[i]))
            if st != curr_stage:
                pump_transitions += 1
                pump_hold_durations.append(curr_len)
                curr_stage = st
                curr_len = 1
            else:
                curr_len += 1
        pump_hold_durations.append(curr_len)

    # Compute correlation matrix
    act_mat = np.column_stack([
        all_pump_acts,
        all_valve_acts,
        all_throttle_acts,
        all_flow_obs,
        all_temp_obs,
        all_pres_obs,
    ])
    corr_mat = np.corrcoef(act_mat, rowvar=False)

    total_pts = float(total_timesteps)
    stage_fractions = {k: float(v / total_pts) for k, v in pump_stage_counts.items()}

    return {
        "num_episodes": len(all_files),
        "total_timesteps": total_timesteps,
        "pump_stage_counts": pump_stage_counts,
        "pump_stage_fractions": stage_fractions,
        "total_pump_transitions": pump_transitions,
        "mean_pump_hold_duration": float(np.mean(pump_hold_durations)) if pump_hold_durations else 0.0,
        "correlations": {
            "corr_pump_valve": float(corr_mat[0, 1]),
            "corr_pump_throttle": float(corr_mat[0, 2]),
            "corr_pump_flow_obs": float(corr_mat[0, 3]),
            "corr_pump_temp_obs": float(corr_mat[0, 4]),
            "corr_pump_pres_obs": float(corr_mat[0, 5]),
        },
    }


def run_transition_jacobian_analysis(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
) -> Dict[str, Any]:
    """Step 4: Functional Jacobian sensitivity analysis ||dZ_{t+1}/dA_j|| and ||dO_{t+1}/dA_j||."""
    model.eval()
    batch_size = 100
    # Sample batch of latent points from standard distribution
    z_samples = torch.randn(batch_size, 64, requires_grad=True)

    # Action point: nominal midpoint
    action_base = torch.tensor([[55.0, 60.0, 2.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1).requires_grad_(True)

    # 1. Evaluate single-step transition Z_{t+1}
    next_z_dist = model.transition(z_samples, action_base)
    mu_next_z = next_z_dist.mean  # [B, 64]

    # Compute Jacobian norm with respect to each action dimension
    # d(mu_next_z)/dA_j
    action_grad_norms = []
    action_names = ["A_valve", "A_throttle", "A_pump", "A_flush"]

    for j in range(4):
        # Compute gradient of sum of output components w.r.t action channel j
        grad_norms = []
        for out_dim in range(64):
            grads = torch.autograd.grad(
                outputs=mu_next_z[:, out_dim].sum(),
                inputs=action_base,
                retain_graph=True,
                create_graph=False,
            )[0]
            grad_norms.append(grads[:, j].abs().mean().item())
        action_grad_norms.append(float(np.mean(grad_norms)))

    # 2. End-to-end Jacobian d(Decoded Observations O_{t+1})/dA_j
    obs_dist = model.decoder(mu_next_z)
    mu_obs = obs_dist.mean  # [B, 8]

    obs_grad_norms = {}
    for j in range(4):
        channel_name = action_names[j]
        obs_grads_for_channel = []
        for obs_idx in range(8):
            grads = torch.autograd.grad(
                outputs=mu_obs[:, obs_idx].sum(),
                inputs=action_base,
                retain_graph=True,
                create_graph=False,
            )[0]
            obs_grads_for_channel.append(grads[:, j].abs().mean().item())
        obs_grad_norms[channel_name] = {
            OBSERVABLE_VARIABLES[k]: float(obs_grads_for_channel[k]) for k in range(8)
        }

    return {
        "latent_jacobian_sensitivity": {
            action_names[i]: action_grad_norms[i] for i in range(4)
        },
        "observation_jacobian_sensitivity": obs_grad_norms,
    }


def execute_full_diagnostic():
    """Execute complete Task 5.7B diagnostic suite and print structured report."""
    print("=========================================================================")
    print("   PRISM TASK 5.7B: PUMP ACTION IDENTIFIABILITY DIAGNOSTIC REPORT        ")
    print("=========================================================================\n")

    model_path = Path("artifacts/baseline_003")
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step 1: Controlled pump sensitivity
    sens_res = run_controlled_pump_sensitivity_experiment(model, norm)
    print("### 1. Controlled Pump Sensitivity Table (A_pump=3 vs A_pump=1)")
    print("| Horizon h | PRISM ΔF (L/min) | Oracle ΔF (L/min) | PRISM ΔP (bar) | Oracle ΔP (bar) | PRISM ΔT_core (°C) | Oracle ΔT_core (°C) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sens_res["horizon_table"]:
        print(f"| h={row['horizon']:2d} | {row['PRISM_delta_F']:+.3f} | {row['Oracle_delta_F']:+.3f} | {row['PRISM_delta_P']:+.3f} | {row['Oracle_delta_P']:+.3f} | {row['PRISM_delta_T_core']:+.3f} | {row['Oracle_delta_T_core']:+.3f} |")

    print(f"\n- Learned L2 Trajectory Divergence ||Traj(p3) - Traj(p1)||: {sens_res['l2_divergence_PRISM']:.4f}")
    print(f"- Oracle L2 Trajectory Divergence ||Traj(p3) - Traj(p1)||:  {sens_res['l2_divergence_Oracle']:.4f}")
    print(f"- Sensitivity Attenuation Ratio (PRISM / Oracle):       {sens_res['sensitivity_ratio']:.4%}\n")

    # Step 2: Training dataset audit
    data_audit = run_training_dataset_audit()
    print("### 2. Training Dataset Pump-Stage Coverage & Statistics")
    print(f"- Total Episodes: {data_audit['num_episodes']} | Total Timesteps: {data_audit['total_timesteps']}")
    print("| Pump Stage | Timestep Count | Distribution Fraction | Status |")
    print("| :--- | ---: | ---: | :--- |")
    for st, count in data_audit["pump_stage_counts"].items():
        frac = data_audit["pump_stage_fractions"][st]
        print(f"| Stage {st} | {count:6d} | {frac:6.2%} | {'✅ Sufficient' if frac > 0.05 else '⚠️ Sparse'} |")
    print(f"- Total In-Episode Pump Transitions: {data_audit['total_pump_transitions']}")
    print(f"- Mean Pump Hold Duration:           {data_audit['mean_pump_hold_duration']:.1f} steps")
    print(f"- Correlations with other signals:   Corr(Pump, Valve)={data_audit['correlations']['corr_pump_valve']:+.3f}, Corr(Pump, Throttle)={data_audit['correlations']['corr_pump_throttle']:+.3f}, Corr(Pump, Flow)={data_audit['correlations']['corr_pump_flow_obs']:+.3f}\n")

    # Step 3: Cross-channel sensitivity audit
    cross_audit = run_cross_channel_sensitivity_audit(model, norm)
    print("### 3. Action Identifiability Across Channels")
    print("| Action Channel | Key Intervened Metric | PRISM Delta | Oracle Delta | PRISM L2 | Oracle L2 | Learned/Oracle Ratio |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for r in cross_audit["cross_channel_audit"]:
        print(f"| `{r['action']}` | {r['metric_name']} | {r['PRISM_metric_delta']:+.3f} | {r['Oracle_metric_delta']:+.3f} | {r['PRISM_L2_response']:.4f} | {r['Oracle_L2_response']:.4f} | {r['learned_to_oracle_ratio']:.4%} |")

    # Step 4: Jacobian analysis
    jac_res = run_transition_jacobian_analysis(model, norm)
    print("\n### 4. Transition Network Functional Jacobian Sensitivity")
    print("| Action Channel | Mean ||dZ_{t+1}/dA_j|| | Decoded dF_cool/dA | Decoded dP_sys/dA | Decoded dT_core/dA |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    for act in ["A_valve", "A_throttle", "A_pump", "A_flush"]:
        z_sens = jac_res["latent_jacobian_sensitivity"][act]
        obs_sens = jac_res["observation_jacobian_sensitivity"][act]
        print(f"| `{act}` | {z_sens:.6f} | {obs_sens['F_cool']:.6f} | {obs_sens['P_sys']:.6f} | {obs_sens['T_core']:.6f} |")

    print("\n=========================================================================")
    print("                     DIAGNOSTIC SUMMARY & VERDICT                         ")
    print("=========================================================================")
    return {
        "sens_res": sens_res,
        "data_audit": data_audit,
        "cross_audit": cross_audit,
        "jac_res": jac_res,
    }


if __name__ == "__main__":
    execute_full_diagnostic()
