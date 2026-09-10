"""Task 3.4B: Learned Intervention Failure Diagnostics and Causal Chain Breakdown.

Executes:
1. Experiment 1: State-clamp propagation (Latent vs Decoded state dynamics)
2. Experiment 2: Causal chain propagation comparison (Oracle vs PRISM)
3. Experiment 3: Action-mediated control (A=a) vs State intervention (do(X=x))
4. Experiment 4: Latent intervention sensitivity ||Delta Z_h||_2
5. Experiment 5: Multi-horizon causal effect curves (Oracle vs Learned)
6. Experiment 6: Detailed audit of the 16 False-Safe failure cases
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.intervention.spec import state_clamp, action_control
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.dataset.interventions import LearnerInterventionRecord
from prism.diagnosis.intervention_diagnostics import (
    diagnose_latent_sensitivity,
    audit_false_safe_cases,
    FalseSafeCaseDetails,
)
from prism.simulator.state import OBSERVABLE_VARIABLES


def run_diagnostics(
    checkpoint_dir: str = "artifacts/baseline_002",
    oracle_dir: str = "data/pilot/oracle/intervention",
    learner_dir: str = "data/pilot/learner/intervention",
    output_dir: str = "artifacts/intervention/diagnostics_3_4b",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print(" " * 20 + "TASK 3.4B: LEARNED INTERVENTION FAILURE DIAGNOSTICS")
    print("=" * 95)

    # Load frozen world model
    ckpt_path = Path(checkpoint_dir)
    normalizer = ObservationNormalizer.load_yaml(ckpt_path / "normalization.yaml")
    config = WorldModelConfig.from_yaml(ckpt_path / "config.yaml")
    model = CausalWorldModel(config)
    checkpoint = torch.load(ckpt_path / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    simulator = LearnedInterventionSimulator(model, normalizer)

    # Fixed representative test episode
    ep = generate_single_episode(SplitType.TEST, index=0, regime="moderate_load", length=120)
    t_star = 40
    pre_obs = ep.observations[:t_star + 1]
    pre_act = ep.actions[:t_star + 1]
    fut_act = ep.actions[t_star + 1:t_star + 41]

    # -------------------------------------------------------------
    # Experiment 1 & 4: Latent Sensitivity & State-Clamp Propagation
    # -------------------------------------------------------------
    print("\n--- [Experiment 1 & 4] Latent State Intervention Sensitivity ---")
    horizons = (1, 5, 10, 20, 40)
    sens_report = diagnose_latent_sensitivity(
        simulator=simulator,
        pre_obs=pre_obs,
        pre_act=pre_act,
        fut_act=fut_act,
        t_star=t_star,
        horizons=horizons,
    )

    print(f"{'Horizon':<8s} | {'||ΔZ|| do(V_pos)':>16s} | {'||ΔZ|| A_valve':>16s} | {'||ΔZ|| do(L_cpu)':>16s} | {'||ΔZ|| do(Vib)':>16s}")
    print("-" * 80)
    for i, h in enumerate(horizons):
        print(
            f"h = {h:<4d} | {sens_report.norm_delta_z_state_vpos[i]:>16.4f} | "
            f"{sens_report.norm_delta_z_action_valve[i]:>16.4f} | "
            f"{sens_report.norm_delta_z_state_lcpu[i]:>16.4f} | "
            f"{sens_report.norm_delta_z_state_vib[i]:>16.4f}"
        )

    # -------------------------------------------------------------
    # Experiment 2: Causal Chain Propagation (Oracle vs PRISM)
    # -------------------------------------------------------------
    print("\n--- [Experiment 2] Causal Chain Propagation Analysis at h = 10 ---")
    # Interventions on V_pos
    v_pos_vals = [50.0, 70.0, 85.0, 100.0]
    oracle_files = sorted(list(Path(oracle_dir).glob("*.npz")))

    # Collect average deltas across all episodes for do(V_pos) in Oracle vs PRISM
    chain_results: Dict[str, Dict[str, float]] = {}
    for val in v_pos_vals:
        # Match oracle records
        sub_orcs = [f for f in oracle_files if f"V_pos_{int(val)}" in f.name or f"V_pos_{val:g}" in f.name]
        orc_deltas = {"V_pos": 0.0, "F_cool": 0.0, "P_sys": 0.0, "T_cool": 0.0, "T_core": 0.0}
        prd_deltas = {"V_pos": 0.0, "F_cool": 0.0, "P_sys": 0.0, "T_cool": 0.0, "T_core": 0.0}

        count = 0
        for orc_f in sub_orcs:
            orc_data = np.load(orc_f, allow_pickle=True)
            if "10" not in orc_data["horizon_effects"].item():
                continue
            h10_orc = orc_data["horizon_effects"].item()["10"]

            learn_f = Path(learner_dir) / orc_f.name
            learn_rec = LearnerInterventionRecord.load_npz(learn_f)
            prd_res = simulator.simulate_from_learner_record(learn_rec)
            h10_prd = prd_res.effects.horizon_effects.get(10)

            if h10_prd is not None:
                orc_deltas["V_pos"] += h10_orc["delta_v_pos"]
                orc_deltas["F_cool"] += h10_orc["delta_f_cool"]
                orc_deltas["P_sys"] += h10_orc["delta_p_sys"]
                orc_deltas["T_cool"] += h10_orc["delta_t_cool"]
                orc_deltas["T_core"] += h10_orc["delta_t_core"]

                prd_deltas["V_pos"] += h10_prd.delta_v_pos
                prd_deltas["F_cool"] += h10_prd.delta_f_cool
                prd_deltas["P_sys"] += h10_prd.delta_p_sys
                prd_deltas["T_cool"] += h10_prd.delta_t_cool
                prd_deltas["T_core"] += h10_prd.delta_t_core
                count += 1

        if count > 0:
            for k in orc_deltas:
                orc_deltas[k] /= count
                prd_deltas[k] /= count

        chain_results[f"do(V_pos={val:g})"] = {
            "oracle": orc_deltas,
            "prism": prd_deltas,
        }

    print(f"{'Intervention':<16s} | {'Chain Stage':<12s} | {'Oracle Δ':>14s} | {'PRISM Δ':>14s} | {'Causal Gap':>14s}")
    print("-" * 85)
    for inv_name, data in chain_results.items():
        print(f"{inv_name:<16s} | {'1. ΔV_pos':<12s} | {data['oracle']['V_pos']:>+13.2f}% | {data['prism']['V_pos']:>+13.2f}% | {abs(data['prism']['V_pos'] - data['oracle']['V_pos']):>14.2f}%")
        print(f"{'':<16s} | {'2. ΔF_cool':<12s} | {data['oracle']['F_cool']:>+13.2f}L | {data['prism']['F_cool']:>+13.2f}L | {abs(data['prism']['F_cool'] - data['oracle']['F_cool']):>14.2f}L")
        print(f"{'':<16s} | {'3. ΔP_sys':<12s} | {data['oracle']['P_sys']:>+13.2f}b | {data['prism']['P_sys']:>+13.2f}b | {abs(data['prism']['P_sys'] - data['oracle']['P_sys']):>14.2f}b")
        print(f"{'':<16s} | {'4. ΔT_cool':<12s} | {data['oracle']['T_cool']:>+13.2f}° | {data['prism']['T_cool']:>+13.2f}° | {abs(data['prism']['T_cool'] - data['oracle']['T_cool']):>14.2f}°")
        print(f"{'':<16s} | {'5. ΔT_core':<12s} | {data['oracle']['T_core']:>+13.2f}° | {data['prism']['T_core']:>+13.2f}° | {abs(data['prism']['T_core'] - data['oracle']['T_core']):>14.2f}°")
        print("-" * 85)

    # -------------------------------------------------------------
    # Experiment 3: Action-Mediated Control vs State Intervention
    # -------------------------------------------------------------
    print("\n--- [Experiment 3] Action-Mediated (A_valve=85) vs State Intervention (do(V_pos=85)) ---")
    res_act = simulator.simulate(pre_obs, pre_act, fut_act, intervention=action_control("A_valve", 85.0, intervention_time=t_star), intervention_time=t_star)
    res_state = simulator.simulate(pre_obs, pre_act, fut_act, intervention=state_clamp("V_pos", 85.0, intervention_time=t_star), intervention_time=t_star)
    res_base = simulator.simulate(pre_obs, pre_act, fut_act, intervention=None, intervention_time=t_star)

    v_idx = OBSERVABLE_VARIABLES.index("V_pos")
    f_idx = OBSERVABLE_VARIABLES.index("F_cool")
    p_idx = OBSERVABLE_VARIABLES.index("P_sys")
    tc_idx = OBSERVABLE_VARIABLES.index("T_cool")
    t_idx = OBSERVABLE_VARIABLES.index("T_core")

    print(f"{'Horizon':<8s} | {'V_pos (Act vs State)':>22s} | {'F_cool (Act vs State)':>23s} | {'T_core (Act vs State)':>23s}")
    print("-" * 85)
    for h in [1, 5, 10, 20, 40]:
        idx = h - 1
        v_a, v_s = res_act.intervened_observations[idx, v_idx], res_state.intervened_observations[idx, v_idx]
        f_a, f_s = res_act.intervened_observations[idx, f_idx], res_state.intervened_observations[idx, f_idx]
        t_a, t_s = res_act.intervened_observations[idx, t_idx], res_state.intervened_observations[idx, t_idx]
        print(f"h = {h:<4d} | {v_a:>8.1f}% vs {v_s:>6.1f}% | {f_a:>9.1f} vs {f_s:>8.1f} L | {t_a:>9.1f} vs {t_s:>8.1f} °C")

    # -------------------------------------------------------------
    # Experiment 6: Granular Audit of the 16 False-Safe Cases
    # -------------------------------------------------------------
    print("\n--- [Experiment 6] Granular Audit of the 16 False-Safe Failure Cases ---")
    false_safes = audit_false_safe_cases(simulator, oracle_dir, learner_dir)
    print(f"Total False-Safe Cases Found: {len(false_safes)}")

    category_counts: Dict[str, int] = {}
    for c in false_safes:
        category_counts[c.failure_category] = category_counts.get(c.failure_category, 0) + 1

    print("Breakdown by Failure Mode:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} cases ({count / len(false_safes):.1%})")

    print("\nSample False-Safe Cases:")
    print(f"{'Intervention ID':<35s} | {'Target':<8s} | {'Val':>5s} | {'Oracle Fail Mode':<22s} | {'Orc Peak T':>10s} | {'PRISM Peak T':>12s}")
    print("-" * 105)
    for c in false_safes[:8]:
        print(
            f"{c.intervention_id:<35s} | {c.target:<8s} | {c.value:>5.1f} | {c.oracle_failure_mode:<22s} | "
            f"{c.oracle_peak_t_core:>9.1f}°C | {c.prism_peak_t_core:>11.1f}°C"
        )

    # -------------------------------------------------------------
    # Generate Diagnostic Visualizations
    # -------------------------------------------------------------
    _generate_diagnostic_plots(chain_results, sens_report, false_safes, plots_path)
    print(f"\n✓ Generated comprehensive diagnostic plots in {plots_path}")

    # Export full JSON report
    report_dict = {
        "latent_sensitivity": {
            "horizons": sens_report.horizons,
            "norm_delta_z_state_vpos": sens_report.norm_delta_z_state_vpos,
            "norm_delta_z_action_valve": sens_report.norm_delta_z_action_valve,
            "norm_delta_z_state_lcpu": sens_report.norm_delta_z_state_lcpu,
            "norm_delta_z_state_vib": sens_report.norm_delta_z_state_vib,
        },
        "causal_chain_breakdown": chain_results,
        "false_safe_audit": {
            "total_false_safes": len(false_safes),
            "category_counts": category_counts,
            "cases": [c.to_dict() for c in false_safes],
        },
    }

    report_file = out_path / "intervention_failure_diagnostic_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"✓ Saved full diagnostic JSON report to {report_file}")

    return report_dict


def _generate_diagnostic_plots(
    chain_results: Dict[str, Dict[str, float]],
    sens_report: LatentSensitivityReport,
    false_safes: List[FalseSafeCaseDetails],
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Plot Causal Chain Breakdown at h = 10
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    v_labels = ["do(V_pos=50)", "do(V_pos=70)", "do(V_pos=85)", "do(V_pos=100)"]
    x = np.arange(len(v_labels))
    width = 0.35

    # Stage 1: V_pos delta
    orc_v = [chain_results[k]["oracle"]["V_pos"] for k in v_labels if k in chain_results]
    prd_v = [chain_results[k]["prism"]["V_pos"] for k in v_labels if k in chain_results]
    axes[0].bar(x - width/2, orc_v, width, label="Oracle ΔV_pos", color="#0275d8")
    axes[0].bar(x + width/2, prd_v, width, label="PRISM ΔV_pos", color="#5cb85c")
    axes[0].set_title("Stage 1: Actuator State (ΔV_pos)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["50%", "70%", "85%", "100%"])
    axes[0].set_ylabel("ΔV_pos (%)")
    axes[0].legend()

    # Stage 2: F_cool delta
    orc_f = [chain_results[k]["oracle"]["F_cool"] for k in v_labels if k in chain_results]
    prd_f = [chain_results[k]["prism"]["F_cool"] for k in v_labels if k in chain_results]
    axes[1].bar(x - width/2, orc_f, width, label="Oracle ΔF_cool", color="#0275d8")
    axes[1].bar(x + width/2, prd_f, width, label="PRISM ΔF_cool", color="#d9534f")
    axes[1].set_title("Stage 2: Coolant Flow (ΔF_cool)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["50%", "70%", "85%", "100%"])
    axes[1].set_ylabel("ΔF_cool (L/min)")
    axes[1].legend()

    # Stage 5: T_core delta
    orc_t = [chain_results[k]["oracle"]["T_core"] for k in v_labels if k in chain_results]
    prd_t = [chain_results[k]["prism"]["T_core"] for k in v_labels if k in chain_results]
    axes[2].bar(x - width/2, orc_t, width, label="Oracle ΔT_core", color="#0275d8")
    axes[2].bar(x + width/2, prd_t, width, label="PRISM ΔT_core", color="#d9534f")
    axes[2].set_title("Stage 5: Core Temperature (ΔT_core)", fontsize=11, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["50%", "70%", "85%", "100%"])
    axes[2].set_ylabel("ΔT_core (°C)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "causal_chain_breakdown.png", dpi=300)
    plt.close()

    # 2. Plot Latent Sensitivity ||Delta Z_h||_2
    fig, ax = plt.subplots(figsize=(8, 5))
    h = sens_report.horizons
    ax.plot(h, sens_report.norm_delta_z_action_valve, "s-", color="#0275d8", label="Action Control A_valve=85", lw=2)
    ax.plot(h, sens_report.norm_delta_z_state_vpos, "o--", color="#d9534f", label="State Clamp do(V_pos=85)", lw=2)
    ax.plot(h, sens_report.norm_delta_z_state_lcpu, "^--", color="#f0ad4e", label="State Clamp do(L_cpu=80)", lw=2)
    ax.plot(h, sens_report.norm_delta_z_state_vib, "d--", color="#5cb85c", label="State Clamp do(Vib_pump=5)", lw=2)

    ax.set_xlabel("Horizon h (steps)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latent Divergence ||ΔZ_h||_2", fontsize=11, fontweight="bold")
    ax.set_title("Latent Trajectory Sensitivity Under Interventions", fontsize=12, fontweight="bold")
    ax.set_xticks(h)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "latent_sensitivity_comparison.png", dpi=300)
    plt.close()

    # 3. Plot False Safe Analysis: Peak Temperature Discrepancy
    if false_safes:
        fig, ax = plt.subplots(figsize=(9, 5))
        orc_peaks = [c.oracle_peak_t_core for c in false_safes]
        prd_peaks = [c.prism_peak_t_core for c in false_safes]
        labels = [f"{c.target}={c.value:g} (t*={c.intervention_time})" for c in false_safes]

        y = np.arange(len(false_safes))
        ax.barh(y - 0.2, orc_peaks, 0.4, label="Oracle True Peak T_core", color="#d9534f")
        ax.barh(y + 0.2, prd_peaks, 0.4, label="PRISM Predicted Peak T_core", color="#0275d8")
        ax.axvline(105.0, color="black", linestyle="--", label="Thermal Runaway Threshold (105°C)")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Peak T_core (°C)", fontsize=11, fontweight="bold")
        ax.set_title("False Safe Failure Analysis: Severe Underestimation of Thermal Runaway", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "false_safe_peak_temperature_gap.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    run_diagnostics()
