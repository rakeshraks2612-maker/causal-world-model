"""Benchmark Evaluation of Learned PRISM World Model on Causal Interventions.

Executes:
- Evaluates frozen baseline_002 against all 220 Task 2.7 Oracle Intervention Records
- Measures causal effect errors: E_causal = |Delta Y_learned - Delta Y_oracle|
- Tests all 10 causal invariants
- Exports JSON report to artifacts/intervention/intervention_benchmark_report.json
- Generates publication-grade diagnostic plots in artifacts/intervention/plots/
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
from prism.dataset.interventions import LearnerInterventionRecord, OracleInterventionRecord
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.evaluation.intervention_metrics import (
    evaluate_single_intervention,
    aggregate_intervention_benchmark,
    InterventionBenchmarkSummary,
    PairedInterventionEvaluation,
)


def run_learned_intervention_benchmark(
    checkpoint_dir: str = "artifacts/baseline_002",
    oracle_dir: str = "data/pilot/oracle/intervention",
    learner_dir: str = "data/pilot/learner/intervention",
    output_dir: str = "artifacts/intervention",
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> InterventionBenchmarkSummary:
    """Run full causal intervention evaluation against all oracle records."""
    art_path = Path(output_dir)
    plots_path = art_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint_dir)
    print(f"Loading frozen PRISM model from {ckpt_path}...")
    normalizer = ObservationNormalizer.load_yaml(ckpt_path / "normalization.yaml")
    config = WorldModelConfig.from_yaml(ckpt_path / "config.yaml")
    model = CausalWorldModel(config)
    checkpoint = torch.load(ckpt_path / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    simulator = LearnedInterventionSimulator(model, normalizer)

    oracle_files = sorted(list(Path(oracle_dir).glob("*.npz")))
    learner_files = sorted(list(Path(learner_dir).glob("*.npz")))

    print(f"Found {len(oracle_files)} Oracle records and {len(learner_files)} Learner records.")
    if not oracle_files:
        raise FileNotFoundError(f"No oracle files found in {oracle_dir}")

    evaluations: List[PairedInterventionEvaluation] = []

    print("Simulating learned interventions across dataset...")
    for i, orc_file in enumerate(oracle_files):
        inv_id = orc_file.stem
        learn_file = Path(learner_dir) / f"{inv_id}.npz"
        if not learn_file.exists():
            continue

        learn_rec = LearnerInterventionRecord.load_npz(learn_file)
        result = simulator.simulate_from_learner_record(
            record=learn_rec,
            horizons=horizons,
            deterministic=True,
        )

        eval_res = evaluate_single_intervention(result, orc_file)
        evaluations.append(eval_res)

        if (i + 1) % 50 == 0 or (i + 1) == len(oracle_files):
            print(f"  Processed {i + 1}/{len(oracle_files)} intervention experiments...")

    # Aggregate results
    summary = aggregate_intervention_benchmark(evaluations, horizons=horizons)

    # 1. Print formatted tables
    print("\n" + "=" * 105)
    print(" " * 32 + "LEARNED PRISM INTERVENTION BENCHMARK")
    print("=" * 105)
    print(f"Total Paired Experiments: {summary.total_records}")
    print(f"Overall Mean Causal Error E_causal: {summary.overall_mean_causal_error:.4f}")
    print(f"Overall Directional Concordance:    {summary.overall_directional_accuracy:.1%}")
    print(f"Peak T_core MAE:                    {summary.peak_t_core_mae:.2f}°C")
    print(f"Max Pressure MAE:                   {summary.max_pressure_mae:.3f} bar")
    print(f"Min Flow MAE:                       {summary.min_flow_mae:.2f} L/min")
    print(f"Failure Classification Accuracy:    {summary.failure_accuracy:.1%}")
    print("-" * 105)

    print("\n" + "=" * 90)
    print(" " * 25 + "CAUSAL EFFECT ACCURACY AT HORIZON h = 10")
    print("=" * 90)
    print(summary.format_table(horizon=10))
    print("=" * 90)

    print("\n" + "=" * 90)
    print(" " * 25 + "CAUSAL ERROR E_causal BY FORECAST HORIZON")
    print("=" * 90)
    print(f"{'Horizon':<8s} | {'ΔT_core (°C)':>13s} | {'ΔT_cool (°C)':>13s} | {'ΔP_sys (b)':>11s} | {'ΔF_cool (L)':>12s} | {'Mean E_causal':>14s}")
    print("-" * 90)
    for h in horizons:
        h_dict = summary.mean_causal_error_by_horizon[h]
        mean_h = float(np.mean(list(h_dict.values())))
        print(
            f"h = {h:<4d} | {h_dict['delta_t_core']:>12.3f}°C | {h_dict['delta_t_cool']:>12.3f}°C | "
            f"{h_dict['delta_p_sys']:>10.3f}b | {h_dict['delta_f_cool']:>11.3f}L | {mean_h:>13.4f}"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print(" " * 25 + "BREAKDOWN BY INTERVENTION TARGET VARIABLE")
    print("=" * 90)
    print(f"{'Target Variable':<18s} | {'Records':>8s} | {'Mean E_causal':>15s} | {'Peak T_core MAE':>18s}")
    print("-" * 90)
    for tgt, stats in summary.breakdown_by_target.items():
        print(f"do({tgt:<14s}) | {stats['count']:>8d} | {stats['mean_causal_error']:>15.4f} | {stats['peak_t_core_mae']:>17.2f}°C")
    print("=" * 90)

    # 2. Save JSON report
    report_data = summary.to_dict()
    report_file = art_path / "intervention_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✓ Saved JSON benchmark report to {report_file}")

    # 3. Generate Publication Plots
    _generate_plots(summary, plots_path)
    print(f"✓ Diagnostic plots generated in {plots_path}")

    return summary


def _generate_plots(summary: InterventionBenchmarkSummary, output_dir: Path) -> None:
    """Generate diagnostic visualization plots comparing Learned PRISM vs Oracle."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Plot Causal Error by Horizon
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = summary.horizons
    dt_core_errs = [summary.mean_causal_error_by_horizon[h]["delta_t_core"] for h in horizons]
    dt_cool_errs = [summary.mean_causal_error_by_horizon[h]["delta_t_cool"] for h in horizons]
    dp_sys_errs = [summary.mean_causal_error_by_horizon[h]["delta_p_sys"] for h in horizons]
    df_cool_errs = [summary.mean_causal_error_by_horizon[h]["delta_f_cool"] for h in horizons]

    ax.plot(horizons, dt_core_errs, "o-", label="ΔT_core Error (°C)", color="#d9534f", lw=2)
    ax.plot(horizons, dt_cool_errs, "s-", label="ΔT_cool Error (°C)", color="#f0ad4e", lw=2)
    ax.plot(horizons, df_cool_errs, "^-", label="ΔF_cool Error (L/min)", color="#0275d8", lw=2)
    ax.plot(horizons, dp_sys_errs, "d-", label="ΔP_sys Error (bar)", color="#5cb85c", lw=2)

    ax.set_xlabel("Forecast Horizon h (timesteps)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Causal Effect Error E_causal", fontsize=11, fontweight="bold")
    ax.set_title("Learned PRISM vs Oracle: Causal Error by Horizon", fontsize=12, fontweight="bold")
    ax.set_xticks(horizons)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "causal_error_by_horizon.png", dpi=300)
    plt.close()

    # 2. Plot Valve and CPU Load Intervention Ladder Response
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Valve responses at h=10
    v_pos_vals = [50.0, 70.0, 85.0, 100.0]
    orc_flows = []
    prd_flows = []
    orc_temps = []
    prd_temps = []

    for val in v_pos_vals:
        sub = [e for e in summary.evaluations if e.target == "V_pos" and e.value == val]
        if sub:
            orc_flows.append(np.mean([e.oracle_deltas[10]["delta_f_cool"] for e in sub if 10 in e.oracle_deltas]))
            prd_flows.append(np.mean([e.learned_deltas[10]["delta_f_cool"] for e in sub if 10 in e.learned_deltas]))
            orc_temps.append(np.mean([e.oracle_deltas[10]["delta_t_core"] for e in sub if 10 in e.oracle_deltas]))
            prd_temps.append(np.mean([e.learned_deltas[10]["delta_t_core"] for e in sub if 10 in e.learned_deltas]))

    ax1.plot(v_pos_vals, orc_flows, "o--", color="#0275d8", label="Oracle ΔF_cool", lw=2)
    ax1.plot(v_pos_vals, prd_flows, "s-", color="#0275d8", label="Learned ΔF_cool", lw=2)
    ax1.plot(v_pos_vals, orc_temps, "o--", color="#d9534f", label="Oracle ΔT_core", lw=2)
    ax1.plot(v_pos_vals, prd_temps, "s-", color="#d9534f", label="Learned ΔT_core", lw=2)
    ax1.set_xlabel("Intervention do(V_pos = v %)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Causal Delta at h=10", fontsize=11, fontweight="bold")
    ax1.set_title("Valve Opening Intervention Ladder (h=10)", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # CPU load responses at h=10
    cpu_vals = [20.0, 50.0, 80.0]
    orc_cpu_temps = []
    prd_cpu_temps = []

    for val in cpu_vals:
        sub = [e for e in summary.evaluations if e.target == "L_cpu" and e.value == val]
        if sub:
            orc_cpu_temps.append(np.mean([e.oracle_deltas[10]["delta_t_core"] for e in sub if 10 in e.oracle_deltas]))
            prd_cpu_temps.append(np.mean([e.learned_deltas[10]["delta_t_core"] for e in sub if 10 in e.learned_deltas]))

    ax2.plot(cpu_vals, orc_cpu_temps, "o--", color="#d9534f", label="Oracle ΔT_core", lw=2)
    ax2.plot(cpu_vals, prd_cpu_temps, "s-", color="#d9534f", label="Learned ΔT_core", lw=2)
    ax2.set_xlabel("Intervention do(L_cpu = l %)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ΔT_core at h=10 (°C)", fontsize=11, fontweight="bold")
    ax2.set_title("CPU Load Intervention Ladder (h=10)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "intervention_ladder_curves.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_learned_intervention_benchmark()
