"""Comprehensive Comparative Evaluation of Baseline 002 vs Baseline 003 (Task 3.4E).

Evaluates:
1. Multi-horizon open-loop forecasting (h in {1, 5, 10, 20, 40}) on held-out test split.
2. Causal intervention effect accuracy (E_causal, E_rel, directional concordance) across all 220 held-out records.
3. Safety boundary classification (TP, FN, FP, TN, and False-Safe Rate on 16 thermal runaway cases).
4. Exports full JSON report to artifacts/evaluation_baseline_003/baseline_003_comparison_report.json and plots.
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
from prism.dataset.schema import LearnerEpisode, OracleEpisode
from prism.dataset.interventions import LearnerInterventionRecord, OracleInterventionRecord
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.evaluation.intervention_metrics import (
    evaluate_single_intervention,
    aggregate_intervention_benchmark,
    InterventionBenchmarkSummary,
)
from prism.simulator.state import OBSERVABLE_VARIABLES


def evaluate_forecasting(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    test_files: List[Path],
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
    context_length: int = 40,
) -> Dict[str, Any]:
    """Evaluate multi-horizon open-loop autoregressive rollout on test episodes."""
    model.eval()
    horizon_maes: Dict[int, List[float]] = {h: [] for h in horizons}
    channel_maes: Dict[str, Dict[int, List[float]]] = {var: {h: [] for h in horizons} for var in OBSERVABLE_VARIABLES}

    with torch.no_grad():
        for f in test_files:
            ep = LearnerEpisode.load_npz(f)
            t_total = len(ep.timestamps)
            max_h = max(horizons)
            if t_total < context_length + max_h:
                continue

            # Context window: [1, context_length, 8]
            ctx_obs = torch.tensor(ep.observations[:context_length], dtype=torch.float32).unsqueeze(0)
            norm_ctx_obs = normalizer.normalize(ctx_obs)
            ctx_mask = torch.tensor(ep.observation_mask[:context_length], dtype=torch.float32).unsqueeze(0)
            ctx_act = torch.tensor(ep.actions[:context_length], dtype=torch.float32).unsqueeze(0)

            from prism.world_model.inputs import ModelInputs
            inputs = ModelInputs(observations=norm_ctx_obs, observation_mask=ctx_mask, actions=ctx_act)
            post_latents, _ = model.encode(inputs)
            z_t0 = post_latents.mean[:, -1]

            # Future action sequence for rollout: [1, max_h + 1, 4]
            fut_act = torch.tensor(ep.actions[context_length - 1 : context_length + max_h], dtype=torch.float32).unsqueeze(0)
            
            traj = model.rollout_manager.rollout_deterministic(initial_z=z_t0, actions=fut_act)
            pred_obs_norm = traj.observations.mean.cpu()  # [1, max_h + 1, 8]
            pred_obs_phys = normalizer.denormalize(pred_obs_norm).numpy()[0]  # [max_h + 1, 8]

            true_obs_phys = ep.observations[context_length - 1 : context_length + max_h]

            for h in horizons:
                step_idx = h
                pred_h = pred_obs_phys[step_idx]
                true_h = true_obs_phys[step_idx]
                abs_err = np.abs(pred_h - true_h)
                horizon_maes[h].append(float(np.mean(abs_err)))
                for ch_i, var in enumerate(OBSERVABLE_VARIABLES):
                    channel_maes[var][h].append(float(abs_err[ch_i]))

    return {
        "mean_mae_by_horizon": {f"h_{h}": float(np.mean(horizon_maes[h])) for h in horizons},
        "channel_mae_by_horizon": {
            var: {f"h_{h}": float(np.mean(channel_maes[var][h])) for h in horizons}
            for var in OBSERVABLE_VARIABLES
        }
    }


def evaluate_intervention_suite(
    model: CausalWorldModel,
    normalizer: ObservationNormalizer,
    oracle_dir: Path = Path("data/pilot/oracle/intervention"),
    learner_dir: Path = Path("data/pilot/learner/intervention"),
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> Tuple[InterventionBenchmarkSummary, List[Any]]:
    """Run full 220-record intervention benchmark."""
    simulator = LearnedInterventionSimulator(model, normalizer)
    oracle_files = sorted(list(oracle_dir.glob("*.npz")))

    evaluations = []
    for orc_file in oracle_files:
        learn_file = learner_dir / orc_file.name
        learn_rec = LearnerInterventionRecord.load_npz(learn_file)
        res = simulator.simulate_from_learner_record(learn_rec, horizons=horizons, deterministic=True)
        eval_res = evaluate_single_intervention(res, orc_file)
        evaluations.append(eval_res)

    summary = aggregate_intervention_benchmark(evaluations, horizons=horizons)
    return summary, evaluations


def main() -> None:
    output_dir = Path("artifacts/evaluation_baseline_003")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("      PRISM BASELINE 003 vs BASELINE 002 BENCHMARK EVALUATION            ")
    print("=========================================================================")

    # 1. Load Baseline 002
    b2_dir = Path("artifacts/baseline_002")
    b2_norm = ObservationNormalizer.load_yaml(b2_dir / "normalization.yaml")
    b2_cfg = WorldModelConfig.from_yaml(b2_dir / "config.yaml")
    b2_model = CausalWorldModel(b2_cfg)
    b2_ckpt = torch.load(b2_dir / "best.pt", map_location="cpu")
    b2_model.load_state_dict(b2_ckpt["model_state_dict"])
    b2_model.eval()

    # 2. Load Baseline 003
    b3_dir = Path("artifacts/baseline_003")
    b3_norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    b3_cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    b3_model = CausalWorldModel(b3_cfg)
    b3_ckpt = torch.load(b3_dir / "best.pt", map_location="cpu")
    b3_model.load_state_dict(b3_ckpt["model_state_dict"])
    b3_model.eval()

    # 3. Forecasting Evaluation on test split
    test_files = sorted(list(Path("data/pilot/learner/test").glob("*.npz")))
    print(f"\n1. Evaluating Forecasting across {len(test_files)} test episodes...")
    b2_forecast = evaluate_forecasting(b2_model, b2_norm, test_files)
    b3_forecast = evaluate_forecasting(b3_model, b3_norm, test_files)

    # 4. Intervention & Safety Benchmark on 220 records
    print("2. Running complete 220-record Causal Intervention Benchmark...")
    b2_int_summary, b2_int_evals = evaluate_intervention_suite(b2_model, b2_norm)
    b3_int_summary, b3_int_evals = evaluate_intervention_suite(b3_model, b3_norm)

    # 5. Compile Comparison Report
    comparison = {
        "forecasting": {
            "baseline_002": b2_forecast,
            "baseline_003": b3_forecast,
        },
        "causal_intervention": {
            "baseline_002": {
                "overall_causal_error": b2_int_summary.overall_mean_causal_error,
                "overall_relative_error": b2_int_summary.overall_mean_relative_error,
                "overall_directional_concordance": b2_int_summary.overall_directional_accuracy,
                "peak_t_core_mae": b2_int_summary.peak_t_core_mae,
                "max_pressure_mae": b2_int_summary.max_pressure_mae,
                "min_flow_mae": b2_int_summary.min_flow_mae,
                "safety": b2_int_summary.safety_summary.to_dict(),
            },
            "baseline_003": {
                "overall_causal_error": b3_int_summary.overall_mean_causal_error,
                "overall_relative_error": b3_int_summary.overall_mean_relative_error,
                "overall_directional_concordance": b3_int_summary.overall_directional_accuracy,
                "peak_t_core_mae": b3_int_summary.peak_t_core_mae,
                "max_pressure_mae": b3_int_summary.max_pressure_mae,
                "min_flow_mae": b3_int_summary.min_flow_mae,
                "safety": b3_int_summary.safety_summary.to_dict(),
            },
        },
    }

    report_path = output_dir / "baseline_003_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print Summary Tables
    print("\n=========================================================================")
    print("                 FORECASTING COMPARISON (Test Split MAE)                 ")
    print("=========================================================================")
    print("| Horizon | Baseline 002 | Baseline 003 | Improvement |")
    print("| :--- | :---: | :---: | :---: |")
    for h in [1, 5, 10, 20, 40]:
        k = f"h_{h}"
        m2 = b2_forecast["mean_mae_by_horizon"][k]
        m3 = b3_forecast["mean_mae_by_horizon"][k]
        diff = m3 - m2
        print(f"| h = {h:2d}   |    {m2:.4f}    |    {m3:.4f}    |   {diff:+.4f} ({diff/m2*100:+.1f}%) |")

    print("\n=========================================================================")
    print("          CAUSAL INTERVENTION BENCHMARK (220 Paired Records)             ")
    print("=========================================================================")
    print(f"Overall Causal Error E_causal:  Baseline 002 = {b2_int_summary.overall_mean_causal_error:.4f}  |  Baseline 003 = {b3_int_summary.overall_mean_causal_error:.4f}")
    print(f"Relative Causal Error E_rel:    Baseline 002 = {b2_int_summary.overall_mean_relative_error*100:.1f}%  |  Baseline 003 = {b3_int_summary.overall_mean_relative_error*100:.1f}%")
    print(f"Directional Concordance:        Baseline 002 = {b2_int_summary.overall_directional_accuracy*100:.1f}%  |  Baseline 003 = {b3_int_summary.overall_directional_accuracy*100:.1f}%")
    print(f"Peak T_core MAE:                Baseline 002 = {b2_int_summary.peak_t_core_mae:.2f}°C  |  Baseline 003 = {b3_int_summary.peak_t_core_mae:.2f}°C")
    print("-------------------------------------------------------------------------")
    print("SAFETY GATE COMPARISON:")
    print(f"  - Ground Truth Critical Failures: 16 (All thermal runaway)")
    print(f"  - Baseline 002: TP = {b2_int_summary.safety_summary.true_critical}, FN = {b2_int_summary.safety_summary.false_safe} (False-Safe Rate: {b2_int_summary.safety_summary.false_safe_rate*100:.1f}%)")
    print(f"  - Baseline 003: TP = {b3_int_summary.safety_summary.true_critical}, FN = {b3_int_summary.safety_summary.false_safe} (False-Safe Rate: {b3_int_summary.safety_summary.false_safe_rate*100:.1f}%)")
    print("=========================================================================")
    print(f"✓ Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()
