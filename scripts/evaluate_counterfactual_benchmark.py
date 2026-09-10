"""Comprehensive Counterfactual Benchmark Evaluation (Task 3.6).

Runs the LearnedCounterfactualEngine across all 448 paired counterfactual records in data/counterfactual_pilot/.
Computes:
1. Multi-horizon counterfactual effect error E_CF and relative error E_rel across h in {1, 5, 10, 20, 40}.
2. Overall directional concordance (sign agreement on treatment effects).
3. Peak metric errors (peak core temperature, max pressure, min flow).
4. Failure risk prediction accuracy.
5. Saves full JSON benchmark report to artifacts/evaluation_counterfactual/counterfactual_benchmark_report.json.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.counterfactuals import LearnerCounterfactualRecord, OracleCounterfactualRecord
from prism.counterfactual.engine import LearnedCounterfactualEngine
from prism.counterfactual.metrics import (
    evaluate_single_counterfactual,
    aggregate_counterfactual_benchmark,
    CounterfactualBenchmarkSummary,
)


def main() -> None:
    output_dir = Path("artifacts/evaluation_counterfactual")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("      PRISM LEVEL-3 COUNTERFACTUAL BENCHMARK EVALUATION (Task 3.6)       ")
    print("=========================================================================")

    # 1. Load Baseline 003 World Model
    b3_dir = Path("artifacts/baseline_003")
    norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(b3_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    engine = LearnedCounterfactualEngine(model, norm)

    # 2. Discover records in data/counterfactual_pilot
    pilot_dir = Path("data/counterfactual_pilot")
    oracle_dir = pilot_dir / "oracle"
    learner_dir = pilot_dir / "learner"
    
    oracle_files = sorted(list(oracle_dir.glob("*.npz")))
    total_files = len(oracle_files)
    print(f"Discovered {total_files} counterfactual benchmark records in {pilot_dir}.")

    # 3. Evaluate each record
    evaluations = []
    horizons = (1, 5, 10, 20, 40)

    print("Running Twin-World Abduction, Substitution, and Counterfactual Replay...")
    for idx, orc_file in enumerate(oracle_files):
        learn_file = learner_dir / orc_file.name
        learn_rec = LearnerCounterfactualRecord.load_npz(learn_file)
        
        result = engine.evaluate_from_learner_record(learn_rec, horizons=horizons, deterministic=True)
        eval_res = evaluate_single_counterfactual(result, orc_file)
        evaluations.append(eval_res)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
            print(f"  Processed {idx + 1:3d} / {total_files} records...")

    # 4. Aggregate Benchmark Summary
    summary = aggregate_counterfactual_benchmark(evaluations, horizons=horizons)

    # 5. Export Report
    report_dict = summary.to_dict()
    report_path = output_dir / "counterfactual_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # 6. Print Summary Table
    print("\n=========================================================================")
    print("                COUNTERFACTUAL BENCHMARK SUMMARY TABLE                   ")
    print("=========================================================================")
    print(f"Total Counterfactual Records Evaluated: {summary.total_records}")
    print(f"Overall Counterfactual Error E_CF:      {summary.overall_mean_causal_error:.4f}")
    print(f"Overall Relative Error E_rel:           {summary.overall_mean_relative_error*100:.1f}%")
    print(f"Overall Directional Concordance:        {summary.overall_directional_accuracy*100:.1f}%")
    print(f"Peak T_core MAE:                        {summary.peak_t_core_mae:.2f}°C")
    print(f"Max P_sys MAE:                          {summary.max_pressure_mae:.3f} bar")
    print(f"Min F_cool MAE:                         {summary.min_flow_mae:.2f} L/min")
    print("-------------------------------------------------------------------------")
    print("BREAKDOWN BY ACTION TARGET:")
    for tgt, stats in summary.breakdown_by_target.items():
        print(f"  {tgt:<12} | Records: {stats['count']:3d} | Mean E_CF: {stats['mean_causal_error']:.4f} | Sign Accuracy: {stats['directional_accuracy']*100:.1f}% | Peak T_core MAE: {stats['peak_t_core_mae']:.2f}°C")
    print("=========================================================================")
    print(f"✓ Saved full report to {report_path}")


if __name__ == "__main__":
    main()
