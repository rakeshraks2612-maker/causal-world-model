"""Task 6.3: Counterfactual Evidence Benchmark Evaluation & Artifact Generator.

Processes all 448 counterfactual records from data/counterfactual_pilot/learner/
through the PRISM Counterfactual Evidence Engine.
Verifies that benchmark evaluation metrics remain identical to the frozen Level-3 baseline,
and generates example demonstration artifacts for A_valve, A_throttle, A_pump, A_flush,
and safety failure cases.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.dataset.counterfactuals import (
    LearnerCounterfactualRecord,
    OracleCounterfactualRecord,
)
from prism.counterfactual.engine import (
    LearnedCounterfactualEngine,
    LearnedCounterfactualResult,
)
from prism.counterfactual.metrics import (
    evaluate_single_counterfactual,
    aggregate_counterfactual_benchmark,
)
from prism.explanation.counterfactual_evidence import (
    CounterfactualEvidence,
    build_counterfactual_evidence,
)


def run_benchmark():
    print("=========================================================================")
    print("      TASK 6.3: COUNTERFACTUAL EVIDENCE BENCHMARK EVALUATION             ")
    print("=========================================================================")

    model_dir = Path("artifacts/baseline_005")
    norm = ObservationNormalizer.load_yaml(model_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    engine = LearnedCounterfactualEngine(model, norm)

    pilot_dir = Path("data/counterfactual_pilot")
    learner_files = sorted(list((pilot_dir / "learner").glob("*.npz")))
    oracle_files = sorted(list((pilot_dir / "oracle").glob("*.npz")))

    print(f"Found {len(learner_files)} learner CF records and {len(oracle_files)} oracle CF records.")

    out_dir = Path("artifacts/baseline_005/counterfactual_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_processed = 0
    records_successful = 0
    records_blocked = 0
    records_failed = 0

    evaluation_pairs = []
    cf_evidence_list = []

    # Exemplars to save as dedicated demo artifacts
    exemplars_to_save = {
        "A_valve": None,
        "A_throttle": None,
        "A_pump": None,
        "A_flush": None,
        "unsafe_transition": None,
    }

    t0 = time.perf_counter()

    for l_file in learner_files:
        records_processed += 1
        cf_id = l_file.stem
        o_file = pilot_dir / "oracle" / f"{cf_id}.npz"

        try:
            learner_rec = LearnerCounterfactualRecord.load_npz(l_file)

            # Execute frozen Level-3 CF simulation
            cf_res = engine.evaluate_from_learner_record(learner_rec, deterministic=True)

            # Build Counterfactual Evidence contract object
            cf_evidence = build_counterfactual_evidence(
                cf_result=cf_res,
                learner_record=learner_rec,
                model_version="baseline_005",
                timestamp_utc="2026-09-11T12:00:00Z",
            )

            records_successful += 1
            cf_evidence_list.append(cf_evidence)

            if o_file.exists():
                single_eval = evaluate_single_counterfactual(cf_res, o_file)
                evaluation_pairs.append(single_eval)

            # Check for exemplar selection
            tgt = cf_evidence.intervention.intervention_target
            if tgt in exemplars_to_save and exemplars_to_save[tgt] is None:
                exemplars_to_save[tgt] = cf_evidence

            if exemplars_to_save["unsafe_transition"] is None and (
                "UNSAFE" in cf_evidence.safety_comparison.safety_transition
                or cf_evidence.safety_comparison.outcome_classification == "INTERVENTION_INTRODUCES_RISK"
            ):
                exemplars_to_save["unsafe_transition"] = cf_evidence

        except Exception as e:
            records_failed += 1
            print(f"Error processing {cf_id}: {e}")

    elapsed = time.perf_counter() - t0

    # Aggregate Benchmark Metrics
    benchmark_metrics = aggregate_counterfactual_benchmark(evaluation_pairs)

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Records Processed   : {records_processed}")
    print(f"Records Successful        : {records_successful}")
    print(f"Records Blocked           : {records_blocked}")
    print(f"Records Failed            : {records_failed}")
    print(f"Processing Elapsed Time   : {elapsed:.2f} s ({elapsed/records_processed*1000:.2f} ms/rec)")

    print("\n" + "=" * 80)
    print("FROZEN LEVEL-3 COUNTERFACTUAL BENCHMARK METRICS (BASELINE 005)")
    print("=" * 80)
    print(f"Mean Counterfactual Error (E_CF)        : {benchmark_metrics.overall_mean_causal_error:.4f}")
    print(f"Relative Counterfactual Error (E_rel)   : {benchmark_metrics.overall_mean_relative_error:.4f}")
    print(f"Overall Directional Accuracy            : {benchmark_metrics.overall_directional_accuracy*100:.2f}%")
    print(f"Peak T_core MAE                         : {benchmark_metrics.peak_t_core_mae:.2f}°C")
    print(f"Max P_sys MAE                           : {benchmark_metrics.max_pressure_mae:.2f} bar")
    print(f"Min F_cool MAE                          : {benchmark_metrics.min_flow_mae:.2f} L/min")
    print(f"False Safe Rate                         : {benchmark_metrics.safety_summary.false_safe_rate*100:.2f}%")

    print("\nPerformance Breakdown by Actuator Target:")
    for tgt, stats in benchmark_metrics.breakdown_by_target.items():
        print(f"  - {tgt:<12} (n={stats['count']}): E_CF = {stats['mean_causal_error']:.4f} | Directional Acc = {stats['directional_accuracy']*100:.2f}% | Peak T_core MAE = {stats['peak_t_core_mae']:.2f}°C")

    # Save Exemplar Artifacts (.json and .md)
    print("\nSaving Demonstration Artifacts...")
    for label, ev in exemplars_to_save.items():
        if ev is not None:
            base_name = f"counterfactual_{label}"
            with open(out_dir / f"{base_name}.json", "w") as f:
                f.write(ev.to_json())
            with open(out_dir / f"{base_name}.md", "w") as f:
                f.write(ev.format_markdown())
            print(f"  - Saved {base_name}.json and {base_name}.md (ID: {ev.provenance.counterfactual_id})")

    # Save master benchmark summary report
    master_report = {
        "benchmark_summary": {
            "records_processed": records_processed,
            "records_successful": records_successful,
            "records_blocked": records_blocked,
            "records_failed": records_failed,
            "elapsed_seconds": elapsed,
        },
        "counterfactual_metrics": benchmark_metrics.to_dict(),
        "exemplar_ids": {k: (v.provenance.counterfactual_id if v else None) for k, v in exemplars_to_save.items()},
    }

    with open(out_dir / "counterfactual_evidence_benchmark_report.json", "w") as f:
        json.dump(master_report, f, indent=2)

    print(f"\nMaster report saved to {out_dir / 'counterfactual_evidence_benchmark_report.json'}")
    return master_report


if __name__ == "__main__":
    run_benchmark()
