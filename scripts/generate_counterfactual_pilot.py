"""Script to generate the Task 2.8 Counterfactual Pilot Dataset and Benchmark Table.

Generates ~480 paired counterfactual records across:
- 10 historical baseline episodes
- 4 counterfactual times: t* in {20, 40, 60, 80}
- 4 action families: A_valve, A_throttle, A_pump, A_flush
- Multiple distinct alternate action values per family
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import List, Dict, Any

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.dataset.counterfactuals import (
    CounterfactualSpec,
    OracleCounterfactualRecord,
    execute_counterfactual_experiment,
)


def generate_counterfactual_pilot(output_dir: str = "data/counterfactual_pilot") -> List[OracleCounterfactualRecord]:
    out_path = Path(output_dir)
    learner_dir = out_path / "learner"
    oracle_dir = out_path / "oracle"
    learner_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    times = [20, 40, 60, 80]
    action_grids = {
        "A_valve": [50.0, 70.0, 85.0, 100.0],
        "A_throttle": [20.0, 50.0, 80.0, 100.0],
        "A_pump": [1.0, 2.0, 3.0, 4.0],
        "A_flush": [0.0, 1.0],
    }

    records: List[OracleCounterfactualRecord] = []
    
    # 16 historical baseline episodes across nominal, ambient, wear, and high load variations
    regimes = [
        "nominal", "nominal", "moderate_load", "moderate_load", "moderate_ambient",
        "moderate_ambient", "moderate_wear", "moderate_wear", "nominal", "moderate_load",
        "moderate_ambient", "moderate_wear", "nominal", "moderate_load", "moderate_ambient", "moderate_wear"
    ]

    print(f"Generating counterfactual pilot across {len(regimes)} historical episodes...")

    for ep_idx, regime in enumerate(regimes):
        base_ep = generate_single_episode(
            split=SplitType.TEST,
            index=ep_idx,
            regime=regime,
            length=100,
        )

        for t_star in times:
            for act_name, candidate_vals in action_grids.items():
                act_idx = {"A_valve": 0, "A_throttle": 1, "A_pump": 2, "A_flush": 3}[act_name]
                curr_val = base_ep.actions[t_star, act_idx]

                # Filter out values close to current factual value
                valid_alternates = [v for v in candidate_vals if not np.isclose(v, curr_val, atol=1.0)]

                # For binary flush, if 1 alternate, take it; for continuous/multi-stage, take up to 3
                if act_name == "A_flush":
                    selected_vals = valid_alternates[:1]
                else:
                    selected_vals = valid_alternates[:3]

                for alt_val in selected_vals:
                    spec = CounterfactualSpec(
                        target_action=act_name,
                        counterfactual_value=float(alt_val),
                        counterfactual_time=t_star,
                    )
                    rec = execute_counterfactual_experiment(base_ep, spec)
                    records.append(rec)

                    # Save paired records
                    rec.save_npz(oracle_dir / f"{rec.counterfactual_id}.npz")
                    learner_rec = rec.to_learner_record()
                    learner_rec.save_npz(learner_dir / f"{learner_rec.counterfactual_id}.npz")

    print(f"Generated {len(records)} counterfactual records successfully!")
    return records


def print_counterfactual_benchmark_table(records: List[OracleCounterfactualRecord]) -> None:
    """Print the aggregate counterfactual benchmark table for Task 2.8 audit."""
    print("\n" + "=" * 115)
    print("TASK 2.8 — COUNTERFACTUAL BENCHMARK TABLE (PEARL LEVEL 3)")
    print("=" * 115)
    print(f"{'Action Target':<14} | {'Factual':<12} | {'Counterfactual':<14} | {'Horizon':<7} | {'Mean ΔT_core':<13} | {'Mean ΔT_cool':<13} | {'Mean ΔP_sys':<12} | {'Mean ΔF_cool':<13} | {'Failure Δ':<10}")
    print("-" * 115)

    # Benchmark test cases across immediate (h=1) and downstream (h=5, h=10) horizons
    benchmark_queries = [
        # Valve changes
        ("A_valve", "mid (~55)", 85.0, 1, lambda r: r.spec.target_action == "A_valve" and r.spec.counterfactual_value == 85.0),
        ("A_valve", "mid (~55)", 85.0, 5, lambda r: r.spec.target_action == "A_valve" and r.spec.counterfactual_value == 85.0),
        ("A_valve", "mid (~55)", 85.0, 10, lambda r: r.spec.target_action == "A_valve" and r.spec.counterfactual_value == 85.0),
        ("A_valve", "mid (~55)", 100.0, 1, lambda r: r.spec.target_action == "A_valve" and r.spec.counterfactual_value == 100.0),
        ("A_valve", "mid (~55)", 100.0, 10, lambda r: r.spec.target_action == "A_valve" and r.spec.counterfactual_value == 100.0),
        
        # Throttle changes
        ("A_throttle", "100 (unthrot)", 80.0, 1, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 80.0),
        ("A_throttle", "100 (unthrot)", 80.0, 5, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 80.0),
        ("A_throttle", "100 (unthrot)", 80.0, 10, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 80.0),
        ("A_throttle", "100 (unthrot)", 20.0, 1, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 20.0),
        ("A_throttle", "100 (unthrot)", 20.0, 5, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 20.0),
        ("A_throttle", "100 (unthrot)", 20.0, 10, lambda r: r.spec.target_action == "A_throttle" and r.spec.counterfactual_value == 20.0),

        # Pump changes
        ("A_pump", "stage 2", 4.0, 1, lambda r: r.spec.target_action == "A_pump" and r.spec.counterfactual_value == 4.0),
        ("A_pump", "stage 2", 4.0, 5, lambda r: r.spec.target_action == "A_pump" and r.spec.counterfactual_value == 4.0),
        ("A_pump", "stage 2", 1.0, 1, lambda r: r.spec.target_action == "A_pump" and r.spec.counterfactual_value == 1.0),
        ("A_pump", "stage 2", 1.0, 5, lambda r: r.spec.target_action == "A_pump" and r.spec.counterfactual_value == 1.0),

        # Flush changes
        ("A_flush", "0 (off)", 1.0, 1, lambda r: r.spec.target_action == "A_flush" and r.spec.counterfactual_value == 1.0),
        ("A_flush", "0 (off)", 1.0, 5, lambda r: r.spec.target_action == "A_flush" and r.spec.counterfactual_value == 1.0),
    ]

    for target, fact_desc, cf_val, h, filter_fn in benchmark_queries:
        matched = [r for r in records if filter_fn(r) and h in r.horizon_effects]
        if matched:
            d_tcore = float(np.mean([r.horizon_effects[h].delta_t_core for r in matched]))
            d_tcool = float(np.mean([r.horizon_effects[h].delta_t_cool for r in matched]))
            d_psys = float(np.mean([r.horizon_effects[h].delta_p_sys for r in matched]))
            d_fcool = float(np.mean([r.horizon_effects[h].delta_f_cool for r in matched]))
            fail_delta = float(np.mean([r.failure_metrics.absolute_failure_probability_delta for r in matched]))

            print(f"{target:<14} | {fact_desc:<12} | {cf_val:<14.1f} | {h:<7} | {d_tcore:>+10.2f}°C   | {d_tcool:>+10.2f}°C   | {d_psys:>+9.2f} bar | {d_fcool:>+9.2f} L/m  | {fail_delta:>+8.1%}")

    print("=" * 115)


if __name__ == "__main__":
    recs = generate_counterfactual_pilot()
    print_counterfactual_benchmark_table(recs)
