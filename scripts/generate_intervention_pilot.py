"""Pilot Intervention Dataset Generator and Oracle Causal Effect Table.

Generates:
- 5 Parent Episodes (Moderate load regime)
- 4 Intervention Timestamps: t* in {20, 40, 60, 80}
- Intervention Matrix:
    - do(V_pos in {50%, 70%, 85%, 100%})
    - do(L_cpu in {20%, 50%, 80%})
    - do(Vib_pump in {0.0, 0.5, 2.0, 5.0})

Computes and prints the Oracle Causal Sanity Check Table at Horizon h = 10.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.dataset.interventions import (
    InterventionSpec,
    InterventionClass,
    execute_intervention_experiment,
    OracleInterventionRecord,
    LearnerInterventionRecord,
)


def run_intervention_pilot(output_dir: str = "data/pilot") -> None:
    base_path = Path(output_dir)
    oracle_inv_dir = base_path / "oracle" / "intervention"
    learner_inv_dir = base_path / "learner" / "intervention"
    oracle_inv_dir.mkdir(parents=True, exist_ok=True)
    learner_inv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Intervention Pilot Dataset to {base_path}...")

    # 1. Generate 5 parent baseline episodes
    parent_episodes = [
        generate_single_episode(SplitType.TEST, index=i, regime="moderate_load", length=120)
        for i in range(5)
    ]

    intervention_matrix = [
        ("V_pos", 50.0),
        ("V_pos", 70.0),
        ("V_pos", 85.0),
        ("V_pos", 100.0),
        ("L_cpu", 20.0),
        ("L_cpu", 50.0),
        ("L_cpu", 80.0),
        ("Vib_pump", 0.0),
        ("Vib_pump", 0.5),
        ("Vib_pump", 2.0),
        ("Vib_pump", 5.0),
    ]

    intervention_times = [20, 40, 60, 80]

    all_oracle_records: List[OracleInterventionRecord] = []
    all_learner_records: List[LearnerInterventionRecord] = []

    for p_idx, parent_ep in enumerate(parent_episodes):
        for t_star in intervention_times:
            for target, val in intervention_matrix:
                spec = InterventionSpec(
                    target=target,
                    value=val,
                    intervention_time=t_star,
                    category=InterventionClass.CLASS_A_ATOMIC_STATE,
                )
                oracle_rec = execute_intervention_experiment(parent_ep, spec, horizons=(1, 5, 10, 20, 40))
                learner_rec = oracle_rec.to_learner_record()

                # Save archives
                oracle_rec.save_npz(oracle_inv_dir / f"{oracle_rec.intervention_id}.npz")
                learner_rec.save_npz(learner_inv_dir / f"{learner_rec.intervention_id}.npz")

                all_oracle_records.append(oracle_rec)
                all_learner_records.append(learner_rec)

    print(f"✓ Successfully generated {len(all_oracle_records)} paired intervention records.")

    # 2. Compute Oracle Causal Sanity Table at Horizon h = 10
    print("\n" + "=" * 95)
    print(" " * 28 + "ORACLE CAUSAL EFFECT TABLE (Horizon h = 10)")
    print("=" * 95)
    print(f"{'Intervention':<20s} | {'Horizon':>7s} | {'Mean ΔT_core':>12s} | {'Mean ΔT_cool':>12s} | {'Mean ΔP_sys':>11s} | {'Mean ΔF_cool':>12s} | {'Failure Δ':>9s}")
    print("-" * 95)

    for target, val in intervention_matrix:
        subset = [r for r in all_oracle_records if r.target == target and r.value == val]
        h10_effs = [r.horizon_effects[10] for r in subset if 10 in r.horizon_effects]

        mean_dt_core = float(np.mean([e.delta_t_core for e in h10_effs]))
        mean_dt_cool = float(np.mean([e.delta_t_cool for e in h10_effs]))
        mean_dp = float(np.mean([e.delta_p_sys for e in h10_effs]))
        mean_df = float(np.mean([e.delta_f_cool for e in h10_effs]))
        fail_delta = float(np.mean([int(e.intervened_failed) - int(e.baseline_failed) for e in h10_effs]))

        int_label = f"do({target}={val:g})"
        print(f"{int_label:<20s} | {10:>7d} | {mean_dt_core:>+11.2f}°C | {mean_dt_cool:>+11.2f}°C | {mean_dp:>+10.2f}b | {mean_df:>+10.2f}L | {fail_delta:>+8.1%}")

    print("=" * 95)
    print("Intervention Pilot complete.")


if __name__ == "__main__":
    run_intervention_pilot()
