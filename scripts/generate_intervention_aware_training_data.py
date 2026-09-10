"""Generate Intervention-Aware Training Dataset (Task 3.4D-B).

Produces:
- 100 Train episodes (12,100 timesteps) across 5 regimes (Nominal, High Load, Runaway, Recovery, Step Interventions)
- 20 Validation episodes (2,420 timesteps)
- Saves strict Learner and Oracle formats to data/intervention_aware/
- Runs full census verification and isolation checks against the 220 held-out test benchmark
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from prism.dataset.intervention_aware_generator import generate_intervention_aware_dataset
from prism.dataset.schema import OracleEpisode, LearnerEpisode


def main() -> None:
    output_dir = Path("data/intervention_aware")
    print("Generating Intervention-Aware Training Dataset for Task 3.4D-B...")
    stats = generate_intervention_aware_dataset(
        output_dir=output_dir,
        train_count=100,
        val_count=20,
        length=120,
    )

    print(f"\nGeneration complete!")
    print(f"Train Episodes: {stats['train_count']} ({stats['total_train_steps']} timesteps)")
    print(f"Val Episodes:   {stats['val_count']} ({stats['total_val_steps']} timesteps)")

    # Run Census on new dataset
    train_files = sorted(list((output_dir / "oracle" / "train").glob("*.npz")))
    all_l_cpu = []
    all_t_core = []
    all_peak_t = []
    runaways = 0
    near_failures = 0
    recovery_count = 0

    for f in train_files:
        orc = OracleEpisode.load_npz(f)
        states = orc.ground_truth_states
        t_core = states[:, 0]
        l_cpu = states[:, 4]
        
        all_l_cpu.extend(l_cpu.tolist())
        all_t_core.extend(t_core.tolist())
        
        peak_t = float(np.max(t_core))
        all_peak_t.append(peak_t)
        
        if orc.failure_latched:
            runaways += 1
        elif peak_t >= 95.0:
            near_failures += 1

        if orc.oracle_metadata.get("regime") == "recovery_supervision":
            recovery_count += 1

    l_arr = np.array(all_l_cpu)
    t_arr = np.array(all_t_core)
    peak_arr = np.array(all_peak_t)

    census_summary = {
        "dataset_name": "intervention_aware_training_v2",
        "total_train_episodes": len(train_files),
        "total_train_steps": len(l_arr),
        "thermal_runaways": runaways,
        "near_failures": near_failures,
        "recovery_episodes": recovery_count,
        "l_cpu_metrics": {
            "mean": float(np.mean(l_arr)),
            "min": float(np.min(l_arr)),
            "max": float(np.max(l_arr)),
            "pct_ge_70": float(np.mean(l_arr >= 70.0) * 100),
            "pct_ge_80": float(np.mean(l_arr >= 80.0) * 100),
            "pct_ge_90": float(np.mean(l_arr >= 90.0) * 100),
        },
        "t_core_metrics": {
            "mean": float(np.mean(t_arr)),
            "min": float(np.min(t_arr)),
            "max": float(np.max(t_arr)),
            "peak_mean": float(np.mean(peak_arr)),
            "peak_max": float(np.max(peak_arr)),
            "pct_ge_95": float(np.mean(t_arr >= 95.0) * 100),
            "pct_ge_105": float(np.mean(t_arr >= 105.0) * 100),
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(census_summary, f, indent=2)

    print("\n=========================================================================")
    print("           INTERVENTION-AWARE TRAINING DATASET CENSUS")
    print("=========================================================================")
    print(f"Total Episodes:             {census_summary['total_train_episodes']}")
    print(f"Total Timesteps:            {census_summary['total_train_steps']}")
    print(f"L_cpu Mean:                 {census_summary['l_cpu_metrics']['mean']:.2f}% (Range: {census_summary['l_cpu_metrics']['min']:.1f}% - {census_summary['l_cpu_metrics']['max']:.1f}%)")
    print(f"L_cpu >= 70% Steps:         {census_summary['l_cpu_metrics']['pct_ge_70']:.2f}%")
    print(f"L_cpu >= 80% Steps:         {census_summary['l_cpu_metrics']['pct_ge_80']:.2f}%")
    print(f"L_cpu >= 90% Steps:         {census_summary['l_cpu_metrics']['pct_ge_90']:.2f}%")
    print(f"Thermal Runaway Trajectories: {census_summary['thermal_runaways']} episodes ({census_summary['thermal_runaways']/len(train_files)*100:.1f}%)")
    print(f"Near-Failure Trajectories:    {census_summary['near_failures']} episodes")
    print(f"Recovery Trajectories:        {census_summary['recovery_episodes']} episodes")
    print(f"T_core Max Peak Observed:     {census_summary['t_core_metrics']['peak_max']:.2f}°C")
    print("=========================================================================")
    print(f"✓ Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
