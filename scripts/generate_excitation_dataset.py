"""Generate Action-Excitation Training Dataset for Task 5.7C.

Generates and audits training data with:
- Balanced pump stages: ~25% each for stages 1, 2, 3, 4
- Short hold durations: 5-15 steps
- Zero pump/valve confounding: Corr(A_pump, A_valve) ~ 0.0
- Full physical coverage: Nominal, High Load, Thermal Runaway, Recovery, Orthogonal Steps
- Strict Learner / Oracle isolation
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from prism.dataset.excitation_generator import generate_excitation_dataset
from prism.dataset.schema import OracleEpisode, LearnerEpisode


def main() -> None:
    output_dir = Path("data/excitation_dataset")
    print("=========================================================================")
    print("      GENERATING ACTION-EXCITATION TRAINING DATASET (TASK 5.7C)          ")
    print("=========================================================================")

    # Clean old files if any
    import shutil
    for d in [output_dir / "oracle", output_dir / "learner"]:
        if d.exists():
            shutil.rmtree(d)

    stats = generate_excitation_dataset(
        output_dir=output_dir,
        train_count=120,
        val_count=25,
        length=120,
    )

    print(f"\nGeneration complete!")
    print(f"Train Episodes: {stats['train_count']} ({stats['total_train_steps']} timesteps)")
    print(f"Val Episodes:   {stats['val_count']} ({stats['total_val_steps']} timesteps)")

    # Run Comprehensive Statistical Audit on Training Set
    train_files = sorted(list((output_dir / "oracle" / "train").glob("*.npz")))
    
    all_pump = []
    all_valve = []
    all_throttle = []
    all_flush = []
    all_t_core = []
    all_f_cool = []
    all_p_sys = []
    
    hold_durations = []
    transitions_count = 0
    pump_trans_matrix = np.zeros((4, 4), dtype=int)

    for f in train_files:
        orc = OracleEpisode.load_npz(f)
        actions = orc.actions
        gt = orc.ground_truth_states
        
        pumps = actions[:, 2].astype(int)
        valves = actions[:, 0]
        throttles = actions[:, 1]
        flushes = actions[:, 3]
        
        all_pump.extend(pumps.tolist())
        all_valve.extend(valves.tolist())
        all_throttle.extend(throttles.tolist())
        all_flush.extend(flushes.tolist())
        
        all_t_core.extend(gt[:, 0].tolist())
        all_p_sys.extend(gt[:, 2].tolist())
        all_f_cool.extend(gt[:, 3].tolist())

        # Measure pump hold durations and transitions
        curr_p = pumps[0]
        curr_hold = 1
        for p in pumps[1:]:
            if p == curr_p:
                curr_hold += 1
            else:
                hold_durations.append(curr_hold)
                transitions_count += 1
                pump_trans_matrix[curr_p - 1, p - 1] += 1
                curr_p = p
                curr_hold = 1
        hold_durations.append(curr_hold)

    p_arr = np.array(all_pump)
    v_arr = np.array(all_valve)
    th_arr = np.array(all_throttle)
    fl_arr = np.array(all_flush)
    t_arr = np.array(all_t_core)
    p_sys_arr = np.array(all_p_sys)
    f_cool_arr = np.array(all_f_cool)
    holds_arr = np.array(hold_durations)

    # Calculate pump stage frequencies
    total_steps = len(p_arr)
    stage_counts = {int(s): int(np.sum(p_arr == s)) for s in [1, 2, 3, 4]}
    stage_pcts = {int(s): float(stage_counts[s] / total_steps * 100) for s in [1, 2, 3, 4]}

    # Correlation Matrix
    corr_pump_valve = float(np.corrcoef(p_arr, v_arr)[0, 1])
    corr_pump_throttle = float(np.corrcoef(p_arr, th_arr)[0, 1])
    corr_valve_throttle = float(np.corrcoef(v_arr, th_arr)[0, 1])

    # Conditional pump distributions across valve tiers
    v_low = p_arr[v_arr < 35.0]
    v_mid = p_arr[(v_arr >= 35.0) & (v_arr <= 65.0)]
    v_high = p_arr[v_arr > 65.0]

    cond_dist = {
        "valve_low (<35%)": {int(s): float(np.mean(v_low == s) * 100) if len(v_low) > 0 else 0.0 for s in [1, 2, 3, 4]},
        "valve_mid (35-65%)": {int(s): float(np.mean(v_mid == s) * 100) if len(v_mid) > 0 else 0.0 for s in [1, 2, 3, 4]},
        "valve_high (>65%)": {int(s): float(np.mean(v_high == s) * 100) if len(v_high) > 0 else 0.0 for s in [1, 2, 3, 4]},
    }

    manifest_data = {
        "dataset_name": "action_excitation_training_v1",
        "task": "Task 5.7C",
        "train_episodes": len(train_files),
        "total_train_steps": total_steps,
        "pump_stage_percentages": stage_pcts,
        "pump_stage_counts": stage_counts,
        "action_hold_steps": {
            "mean": float(np.mean(holds_arr)),
            "median": float(np.median(holds_arr)),
            "min": int(np.min(holds_arr)),
            "max": int(np.max(holds_arr)),
            "total_transitions": int(transitions_count),
            "transition_rate_pct": float(transitions_count / total_steps * 100),
        },
        "action_correlations": {
            "corr_pump_valve": corr_pump_valve,
            "corr_pump_throttle": corr_pump_throttle,
            "corr_valve_throttle": corr_valve_throttle,
        },
        "conditional_pump_distribution": cond_dist,
        "pump_transition_matrix": pump_trans_matrix.tolist(),
        "physical_summary": {
            "T_core_mean": float(np.mean(t_arr)),
            "T_core_max": float(np.max(t_arr)),
            "P_sys_mean": float(np.mean(p_sys_arr)),
            "F_cool_mean": float(np.mean(f_cool_arr)),
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n=========================================================================")
    print("           ACTION-EXCITATION TRAINING DATASET AUDIT (5.7C)               ")
    print("=========================================================================")
    print(f"Total Train Steps:          {total_steps}")
    print(f"Pump Stage 1:               {stage_pcts[1]:.2f}% ({stage_counts[1]} steps)")
    print(f"Pump Stage 2:               {stage_pcts[2]:.2f}% ({stage_counts[2]} steps)")
    print(f"Pump Stage 3:               {stage_pcts[3]:.2f}% ({stage_counts[3]} steps)")
    print(f"Pump Stage 4:               {stage_pcts[4]:.2f}% ({stage_counts[4]} steps)")
    print(f"Mean Hold Duration:         {np.mean(holds_arr):.2f} steps (Median: {np.median(holds_arr):.1f})")
    print(f"Total Transitions:          {transitions_count} ({transitions_count/total_steps*100:.2f}% of steps)")
    print(f"Corr(A_pump, A_valve):      {corr_pump_valve:+.4f} (previously +0.7770)")
    print(f"Corr(A_pump, A_throttle):   {corr_pump_throttle:+.4f}")
    print(f"Corr(A_valve, A_throttle):  {corr_valve_throttle:+.4f}")
    print("-------------------------------------------------------------------------")
    print("Conditional Pump Distribution across Valve Ranges:")
    for tier, dist in cond_dist.items():
        print(f"  {tier:20s}: P1={dist[1]:.1f}%, P2={dist[2]:.1f}%, P3={dist[3]:.1f}%, P4={dist[4]:.1f}%")
    print("-------------------------------------------------------------------------")
    print("Pump Transition Matrix (from row -> to col):")
    for r in range(4):
        print(f"  Stage {r+1} -> Stage 1-4: {pump_trans_matrix[r].tolist()}")
    print("=========================================================================")
    print(f"✓ Saved excitation dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
