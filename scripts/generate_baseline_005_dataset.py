"""Generate and Audit Baseline 005 Unified Training Dataset (Task 5.10A).

Creates data/unified_dataset/ and prints comprehensive audit statistics:
- Proportions: 40% PID, 40% Excitation, 20% Acute Transients
- Actuator cross-correlations (valve, throttle, pump, flush)
- Pump stage distributions & transition frequencies
- Thermal regime distributions (nominal, high-safe, boundary, runaway)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import json
import numpy as np

from prism.dataset.unified_generator import generate_baseline_005_dataset
from prism.dataset.schema import OracleEpisode


def main() -> None:
    out_dir = Path("data/unified_dataset")
    print("=========================================================================")
    print("      GENERATING BASELINE 005 UNIFIED TRAINING DATASET (TASK 5.10A)      ")
    print("=========================================================================")
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    stats = generate_baseline_005_dataset(
        output_dir=out_dir,
        train_count=150,
        val_count=30,
        length=120,
    )
    
    print(f"\nGeneration complete!")
    print(f"Train Episodes: {stats['total_train_episodes']} ({stats['total_train_timesteps']} timesteps)")
    print(f"Val Episodes:   {stats['total_val_episodes']} ({stats['total_val_timesteps']} timesteps)")
    
    # Statistical Audit on Training Data
    train_files = sorted(list((out_dir / "oracle" / "train").glob("*.npz")))
    
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
        all_f_cool.extend(gt[:, 3].tolist())
        all_p_sys.extend(gt[:, 2].tolist())
        
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
    f_arr = np.array(all_f_cool)
    p_sys_arr = np.array(all_p_sys)
    holds_arr = np.array(hold_durations)
    
    print("\n" + "=" * 90)
    print("                     TRAINING SET STATISTICAL AUDIT                       ")
    print("=" * 90)
    
    # 1. Pump Stage Distribution
    print("1. Pump Stage Distribution:")
    for stage in [1, 2, 3, 4]:
        count = int(np.sum(p_arr == stage))
        pct = (count / len(p_arr)) * 100.0
        print(f"   - Pump {stage}: {count:6d} timesteps ({pct:5.2f}%)")
        
    print(f"\n   - Total Pump Transitions: {transitions_count}")
    print(f"   - Mean Pump Hold Duration: {float(np.mean(holds_arr)):.2f} steps (Median: {int(np.median(holds_arr))})")
    
    # 2. Cross-Actuator Correlation Matrix
    act_matrix = np.column_stack([v_arr, th_arr, p_arr, fl_arr])
    corr_matrix = np.corrcoef(act_matrix, rowvar=False)
    print("\n2. Actuator Cross-Correlation Matrix:")
    print("             Valve     Throttle    Pump      Flush")
    print(f"   Valve    {corr_matrix[0,0]:7.4f}   {corr_matrix[0,1]:7.4f}   {corr_matrix[0,2]:7.4f}   {corr_matrix[0,3]:7.4f}")
    print(f"   Throttle {corr_matrix[1,0]:7.4f}   {corr_matrix[1,1]:7.4f}   {corr_matrix[1,2]:7.4f}   {corr_matrix[1,3]:7.4f}")
    print(f"   Pump     {corr_matrix[2,0]:7.4f}   {corr_matrix[2,1]:7.4f}   {corr_matrix[2,2]:7.4f}   {corr_matrix[2,3]:7.4f}")
    print(f"   Flush    {corr_matrix[3,0]:7.4f}   {corr_matrix[3,1]:7.4f}   {corr_matrix[3,2]:7.4f}   {corr_matrix[3,3]:7.4f}")
    
    # 3. Thermal Regime Coverage
    nom_pct = float(np.mean(t_arr < 85.0)) * 100.0
    safe_pct = float(np.mean((t_arr >= 85.0) & (t_arr < 94.0))) * 100.0
    bound_pct = float(np.mean((t_arr >= 94.0) & (t_arr <= 104.0))) * 100.0
    runaway_pct = float(np.mean(t_arr > 104.0)) * 100.0
    
    print("\n3. Thermal Regime Distribution (T_core):")
    print(f"   - Nominal (T < 85°C):       {nom_pct:5.2f}%")
    print(f"   - High-Safe (85-94°C):      {safe_pct:5.2f}%")
    print(f"   - Boundary (94-104°C):      {bound_pct:5.2f}%")
    print(f"   - Runaway (T > 104°C):      {runaway_pct:5.2f}%")
    print(f"   - Mean T_core: {float(np.mean(t_arr)):.2f}°C | Min: {float(np.min(t_arr)):.2f}°C | Max: {float(np.max(t_arr)):.2f}°C")
    print("=" * 90)


if __name__ == "__main__":
    main()
