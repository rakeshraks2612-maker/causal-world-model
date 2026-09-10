"""Print summary tables for Task 3.3 report."""

import json

def main():
    with open("artifacts/rollout/multistep_metrics.json") as f:
        ms = json.load(f)

    print("=== TABLE A: MULTI-STEP TRAIN-NORMALIZED AGGREGATE MAE ===")
    print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format("Model", "h=1", "h=5", "h=10", "h=20", "h=40"))
    print("-" * 75)
    for m_key, m_name in [
        ("persistence_open_loop", "Persistence"),
        ("mlp_dynamics_open_loop", "MLP Dynamics"),
        ("prism_open_loop", "PRISM Open-Loop"),
        ("prism_teacher_forced", "PRISM Teacher-Forced")
    ]:
        row = [f"{ms[m_key]['by_horizon'][str(h)]['train_normalized_aggregate_mae']:.4f}" for h in [1, 5, 10, 20, 40]]
        print("{:<25} | {}".format(m_name, " | ".join(f"{v:>8}" for v in row)))

    print("\n=== PER-CHANNEL PHYSICAL MAE AT h=1, 5, 10, 20, 40 FOR PRISM OPEN-LOOP ===")
    p_ol = ms["prism_open_loop"]["by_horizon"]
    print("{:<8} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}".format(
        "Horizon", "T_core (°C)", "T_cool (°C)", "P_sys (bar)", "F_cool (L/m)", "L_cpu (%)", "V_pos (%)", "Vib (mm/s)", "P_elec (kW)"
    ))
    print("-" * 115)
    for h in [1, 5, 10, 20, 40]:
        ph = p_ol[str(h)]["per_channel"]
        print("{:<8} | {:>10.3f}   | {:>10.3f}   | {:>10.3f}   | {:>10.3f}   | {:>10.3f}   | {:>10.3f}   | {:>10.3f}   | {:>10.3f}".format(
            h, ph["T_core"]["mae"], ph["T_cool"]["mae"], ph["P_sys"]["mae"], ph["F_cool"]["mae"],
            ph["L_cpu"]["mae"], ph["V_pos"]["mae"], ph["Vib_pump"]["mae"], ph["P_elec"]["mae"]
        ))

    with open("artifacts/rollout/uncertainty_metrics.json") as f:
        unc = json.load(f)

    print("\n=== TABLE B: UNCERTAINTY CALIBRATION ACROSS HORIZONS ===")
    print("{:<8} | {:<12} | {:<14} | {:<12} | {:<14} | {:<12} | {:<14} | {:<12}".format(
        "Horizon", "T_core Cov", "T_core Width", "T_cool Cov", "T_cool Width", "P_sys Cov", "P_sys Width", "Overall Cov"
    ))
    print("-" * 115)
    for h in [1, 5, 10, 20, 40]:
        uh = unc["by_horizon"][str(h)]
        pch = uh["per_channel"]
        tcore_c = f"{pch['T_core']['empirical_coverage_90']:.1%}"
        tcore_w = f"{pch['T_core']['mean_interval_width_90']:.3f}°C"
        tcool_c = f"{pch['T_cool']['empirical_coverage_90']:.1%}"
        tcool_w = f"{pch['T_cool']['mean_interval_width_90']:.3f}°C"
        psys_c = f"{pch['P_sys']['empirical_coverage_90']:.1%}"
        psys_w = f"{pch['P_sys']['mean_interval_width_90']:.3f} bar"
        ov_c = f"{uh['overall_empirical_coverage_90']:.1%}"
        print("{:<8} | {:>12} | {:>14} | {:>12} | {:>14} | {:>12} | {:>14} | {:>12}".format(
            h, tcore_c, tcore_w, tcool_c, tcool_w, psys_c, psys_w, ov_c
        ))

    with open("artifacts/rollout/ood_metrics.json") as f:
        ood = json.load(f)

    print("\n=== TABLE C: IN-DISTRIBUTION VS OOD COMPARISON (h=40) ===")
    print("{:<30} | {:<15} | {:<15} | {:<22} | {:<15} | {:<18}".format(
        "Regime", "h=40 Norm MAE", "h=40 Raw MAE", "h=40 Uncertainty (Std)", "h=40 Coverage", "Corr(Error, Sigma)"
    ))
    print("-" * 125)
    for reg, sum_dict in ood["regimes"].items():
        print("{:<30} | {:>13.4f}   | {:>13.4f}   | {:>20.4f}   | {:>13.1%}   | {:>16.3f}".format(
            reg, sum_dict["train_normalized_mae_h40"], sum_dict["raw_mae_h40"],
            sum_dict["mean_uncertainty_std_h40"], sum_dict["empirical_coverage_90_h40"],
            sum_dict["error_uncertainty_correlation_h40"]
        ))

    with open("artifacts/rollout/action_branches.json") as f:
        ab = json.load(f)

    print("\n=== CAUSAL ACTION BRANCHING INVARIANTS ===")
    print(f"All Invariants Passed: {ab['all_invariants_passed']}")
    print(f"Valve Pre-Branch Identical: {ab['valve_experiment']['pre_branch_identical']}")
    print(f"Valve Post-Branch Divergent: {ab['valve_experiment']['post_branch_divergent']}")
    print(f"Valve Directional Checks: {ab['valve_experiment']['directional_causal_checks']}")
    print(f"Throttle Pre-Branch Identical: {ab['throttle_experiment']['pre_branch_identical']}")
    print(f"Throttle Post-Branch Divergent: {ab['throttle_experiment']['post_branch_divergent']}")
    print(f"Throttle Directional Checks: {ab['throttle_experiment']['directional_causal_checks']}")


if __name__ == "__main__":
    main()
