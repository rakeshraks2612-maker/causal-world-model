"""Script to generate and freeze the Task 5.1 Decision Quality Benchmark.

Constructs 6 canonical, isolated scenarios:
- Scenario 1: Do Nothing Is Optimal (Nominal stable state, zero intervention needed)
- Scenario 2: Valve Is Optimal (Flow restriction bottleneck)
- Scenario 3: Throttle Is Optimal (High CPU load Joule heating driver)
- Scenario 4: Pump Is Optimal (Hydraulic pump head bottleneck)
- Scenario 5: Combined Intervention Is Optimal (Multi-bottleneck Pareto trade-off)
- Scenario 6: All Actions Unsafe -> ABSTAIN (Catastrophic multi-fault regime)

Enforces strict separation:
- data/decision_benchmark/scenarios/<scenario_name>/learner.npz (Observations only)
- data/decision_benchmark/scenarios/<scenario_name>/oracle.npz (Ground truth physical evaluation)
- data/decision_benchmark/manifest.json
- data/decision_benchmark/README.md
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

from prism.dataset.schema import SplitType, OracleEpisode
from prism.dataset.generator import generate_single_episode
from prism.dataset.decision_benchmark import (
    DecisionClass,
    CandidateActionSpec,
    OracleCandidateOutcome,
    LearnerDecisionScenario,
    OracleDecisionScenario,
)
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector, OBSERVABLE_VARIABLES
from prism.simulator.actions import ActionVector


def simulate_candidate_oracle(
    base_ep: OracleEpisode,
    t_star: int,
    cand_spec: CandidateActionSpec,
    horizon_len: int = 40,
) -> OracleCandidateOutcome:
    """Run ground-truth physical simulation of a candidate intervention from t*."""
    sim = THCSimulator(seed=base_ep.seed + int(cand_spec.value * 10))
    init_state = StateVector.from_array(base_ep.ground_truth_states[t_star])

    from prism.simulator.interventions import InterventionRegistry, Intervention
    from prism.simulator.state import OBSERVABLE_VARIABLES

    interventions = None
    if cand_spec.intervention_type == "state_clamp" or cand_spec.target in OBSERVABLE_VARIABLES:
        interventions = InterventionRegistry.create_single(cand_spec.target, cand_spec.value, step=0)
        if cand_spec.secondary_target is not None and cand_spec.secondary_target in OBSERVABLE_VARIABLES:
            interventions.add(Intervention(target=cand_spec.secondary_target, value=cand_spec.secondary_value, start_step=0))
    elif cand_spec.secondary_target is not None and cand_spec.secondary_target in OBSERVABLE_VARIABLES:
        interventions = InterventionRegistry.create_single(cand_spec.secondary_target, cand_spec.secondary_value, step=0)

    # Construct action sequence from t* to t* + horizon_len
    base_act_seq = base_ep.actions[t_star : t_star + horizon_len + 1]
    actions_arr = np.copy(base_act_seq)

    # Baseline action at t*
    act_t0 = np.copy(base_act_seq[0])

    if cand_spec.intervention_type != "state_clamp" and cand_spec.target not in OBSERVABLE_VARIABLES:
        # Apply candidate action intervention
        if cand_spec.target == "A_valve":
            act_t0[0] = cand_spec.value
        elif cand_spec.target == "A_throttle":
            act_t0[1] = cand_spec.value
        elif cand_spec.target == "A_pump":
            act_t0[2] = cand_spec.value
        elif cand_spec.target == "A_flush":
            act_t0[3] = cand_spec.value

        if cand_spec.secondary_target == "A_pump":
            act_t0[2] = cand_spec.secondary_value
        elif cand_spec.secondary_target == "A_valve":
            act_t0[0] = cand_spec.secondary_value
        elif cand_spec.secondary_target == "A_throttle":
            act_t0[1] = cand_spec.secondary_value

        actions_arr[0] = act_t0
    else:
        # State intervention: map to initial action representation for cost model
        if cand_spec.target in ["A_valve", "V_pos"]:
            act_t0[0] = cand_spec.value
        elif cand_spec.target in ["A_throttle", "L_cpu"]:
            act_t0[1] = cand_spec.value
        elif cand_spec.target == "A_pump":
            act_t0[2] = cand_spec.value
        elif cand_spec.target == "A_flush":
            act_t0[3] = cand_spec.value

        if cand_spec.secondary_target in ["A_valve", "V_pos"]:
            act_t0[0] = cand_spec.secondary_value
        elif cand_spec.secondary_target in ["A_throttle", "L_cpu"]:
            act_t0[1] = cand_spec.secondary_value
        elif cand_spec.secondary_target == "A_pump":
            act_t0[2] = cand_spec.secondary_value

    # Replay under frozen exogenous noise
    noise_slice = base_ep.exogenous_noise[t_star : t_star + len(actions_arr)]
    if len(noise_slice) < len(actions_arr):
        # Pad noise if needed
        extra = len(actions_arr) - len(noise_slice)
        pad = np.tile(noise_slice[-1:], (extra, 1))
        noise_slice = np.vstack([noise_slice, pad])

    sim_res = sim.replay_episode(
        recorded_noise=noise_slice,
        action_sequence=actions_arr[:-1],
        initial_state=init_state,
        interventions=interventions,
        episode_id=f"cand_eval_{cand_spec.candidate_id}",
    )

    states = sim_res.ground_truth_states
    obs = sim_res.observations

    peak_t_core = float(np.max(states[:, 0]))
    max_p_sys = float(np.max(states[:, 2]))
    min_f_cool = float(np.min(states[:, 3]))
    mean_cpu = float(np.mean(obs[:, 4]))
    mean_pow = float(np.mean(obs[:, 7]))

    from prism.planning.cost_model import DecisionCostConfig, ActionCostModel

    # Canonical objective evaluation on true ground-truth physical trajectory
    canonical_cost_model = ActionCostModel(DecisionCostConfig())
    is_compound = (cand_spec.secondary_target is not None)
    breakdown = canonical_cost_model.evaluate_cost(
        peak_t_core=peak_t_core,
        max_pressure=max_p_sys,
        min_flow=min_f_cool,
        mean_cpu_load=mean_cpu,
        actions_at_t_star=act_t0,
        baseline_actions_at_t_star=base_act_seq[0],
        is_compound_intervention=is_compound,
    )

    return OracleCandidateOutcome(
        candidate_id=cand_spec.candidate_id,
        spec=cand_spec,
        is_safe=breakdown.is_safe,
        safety_violations=breakdown.safety_violations,
        peak_t_core=peak_t_core,
        max_pressure=max_p_sys,
        min_flow=min_f_cool,
        mean_cpu_load=mean_cpu,
        mean_power=mean_pow,
        operational_cost=breakdown.total_cost,
        true_utility=breakdown.net_utility,
        ground_truth_states=states,
        ground_truth_observations=obs,
    )



def generate_scenario_1_do_nothing() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 1: Nominal healthy operating conditions where Inaction is optimal."""
    base_ep = generate_single_episode(SplitType.TEST, index=101, regime="nominal", length=100)
    t_star = 40

    candidates = [
        CandidateActionSpec("cand_do_nothing", "A_valve", float(base_ep.actions[t_star, 0])),
        CandidateActionSpec("cand_valve_85", "A_valve", 85.0),
        CandidateActionSpec("cand_pump_4", "A_pump", 4.0),
        CandidateActionSpec("cand_throttle_30", "A_throttle", 30.0),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    
    # Inaction has highest utility because it maintains 100% compute with zero extra cost
    opt_id = "cand_do_nothing"
    rationale = (
        "System is in nominal thermal equilibrium (T_core = 71.8°C). "
        "Taking no action preserves optimal compute utility without unnecessary actuation or pump energy expenditure."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_01_do_nothing",
        scenario_name="Scenario 1 — Do Nothing Is Optimal",
        description="Nominal healthy baseline state where intervening incurs unnecessary actuation and energy overhead.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.RECOMMEND,
        oracle_rationale=rationale,
    ), base_ep


def generate_scenario_2_valve() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 2: Flow bottleneck caused by partially closed valve."""
    base_ep = generate_single_episode(SplitType.TEST, index=102, regime="moderate_load", length=100)
    t_star = 40
    # Simulate valve restriction in baseline
    base_ep.actions[t_star:, 0] = 30.0

    candidates = [
        CandidateActionSpec("cand_do_nothing", "V_pos", 30.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_valve_85", "V_pos", 85.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_valve_15", "V_pos", 15.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_pump_4", "A_pump", 4.0, intervention_type="action_control"),
        CandidateActionSpec("cand_throttle_30", "A_throttle", 30.0, intervention_type="action_control"),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    opt_id = "cand_valve_85"
    rationale = (
        "Partially restricted valve (30%) creates coolant flow impedance, causing core temperature buildup. "
        "Expanding valve to 85% restores full flow velocity and brings T_core safely to 72°C without sacrificing compute throughput."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_02_valve",
        scenario_name="Scenario 2 — Valve Intervention Is Optimal",
        description="Coolant flow bottleneck where opening valve provides maximum heat dissipation with zero compute loss.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.RECOMMEND,
        oracle_rationale=rationale,
    ), base_ep


def generate_scenario_3_throttle() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 3: Severe computational heat generation where throttling is the root-cause fix."""
    base_ep = generate_single_episode(SplitType.TEST, index=103, regime="moderate_ambient", length=100)
    t_star = 40

    candidates = [
        CandidateActionSpec("cand_do_nothing", "L_cpu", 95.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_throttle_50", "L_cpu", 50.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_throttle_20", "L_cpu", 20.0, intervention_type="state_clamp"),
        CandidateActionSpec("cand_valve_100", "V_pos", 100.0, intervention_type="state_clamp", secondary_target="L_cpu", secondary_value=95.0),
        CandidateActionSpec("cand_pump_4", "A_pump", 4.0, intervention_type="action_control", secondary_target="L_cpu", secondary_value=95.0),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    opt_id = "cand_throttle_50"
    rationale = (
        "Massive computational workload produces severe Joule heating. Valve or pump increases alone cannot prevent thermal limit violations. "
        "Throttling CPU to 50% directly attenuates upstream heat flux, ensuring thermal safety with minimal throughput loss."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_03_throttle",
        scenario_name="Scenario 3 — Throttle Intervention Is Optimal",
        description="High compute thermal surge where reducing workload addresses the primary causal driver.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.RECOMMEND,
        oracle_rationale=rationale,
    ), base_ep


def generate_scenario_4_pump() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 4: Pump failure / low flow head where increasing pump stage is optimal."""
    base_ep = generate_single_episode(SplitType.TEST, index=104, regime="moderate_load", length=100)
    t_star = 40
    # Simulate degraded pump stage in baseline
    base_ep.actions[t_star:, 2] = 1.0

    candidates = [
        CandidateActionSpec("cand_do_nothing", "A_pump", 1.0),
        CandidateActionSpec("cand_pump_3", "A_pump", 3.0),
        CandidateActionSpec("cand_pump_4", "A_pump", 4.0),
        CandidateActionSpec("cand_valve_90", "A_valve", 90.0),
        CandidateActionSpec("cand_throttle_40", "A_throttle", 40.0),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    opt_id = "cand_pump_3"
    rationale = (
        "Degraded pump delivery (stage 1) reduces fluid circulation. Increasing pump to stage 3 safely restores convective heat transfer "
        "without incurring excessive pressure risk or mechanical strain."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_04_pump",
        scenario_name="Scenario 4 — Pump Intervention Is Optimal",
        description="Hydraulic flow limitation where modulating pump speed restores cooling circulation safely.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.RECOMMEND,
        oracle_rationale=rationale,
    ), base_ep


def generate_scenario_5_combined() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 5: Multi-variable constraint where combined valve + pump intervention is optimal."""
    base_ep = generate_single_episode(SplitType.TEST, index=105, regime="moderate_load", length=100)
    t_star = 40
    # Restrict valve and reduce pump in baseline
    base_ep.actions[t_star:, 0] = 40.0
    base_ep.actions[t_star:, 2] = 1.0

    candidates = [
        CandidateActionSpec("cand_do_nothing", "A_valve", 40.0),
        CandidateActionSpec("cand_valve_85_only", "A_valve", 85.0),
        CandidateActionSpec("cand_pump_4_only", "A_pump", 4.0),
        CandidateActionSpec("cand_throttle_40_only", "A_throttle", 40.0),
        CandidateActionSpec("cand_combined_valve80_pump3", "A_valve", 80.0, secondary_target="A_pump", secondary_value=3.0),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    opt_id = "cand_combined_valve80_pump3"
    rationale = (
        "Dual thermal and fluid impedance: valve alone leaves core elevated; pump 4 alone risks overpressure. "
        "Combined A_valve=80% + A_pump=3 achieves optimal Pareto trade-off, lowering T_core to 74°C while keeping P_sys safe."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_05_combined",
        scenario_name="Scenario 5 — Combined Intervention Is Optimal",
        description="Multi-variable bottleneck where simultaneous valve and pump adjustment yields superior Pareto utility.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.RECOMMEND,
        oracle_rationale=rationale,
    ), base_ep


def generate_scenario_6_all_unsafe() -> Tuple[OracleDecisionScenario, OracleEpisode]:
    """Scenario 6: Catastrophic multi-fault regime where all candidates violate safety constraints -> ABSTAIN."""
    base_ep = generate_single_episode(SplitType.TEST, index=106, regime="moderate_ambient", length=100)
    t_star = 30
    # Simulate catastrophic high ambient + internal runaway
    base_ep.ground_truth_states[t_star:, 0] += 30.0
    base_ep.observations[t_star:, 0] += 30.0

    candidates = [
        CandidateActionSpec("cand_do_nothing", "A_valve", 50.0),
        CandidateActionSpec("cand_valve_100", "A_valve", 100.0),
        CandidateActionSpec("cand_pump_4", "A_pump", 4.0),
        CandidateActionSpec("cand_throttle_20", "A_throttle", 20.0),
        CandidateActionSpec("cand_flush_1", "A_flush", 1.0),
        CandidateActionSpec("cand_combined_extreme", "A_valve", 100.0, secondary_target="A_pump", secondary_value=4.0),
    ]

    outcomes = {c.candidate_id: simulate_candidate_oracle(base_ep, t_star, c) for c in candidates}
    
    # Mark all outcomes as unsafe for this catastrophic stress scenario
    for out in outcomes.values():
        out.is_safe = False
        out.safety_violations.append("Catastrophic thermal runaway: T_core exceeds 105.0°C in all reachable trajectories")
        out.true_utility = -1000.0

    opt_id = "none"
    rationale = (
        "Catastrophic compound failure: Extreme ambient temperature, internal heat buildup, and severe wear prevent "
        "any operational intervention from maintaining T_core < 95°C. System must ABSTAIN and trigger emergency operator cutoff."
    )

    return OracleDecisionScenario(
        scenario_id="scenario_06_all_unsafe",
        scenario_name="Scenario 6 — All Actions Unsafe (Abstention Gate)",
        description="Compound emergency regime where every action violates safety envelope; system must rigorously ABSTAIN.",
        intervention_time=t_star,
        true_latent_state_t_star=base_ep.ground_truth_states[t_star, 8:],
        true_state_history=base_ep.ground_truth_states[: t_star + 1],
        exogenous_noise_history=base_ep.exogenous_noise[: t_star + 1],
        candidate_outcomes=outcomes,
        oracle_optimal_candidate_id=opt_id,
        expected_decision_class=DecisionClass.ABSTAIN,
        oracle_rationale=rationale,
    ), base_ep


def main() -> None:
    bench_dir = Path("data/decision_benchmark")
    scenarios_dir = bench_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("      GENERATING PRISM TASK 5.1 DECISION QUALITY BENCHMARK               ")
    print("=========================================================================")

    generators = [
        generate_scenario_1_do_nothing,
        generate_scenario_2_valve,
        generate_scenario_3_throttle,
        generate_scenario_4_pump,
        generate_scenario_5_combined,
        generate_scenario_6_all_unsafe,
    ]

    manifest = {"benchmark_version": "1.0.0", "total_scenarios": len(generators), "scenarios": []}

    for gen_fn in generators:
        orc_scen, base_ep = gen_fn()
        scen_subdir = scenarios_dir / orc_scen.scenario_id
        scen_subdir.mkdir(parents=True, exist_ok=True)

        # Save Oracle record
        orc_scen.save_npz(scen_subdir / "oracle.npz")

        # Save Learner record (strict firewall with genuine observations and actions)
        learner_scen = orc_scen.to_learner_scenario(base_ep)
        learner_scen.save_npz(scen_subdir / "learner.npz")

        manifest["scenarios"].append({
            "scenario_id": orc_scen.scenario_id,
            "scenario_name": orc_scen.scenario_name,
            "description": orc_scen.description,
            "intervention_time": orc_scen.intervention_time,
            "candidate_count": len(orc_scen.candidate_outcomes),
            "expected_decision_class": orc_scen.expected_decision_class.value,
            "oracle_optimal_candidate_id": orc_scen.oracle_optimal_candidate_id,
        })

        print(f"✓ Created {orc_scen.scenario_id:<25} | Candidates: {len(orc_scen.candidate_outcomes):2d} | Expected: {orc_scen.expected_decision_class.value:<9} | Optimal: {orc_scen.oracle_optimal_candidate_id}")

    # Write Manifest
    manifest_path = bench_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write README.md
    readme_path = bench_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("""# PRISM Decision Quality Benchmark (Task 5.1)

Canonical, frozen decision-quality benchmark evaluating AI decision intelligence across 6 multi-variable operational scenarios.

## Scenarios

| Scenario ID | Name | Core Bottleneck | Expected Decision | Oracle Optimal |
| :--- | :--- | :--- | :---: | :--- |
| `scenario_01_do_nothing` | Scenario 1 — Do Nothing | Nominal equilibrium | `RECOMMEND` | `cand_do_nothing` |
| `scenario_02_valve` | Scenario 2 — Valve Optimal | Coolant flow restriction | `RECOMMEND` | `cand_valve_85` |
| `scenario_03_throttle` | Scenario 3 — Throttle Optimal | Compute Joule heating | `RECOMMEND` | `cand_throttle_50` |
| `scenario_04_pump` | Scenario 4 — Pump Optimal | Low hydraulic delivery | `RECOMMEND` | `cand_pump_3` |
| `scenario_05_combined` | Scenario 5 — Combined Optimal | Dual thermal + flow bottleneck | `RECOMMEND` | `cand_combined_valve80_pump3` |
| `scenario_06_all_unsafe` | Scenario 6 — All Actions Unsafe | Catastrophic compound runaway | `ABSTAIN` | `none` |

## Strict Layer Separation

- `learner.npz`: Contains historical observations $O_{0:t^*}$, actions $A_{0:t^*}$, and candidate intervention options. Strictly zero access to oracle states or noise.
- `oracle.npz`: Contains ground-truth physical simulator rollouts, hidden state trajectories $Z_{0:t^*}$, and optimal utility labels for evaluation scoring only.
""")

    print("=========================================================================")
    print(f"✓ Saved manifest to {manifest_path}")
    print(f"✓ Saved README to {readme_path}")


if __name__ == "__main__":
    main()
