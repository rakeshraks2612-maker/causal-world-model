# PRISM Decision Quality Benchmark (Task 5.1)

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
