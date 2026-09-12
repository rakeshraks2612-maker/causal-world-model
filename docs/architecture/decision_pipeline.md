# PRISM Decision Pipeline

This document describes the decision-making and planning architecture of PRISM, detailing candidate action generation, multi-step temporal rollout, multi-objective utility optimization, and safety constraint arbitration.

---

## 1. Decision Flow Overview

The PRISM decision pipeline operates on a fixed planning cycle:

```text
                     TRUSTED CURRENT STATE (z_t, O_t)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             CANDIDATE ACTION GENERATION                │
       │                                                        │
       │  • cand_do_nothing (Baseline)                          │
       │  • Valve modulation candidates (e.g. 50%, 85%, 100%)    │
       │  • Workload throttling candidates (e.g. 20%, 50%, 80%) │
       │  • Pump staging candidates (e.g. Speed 1, 2, 3, 4)     │
       │  • Compound multi-actuator candidates                  │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             MULTI-STEP TEMPORAL ROLLOUT                │
       │             (K = 10 steps / 5.0 seconds)               │
       │                                                        │
       │  For each candidate a:                                 │
       │    1. Apply structural intervention do(A = a)          │
       │    2. Roll out N=50 Monte Carlo latent particles       │
       │    3. Decode trajectories: μ(t+1:t+K), σ(t+1:t+K)      │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │              HARD SAFETY GATING (k = 2.0)              │
       │                                                        │
       │  T_eff = max(μ_T + 2σ_T) < 95.0°C                      │
       │  P_eff = max(μ_P + 2σ_P) < 5.5 bar                     │
       │  F_eff = min(μ_F - 2σ_F) > 8.0 L/min                   │
       └────────────────────────────┬───────────────────────────┘
                                    │
                    Any Admissible Candidates?
                       │                         │
                     Yes                         No
                       │                         │
                       ▼                         ▼
       ┌────────────────────────────┐  ┌────────────────────────────┐
       │  MULTI-OBJECTIVE UTILITY   │  │    ALL CANDIDATES UNSAFE   │
       │          RANKING           │  │                            │
       │  Maximize U(a) over safe   │  │  Identify limiting failure │
       │  Select optimal candidate  │  │  Fallback or alert operator│
       └───────────────┬────────────┘  └──────────────┬─────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │  EXECUTE ACTION / EMIT PLAN │
                       └─────────────────────────────┘
```

---

## 2. Candidate Generation

The action space $\mathbf{A}_t = [A_{\text{valve}}, A_{\text{throttle}}, A_{\text{pump}}, A_{\text{flush}}]$ defines a structured set of candidate interventions:

| Candidate ID | Target Actuator | Commanded Value | Description / Operational Intent |
| :--- | :--- | :--- | :--- |
| `cand_do_nothing` | None (Hold) | Current Setpoints | Baseline holding state; zero actuator slew |
| `cand_valve_85` | $A_{\text{valve}}$ | $85.0\%$ | Expand valve aperture to increase coolant throughput |
| `cand_valve_100` | $A_{\text{valve}}$ | $100.0\%$ | Full valve opening for maximum flow expansion |
| `cand_throttle_50`| $A_{\text{throttle}}$| $50.0\%$ | Reduce CPU compute ceiling to cap electrical heat |
| `cand_throttle_20`| $A_{\text{throttle}}$| $20.0\%$ | Severe compute throttle for aggressive emergency cooling |
| `cand_pump_3` | $A_{\text{pump}}$ | Speed $3$ | Step up mechanical pump speed from Stage 2 $\to$ 3 |
| `cand_pump_4` | $A_{\text{pump}}$ | Speed $4$ | Maximum pump speed for high hydraulic head |
| `cand_combined` | $A_{\text{valve}} + A_{\text{pump}}$ | $80\% + \text{Speed } 3$ | Coordinated multi-actuator cooling response |

---

## 3. Planning Horizon & Temporal Dynamics ($K=10$)

* **Temporal Discretization**: $\Delta t = 0.5\text{ s}$ per step.
* **Planning Horizon**: $K = 10$ steps ($5.0\text{ s}$ forward window).

### Why $K=10$ is Crucial (Case Study: Pump Identifiability)
In thermal-hydraulic loops, fluid transport latency delays thermal dissipation:
* **$K=1$ step ($0.5\text{ s}$)**: Increasing pump speed increases hydraulic flow immediately, but coolant has not yet circulated through the heat exchanger. The instantaneous thermal derivative $\frac{\partial T_{\text{core}}}{\partial t} \approx 0$. A 1-step planner defaults to `cand_do_nothing` due to actuator slew costs.
* **$K=10$ steps ($5.0\text{ s}$)**: The full convective circulation loop completes. $T_{\text{core}}$ drops by $-2.87^\circ\text{C}$. The $K=10$ planner correctly identifies `cand_pump_3` as the globally optimal action with zero regret.

---

## 4. Multi-Objective Cost & Utility Model

For each candidate $a$, PRISM computes expected utility $U(a)$ as a weighted combination of thermal margin, actuator slew, and electrical cost:

$$U(a) = w_{\text{thermal}} \cdot \text{Margin}_{\text{thermal}}(a) - w_{\text{slew}} \cdot \text{Cost}_{\text{slew}}(a, A_{\text{curr}}) - w_{\text{power}} \cdot \text{Cost}_{\text{power}}(a)$$

### 4.1 Cost Components
1. **Thermal Margin ($\text{Margin}_{\text{thermal}}$)**:
   $$\text{Margin}_{\text{thermal}}(a) = 95.0^\circ\text{C} - \max_{h \in [1, K]} \left( \mu_{T_{\text{core}}}(t+h) + 2\sigma_{T_{\text{core}}}(t+h) \right)$$
2. **Actuator Slew Cost ($\text{Cost}_{\text{slew}}$)**:
   Penalizes mechanical wear and sudden valve/pump adjustments:
   $$\text{Cost}_{\text{slew}}(a, A_{\text{curr}}) = \sum_{j} c_j |a_j - A_{\text{curr}, j}|^2$$
3. **Power Consumption Cost ($\text{Cost}_{\text{power}}$)**:
   Penalizes parasitic electrical energy consumed by pump motors and auxiliary cooling:
   $$\text{Cost}_{\text{power}}(a) = \frac{1}{K}\sum_{h=1}^K P_{\text{elec}}(t+h)$$

---

## 5. Safety Constraint Arbitration & Limiting Constraints

If multiple safety boundaries are breached, PRISM evaluates the **cross-unit normalized severity** $V_i$ across constraints:

$$V_i = \frac{-\text{Margin}_i}{S_i}$$

where $S_i$ is the canonical scaling denominator ($S_{T_{\text{core}}} = 10.0^\circ\text{C}$, $S_{P_{\text{sys}}} = 1.0\text{ bar}$, $S_{F_{\text{cool}}} = 10.0\text{ L/min}$).

$$\text{Limiting Constraint} = \arg\max_{i \in \text{Breached}} V_i$$

This ensures consistent, unit-independent identification of the primary limiting safety bottleneck across temperature, pressure, and fluid flow.
