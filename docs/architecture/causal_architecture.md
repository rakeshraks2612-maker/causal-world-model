# PRISM Causal Architecture

This document describes the causal foundation of PRISM, detailing the Structural Causal Model (SCM), Pearl's intervention semantics ($do(A)$), graph surgery, and the critical distinction between action interventions and state interventions.

---

## 1. Structural Causal Model (ThermoHydro-Compute SCM)

The industrial environment modeled by PRISM represents a high-density compute server rack coupled with a closed-loop liquid-to-air cooling system.

### 1.1 State Variables ($\mathbf{X}_t \in \mathbb{R}^{12}$)
The physical state vector $\mathbf{X}_t = [\mathbf{Y}_t^T, \mathbf{Z}_t^T]^T$ is partitioned into 8 observable state variables $\mathbf{Y}_t$ and 4 hidden/confounding variables $\mathbf{Z}_t$:

| Variable | Symbol | Nature | Unit | Physical Range | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Core Temperature** | $T_{\text{core}}$ | **Observable** | $^\circ\text{C}$ | $[20.0, 125.0]$ | Semiconductor junction temperature |
| **Coolant Outlet Temp** | $T_{\text{cool}}$ | **Observable** | $^\circ\text{C}$ | $[15.0, 95.0]$ | Coolant exiting heat sink block |
| **Loop Pressure** | $P_{\text{sys}}$ | **Observable** | $\text{bar}$ | $[0.5, 6.0]$ | Total hydraulic system pressure |
| **Coolant Flow Rate** | $F_{\text{cool}}$ | **Observable** | $\text{L/min}$ | $[0.0, 60.0]$ | Volumetric fluid displacement |
| **Compute Workload** | $L_{\text{cpu}}$ | **Observable** | $\%$ | $[0.0, 100.0]$ | Normalized CPU load |
| **Coolant Valve Position** | $V_{\text{pos}}$ | **Observable** | $\%$ | $[0.0, 100.0]$ | Motorized valve aperture opening |
| **Pump Vibration** | $\text{Vib}_{\text{pump}}$ | **Observable** | $\text{mm/s}$ | $[0.0, 25.0]$ | Chassis acoustic vibration |
| **Electrical Power** | $P_{\text{elec}}$ | **Observable** | $\text{kW}$ | $[0.2, 5.0]$ | Rack electrical power consumption |
| **Ambient Temperature** | $T_{\text{amb}}$ | **Hidden Confounder** | $^\circ\text{C}$ | $[10.0, 45.0]$ | Facility ambient temperature |
| **Mechanical Wear** | $W_{\text{wear}}$ | **Hidden State** | $[0, 1]$ | $[0.0, 1.0]$ | Cumulative fouling & actuator friction |
| **Hotspot Heat Flux** | $Q_{\text{internal}}$ | **Hidden State** | $\text{kW}$ | $[-0.5, 2.5]$ | Silicon micro-architectural hotspot flux |
| **Coolant Micro-Leak** | $\xi_{\text{leak}}$ | **Hidden State** | $\text{mL/hr}$ | $[0.0, 50.0]$ | Seal degradation causing pressure drop |

---

## 2. Structural Causal Graph

The causal relationships and dependencies are governed by the following DAG:

```text
T_amb ───────► L_cpu ─────► P_elec ─────► T_core
  │
  └──────────────────────────────────────► cooling context

A_throttle ─► L_cpu

A_valve ────► V_pos ──────► F_cool ─────► T_core
                 ▲             │
                 │             ├────────► P_sys
                 │             └────────► T_cool
                 │
W_wear ──────────┘

A_pump ─────────► P_sys
      └─────────► F_cool

A_flush ────────► F_cool
      └─────────► T_cool / W_wear

Q_internal ─────────────────────────────► T_core

xi_leak ───────► F_cool / P_sys

P_sys ─────────► T_core
P_sys/F_cool/A_pump ─► Vib_pump
```

### Primary Structural Equations
1. **Compute & Thermal Load**:
   $$L_{\text{cpu}, t} = f_L(T_{\text{amb}, t}, A_{\text{throttle}, t}, U_L)$$
   $$P_{\text{elec}, t} = f_P(L_{\text{cpu}, t}, U_P)$$
   $$T_{\text{core}, t+1} = T_{\text{core}, t} + \alpha_1 P_{\text{elec}, t} + Q_{\text{internal}, t} - \beta_1 F_{\text{cool}, t} (T_{\text{core}, t} - T_{\text{cool}, t}) + U_T$$
2. **Hydraulic Dynamics**:
   $$V_{\text{pos}, t} = f_V(A_{\text{valve}, t}, W_{\text{wear}, t}, U_V)$$
   $$F_{\text{cool}, t} = f_F(A_{\text{pump}, t}, V_{\text{pos}, t}, \xi_{\text{leak}, t}, U_F)$$
   $$P_{\text{sys}, t} = f_P(A_{\text{pump}, t}, V_{\text{pos}, t}, \xi_{\text{leak}, t}, U_P)$$
3. **Vibration & Acoustics**:
   $$\text{Vib}_{\text{pump}, t} = f_{\text{vib}}(A_{\text{pump}, t}, P_{\text{sys}, t}, W_{\text{wear}, t}, U_{\text{vib}})$$

---

## 3. Intervention Semantics ($do$-Calculus)

PRISM implements structural interventions rather than observational conditioning:

### Observational Conditioning vs. Structural Intervention
* **Observational Conditioning $P(Y \mid X = x)$**:
  Computes the distribution of $Y$ when $X$ happens to be observed as $x$. This conflates the causal effect of $X$ on $Y$ with confounding from common causes ($T_{\text{amb}}$).
* **Structural Intervention $P(Y \mid do(X = x))$**:
  Physically sets $X \leftarrow x$ via graph surgery, replacing the structural equation $X = f_X(\text{Parents}(X), U_X)$ with a constant setpoint $x$ and deleting all incoming causal edges to $X$.

### Graph Surgery Mechanism
When evaluating $do(A_t = a)$:
1. Invert the structural equation for $A_t$.
2. Cut all incoming edges: $\text{Parents}(A_t) \to A_t$.
3. Keep all downstream structural mechanisms intact:
   $$A_t \to \text{Children}(A_t) \to \dots \to T_{\text{core}}$$

---

## 4. Action Interventions vs. State Interventions

PRISM explicitly differentiates between **Action Interventions** and **State Interventions**:

```text
Action Intervention: do(A_pump = 3)
┌──────────┐      ┌─────────────┐      ┌────────────┐      ┌────────────┐
│ A_pump=3 │ ───► │ F_cool ↑    │ ───► │ T_cool ↓   │ ───► │ T_core ↓   │
└──────────┘      │ (+10.81 L)  │      │ (-0.85°C)  │      │ (-2.87°C)  │
                  └─────────────┘      └────────────┘      └────────────┘

State Intervention: do(V_pos = 85%)
                  ┌─────────────┐      ┌────────────┐      ┌────────────┐
[Cut: A_valve]    │ V_pos = 85% │ ───► │ F_cool ↑   │ ───► │ T_core ↓   │
[Cut: W_wear ] ─X │ (Direct set)│      │            │      │            │
                  └─────────────┘      └────────────┘      └────────────┘
```

1. **Action Intervention $do(A_{\text{target}} = a)$**:
   * Simulates an operator actuating a physical control knob ($A_{\text{pump}}, A_{\text{valve}}, A_{\text{throttle}}, A_{\text{flush}}$).
   * Actuator dynamics, wear, and physical response times are naturally modeled as intermediate causal steps.
2. **State Intervention $do(X_{\text{state}} = x)$**:
   * Directly clamps a physical state variable ($V_{\text{pos}} \leftarrow 85\%$).
   * Cuts incoming structural links (e.g. actuator motor lag and wear friction) while evaluating pure downstream hydraulic and thermal propagation.

---

## 5. Major Causal Propagation Pathways

PRISM's explanation layer decomposes multi-variable effects into four canonical pathways:

1. **Cooling Circulation Mechanism**:
   $$A_{\text{pump}} \uparrow \implies F_{\text{cool}} \uparrow \implies T_{\text{cool}} \downarrow \implies T_{\text{core}} \downarrow$$
2. **Hydraulic Restriction Mechanism**:
   $$A_{\text{valve}} \uparrow \implies V_{\text{pos}} \uparrow \implies F_{\text{cool}} \uparrow, \, P_{\text{sys}} \downarrow \implies T_{\text{core}} \downarrow$$
3. **Heat Generation Throttling Mechanism**:
   $$A_{\text{throttle}} \downarrow \implies L_{\text{cpu}} \downarrow \implies P_{\text{elec}} \downarrow \implies T_{\text{core}} \downarrow$$
4. **Emergency Line Purge Mechanism**:
   $$A_{\text{flush}} = 1 \implies F_{\text{cool}} \uparrow \uparrow, \, T_{\text{cool}} \downarrow \downarrow, \, W_{\text{wear}} \downarrow \implies T_{\text{core}} \downarrow \downarrow$$
