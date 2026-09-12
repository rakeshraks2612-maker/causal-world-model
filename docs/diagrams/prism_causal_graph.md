# PRISM Causal Graph Topology

This document details the Structural Causal Model (SCM) DAG for the ThermoHydro-Compute cyber-physical cooling system.

---

## 1. ASCII Causal DAG

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

---

## 2. Mermaid Structural Causal Model (DAG)

```mermaid
graph TD
    %% Latent & Confounders
    T_amb["T_amb (Ambient Temp - Confounder)"]:::hidden
    W_wear["W_wear (Mechanical Wear)"]:::hidden
    Q_int["Q_internal (Hotspot Heat Flux)"]:::hidden
    Xi_leak["ξ_leak (Micro-Leak Rate)"]:::hidden

    %% Actions
    A_thr["A_throttle (Compute Limit)"]:::action
    A_val["A_valve (Valve Aperture)"]:::action
    A_pmp["A_pump (Pump Speed)"]:::action
    A_fls["A_flush (Emergency Flush)"]:::action

    %% Observable Physical State
    L_cpu["L_cpu (CPU Load %)"]:::obs
    P_elec["P_elec (Electrical Power kW)"]:::obs
    V_pos["V_pos (Valve Position %)"]:::obs
    F_cool["F_cool (Coolant Flow L/min)"]:::obs
    P_sys["P_sys (System Pressure bar)"]:::obs
    T_cool["T_cool (Coolant Outlet Temp °C)"]:::obs
    Vib_pmp["Vib_pump (Chassis Vibration mm/s)"]:::obs
    T_core["T_core (Core Junction Temp °C)"]:::obs

    %% Edges - Compute & Thermal
    T_amb --> L_cpu
    A_thr --> L_cpu
    L_cpu --> P_elec
    P_elec --> T_core
    Q_int --> T_core

    %% Edges - Hydraulics & Valve
    A_val --> V_pos
    W_wear --> V_pos
    V_pos --> F_cool
    V_pos --> P_sys
    V_pos --> T_cool

    %% Edges - Pump & Flush
    A_pmp --> P_sys
    A_pmp --> F_cool
    A_pmp --> Vib_pmp
    A_fls --> F_cool
    A_fls --> T_cool
    A_fls --> W_wear

    %% Edges - Leaks & Downstream Thermal
    Xi_leak --> F_cool
    Xi_leak --> P_sys
    F_cool --> T_cool
    F_cool --> T_core
    T_cool --> T_core
    P_sys --> T_core
    P_sys --> Vib_pmp
    F_cool --> Vib_pmp

    classDef obs fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef action fill:#065f46,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef hidden fill:#701a75,stroke:#f472b6,stroke-width:2px,stroke-dasharray: 5 5,color:#f8fafc;
```
