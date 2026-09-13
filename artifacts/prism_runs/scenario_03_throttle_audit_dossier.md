# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_03_throttle` | **Decision Status:** `RECOMMENDED`
**Recommendation:** `cand_throttle_50` | **Safe:** `True`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `MARGINAL`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Throttle CPU workload to 50.0%
- **Primary Mechanism:** Throttling CPU compute workload directly lowers electrical power consumption, diminishing internal Joule heating generation and stabilizing core temperature.
- **Dominant Pathway:** `A_throttle -> L_cpu -> P_elec -> T_core`
- **Net Direction of Effect:** `DECREASE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `1.54 °C` (Threshold: `6.08 °C`)
- **Multivariate 8D Residual (R_8D):** `0.52 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `6.50 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Nominal in-distribution telemetry. Model predictions validated for planning.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** +9.83 °C (+13.1% headroom)
- **Safety Assessment:** Thermal has the tightest operating buffer (+9.83 °C, 13.1% headroom).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 85.17 °C | < 95.00 °C | +9.83 °C | ⚠️ MARGINAL |
| **Pressure** | `P_sys` | 3.75 bar | < 5.50 bar | +1.75 bar | ✅ PASS |
| **Flow** | `F_cool` | 45.36 L/min | > 8.00 L/min | +37.36 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 6.50 d_M | <= 15.00 d_M | +8.50 d_M | ✅ PASS |

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `0.1659`
- **Second-Best Alternative:** `cand_do_nothing`
- **Decision Margin over Second-Best:** `0.3244`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `401017950d6e978f8a784414c22c305415e03982013d7626b71ecb1f33667fc4`
- **Decision Evidence Hash:** `74527146c7eb76eae5e1c71f617f7b3d74f977a1c5c67f6eb91f55fcf2ecb09f`
- **Safety Evidence Hash:** `96066ec498a568f2d50670374e82739df0aefece896a7aa48c5badcfb034f4cf`
- **Abstention Evidence Hash:** `a7fb27135356dd6b3a3664863a94c21679aabb7efde71d0adac4a5cc7c66a849`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-13T03:34:02.014460+00:00`