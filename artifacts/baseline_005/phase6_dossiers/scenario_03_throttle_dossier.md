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
- **Unified Record Hash (SHA-256):** `65762323a77d42fffd140c83783a2f582a482aa463f56dee5865f420248a3085`
- **Decision Evidence Hash:** `ce10b604f2a6befcff31c3458c434e3c12e7b10d6605842c75138c42de9dde82`
- **Safety Evidence Hash:** `c606b7d9c129fa1d72c4e925dc9461905b2a73a93025fff0733db752ca2ddff5`
- **Abstention Evidence Hash:** `4820f69647e69f9724dbdb4eb7f6d0e494b6b8397fc9647a40f312cbc29aba48`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`