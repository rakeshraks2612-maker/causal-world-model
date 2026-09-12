# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_02_valve` | **Decision Status:** `RECOMMENDED`
**Recommendation:** `cand_pump_4` | **Safe:** `False`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `UNSAFE`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Increase pump stage to 4
- **Primary Mechanism:** Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the server compute core, reducing peak core temperature.
- **Dominant Pathway:** `A_pump -> F_cool -> T_cool -> T_core`
- **Net Direction of Effect:** `DECREASE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `0.23 °C` (Threshold: `6.08 °C`)
- **Multivariate 8D Residual (R_8D):** `0.28 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `5.98 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Nominal in-distribution telemetry. Model predictions validated for planning.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** -2.16 °C (-2.9% headroom)
- **Safety Assessment:** Hard safety boundary breached: Thermal exceeds limit with dimensionless normalized violation score 0.029 (2.9% over span, raw breach: 2.16 °C).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 97.16 °C | < 95.00 °C | -2.16 °C | ❌ FAIL |
| **Pressure** | `P_sys` | 3.72 bar | < 5.50 bar | +1.78 bar | ✅ PASS |
| **Flow** | `F_cool` | 15.90 L/min | > 8.00 L/min | +7.90 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 5.98 d_M | <= 15.00 d_M | +9.02 d_M | ✅ PASS |

### Detected Safety Violations
- **`THERMAL_LIMIT_EXCEEDED`** [UNSAFE]: T_core effective (97.16°C) exceeds thermal limit (95.0°C) by 2.16°C.

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `-0.7752`
- **Second-Best Alternative:** `cand_throttle_30`
- **Decision Margin over Second-Best:** `0.3222`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `5cb725c1af3b14a399459aa437ab729fb7a82e640285df07f57ebc231494b313`
- **Decision Evidence Hash:** `da2d64c84117414fd63d9050549b00af05fdfed21f826e370fe1ad661e25d7de`
- **Safety Evidence Hash:** `c0338eea7b09e822e9c093b6a4f3b4b2ef1a1b4275d3dbc3188b852a8cfdf2e7`
- **Abstention Evidence Hash:** `e28ab65ecacf1fc163ab0e539474c750fef55bb8e325ada928ab0708c650512e`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`