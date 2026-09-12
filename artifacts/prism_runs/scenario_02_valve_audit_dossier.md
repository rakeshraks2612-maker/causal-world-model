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
- **Unified Record Hash (SHA-256):** `7f54c15e81bc9a829da6c1ea9cf8513d013022cdaf5ec796a2e5fa75e75d5989`
- **Decision Evidence Hash:** `e21069e603b604bef4696ba3cc59c2b613a21ae221d329100926b89cac437039`
- **Safety Evidence Hash:** `ce351f34a351468bde881768507dceb4f3e078b52fb18c2932e5c8daea66670f`
- **Abstention Evidence Hash:** `081828327436bfc905b08234a31611cf8b6a969f55916495e47ea0635a6f9a86`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-12T13:42:39.946030+00:00`