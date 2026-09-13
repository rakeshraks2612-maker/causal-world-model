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
- **Unified Record Hash (SHA-256):** `9a8b5c00750e8cab8821c64a1941b03d72737a17cefe4509892bb1d4e56aa723`
- **Decision Evidence Hash:** `86932508314f41fe618233ee2f706be74256f1cd9bd79195bc7dd03063e0a2b6`
- **Safety Evidence Hash:** `8be377da1054a4cbc9f0eef5b806a8eca9a894624b264ce69dc37ee147d07345`
- **Abstention Evidence Hash:** `23a377a58f2e65d40d0d5fea754ae48a5f4489e673d2862783aef4478fe9e96c`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-13T04:38:41.967581+00:00`