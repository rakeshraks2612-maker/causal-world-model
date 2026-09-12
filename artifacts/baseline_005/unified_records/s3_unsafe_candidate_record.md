# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_03` | **Decision Status:** `NO_ACTION_REQUIRED`
**Recommendation:** `NONE (ABSTAINED)` | **Safe:** `False`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `UNSAFE`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Throttle CPU workload to 0.2%
- **Primary Mechanism:** Throttling CPU compute workload directly lowers electrical power consumption, diminishing internal Joule heating generation and stabilizing core temperature.
- **Dominant Pathway:** `A_throttle -> L_cpu -> P_elec -> T_core`
- **Net Direction of Effect:** `STABLE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `0.85 °C` (Threshold: `6.00 °C`)
- **Multivariate 8D Residual (R_8D):** `0.95 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `3.80 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Observations match representation space.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** -3.10 °C (-4.1% headroom)
- **Safety Assessment:** Hard safety boundary breached: Thermal exceeds limit with dimensionless normalized violation score 0.041 (4.1% over span, raw breach: 3.10 °C).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 98.10 °C | < 95.00 °C | -3.10 °C | ❌ FAIL |
| **Pressure** | `P_sys` | 4.50 bar | < 5.50 bar | +1.00 bar | ✅ PASS |
| **Flow** | `F_cool` | 16.00 L/min | > 8.00 L/min | +8.00 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 3.80 d_M | <= 15.00 d_M | +11.20 d_M | ✅ PASS |

### Detected Safety Violations
- **`THERMAL_LIMIT_EXCEEDED`** [UNSAFE]: T_core effective (98.10°C) exceeds thermal limit (95.0°C) by 3.10°C.

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `N/A (Abstained)`
- **Second-Best Alternative:** `None`
- **Decision Margin:** `N/A`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `563a79909ead192bd0cefd73afe822c805332dfe5d01ea3b2dd393981f59b91b`
- **Decision Evidence Hash:** `dec_hash_s3_unsafe`
- **Safety Evidence Hash:** `21c4998d3a67021323ed825e9933605b3d14fc230725a15259194b793ebbc2c6`
- **Abstention Evidence Hash:** `40b13c03423cb18d1dbf0257e1451cde53ecaa0acdaa6f0f5a15b7599ea04a11`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`