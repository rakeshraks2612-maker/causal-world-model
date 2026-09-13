# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_05_combined` | **Decision Status:** `RECOMMENDED`
**Recommendation:** `cand_pump_4_only` | **Safe:** `True`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `MARGINAL`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Increase pump stage to 4
- **Primary Mechanism:** Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the server compute core, reducing peak core temperature.
- **Dominant Pathway:** `A_pump -> F_cool -> T_cool -> T_core`
- **Net Direction of Effect:** `DECREASE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `0.64 °C` (Threshold: `6.08 °C`)
- **Multivariate 8D Residual (R_8D):** `0.22 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `5.86 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Nominal in-distribution telemetry. Model predictions validated for planning.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** +0.57 °C (+0.8% headroom)
- **Safety Assessment:** Thermal has the tightest operating buffer (+0.57 °C, 0.8% headroom).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 94.43 °C | < 95.00 °C | +0.57 °C | ⚠️ MARGINAL |
| **Pressure** | `P_sys` | 3.65 bar | < 5.50 bar | +1.85 bar | ✅ PASS |
| **Flow** | `F_cool` | 19.95 L/min | > 8.00 L/min | +11.95 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 5.86 d_M | <= 15.00 d_M | +9.14 d_M | ✅ PASS |

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `-0.4950`
- **Second-Best Alternative:** `cand_combined_valve80_pump3`
- **Decision Margin over Second-Best:** `0.0071`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `c8660c8aa6ba73f50ca77c8f1cada8f9ab4122f5145a0d11667ed55c4098d664`
- **Decision Evidence Hash:** `685eec70df29ab073c0908ac6e2d3e331cb94858155950e0fb64fa1c1b706847`
- **Safety Evidence Hash:** `56f1acbab26e4d05a78e4dd044916c815aefe53ef2a584c7f9995d97f50040e6`
- **Abstention Evidence Hash:** `64d496aa87ebbb96b82f85bb715fef6b2b5efc6248cd536187f2fb2e24add1fe`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-13T03:59:32.792343+00:00`