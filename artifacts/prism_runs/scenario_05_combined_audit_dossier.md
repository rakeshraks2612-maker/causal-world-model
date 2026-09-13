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
- **Unified Record Hash (SHA-256):** `99ddd4a629a0645b90ed344dfc537ffca840202c867ab8c23e72adb1605548c4`
- **Decision Evidence Hash:** `f5c195d0db0a1b0ad635e7c13e67babe66a7603b52a237b80e85522608d77432`
- **Safety Evidence Hash:** `d561821b689de1ca518e4098ee7b56d40b3eea3d5b1df4bb2506971d8769dc04`
- **Abstention Evidence Hash:** `e0ece489b2c68980ad7f9f9eaf8185578bc4585ff82b12614a2d38bccb2390e4`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-13T03:05:40.629060+00:00`