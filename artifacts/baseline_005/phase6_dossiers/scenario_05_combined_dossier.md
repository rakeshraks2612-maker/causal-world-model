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
- **Unified Record Hash (SHA-256):** `6eeb0d61461e3fd9568776a67d0fd9a40e78bf87800f44866ffdc18fe5657184`
- **Decision Evidence Hash:** `04544fd96086975bbecaf799dde79bbc97896d81533e71884fa47d3cb1b057d1`
- **Safety Evidence Hash:** `24d54172efb21272478bdfd340cc6ba48bc379733ea2a0e1c5dc99e28082eee9`
- **Abstention Evidence Hash:** `bff31b276cdeca23994cef8023499238ebf0efa42d67922f4e03f5643af3cdd0`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`