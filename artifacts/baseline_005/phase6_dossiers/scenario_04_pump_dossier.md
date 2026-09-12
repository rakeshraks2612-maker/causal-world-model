# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_04_pump` | **Decision Status:** `RECOMMENDED`
**Recommendation:** `cand_pump_3` | **Safe:** `True`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `MARGINAL`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Increase pump stage to 3
- **Primary Mechanism:** Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the server compute core, reducing peak core temperature.
- **Dominant Pathway:** `A_pump -> F_cool -> T_cool -> T_core`
- **Net Direction of Effect:** `DECREASE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `1.35 °C` (Threshold: `6.08 °C`)
- **Multivariate 8D Residual (R_8D):** `0.38 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `5.66 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Nominal in-distribution telemetry. Model predictions validated for planning.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** +5.48 °C (+7.3% headroom)
- **Safety Assessment:** Thermal has the tightest operating buffer (+5.48 °C, 7.3% headroom).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 89.52 °C | < 95.00 °C | +5.48 °C | ⚠️ MARGINAL |
| **Pressure** | `P_sys` | 4.14 bar | < 5.50 bar | +1.36 bar | ✅ PASS |
| **Flow** | `F_cool` | 44.31 L/min | > 8.00 L/min | +36.31 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 5.66 d_M | <= 15.00 d_M | +9.34 d_M | ✅ PASS |

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `0.0034`
- **Second-Best Alternative:** `cand_do_nothing`
- **Decision Margin over Second-Best:** `0.1911`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `436957cc7fbd9ede9915f609344d8f0248f758903f6891c9fa5068b975673dbb`
- **Decision Evidence Hash:** `13888997244686c603ff2e1647b49c7674c88d5bfe149f89143b4c1f70fc6ab9`
- **Safety Evidence Hash:** `d6d2f3f3a38799dfbc79a80273a4fffa3bd268087346a3775523798648e96ae9`
- **Abstention Evidence Hash:** `45927b639c1a018b73fbcb45470578b8993b9f4493184795b25dcab3bda67be4`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`