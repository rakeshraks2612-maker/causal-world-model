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
- **Unified Record Hash (SHA-256):** `59266816342b0c5b36ae2488ff99d8eb02c66ff27b813fa55d82800b9d74c905`
- **Decision Evidence Hash:** `a9e82727958ff6e4fe82585c3b7229c1f485ac1915bfd33e26b6e488d084bc1c`
- **Safety Evidence Hash:** `30139cd65e44ff0a3418251e7a21bcc6033fe603640a0ee47eb8e845c0da9906`
- **Abstention Evidence Hash:** `c69566fc81b6bd37a24d659e80dbae8b7aac0214100c36002ec5ef9639e21130`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-13T03:33:59.914404+00:00`