# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_01` | **Decision Status:** `RECOMMENDED`
**Recommendation:** `cand_valve_open_1.0` | **Safe:** `True`
**Trust State:** `MODEL_TRUSTED` | **Overall Safety State:** `SAFE`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Adjust valve position to 1.0%
- **Primary Mechanism:** Opening the coolant valve expands hydraulic conductance, elevating flow rate and convective cooling to dissipate heat from the core.
- **Dominant Pathway:** `A_valve -> V_pos -> F_cool -> P_sys -> T_core`
- **Net Direction of Effect:** `STABLE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_TRUSTED`
- **Thermal Reconstruction Residual (R_T):** `0.45 °C` (Threshold: `6.00 °C`)
- **Multivariate 8D Residual (R_8D):** `0.82 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `2.10 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Observations match representation space.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Thermal` (T_core)
- **Limiting Buffer / Headroom:** +14.10 °C (+18.8% headroom)
- **Safety Assessment:** Thermal has the tightest operating buffer (+14.10 °C, 18.8% headroom).

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 80.90 °C | < 95.00 °C | +14.10 °C | ✅ PASS |
| **Pressure** | `P_sys` | 3.80 bar | < 5.50 bar | +1.70 bar | ✅ PASS |
| **Flow** | `F_cool` | 24.00 L/min | > 8.00 L/min | +16.00 L/min | ✅ PASS |
| **Latent_Support** | `D_latent` | 2.10 d_M | <= 15.00 d_M | +12.90 d_M | ✅ PASS |

## 5. Counterfactual Twin-World Verification
- **Intervention Target:** `A_valve` = `1.0` (Factual: `0.5`)
- **Peak T_core Delta:** `-13.50 °C`
- **Max Pressure Delta:** `-0.20 bar`
- **Min Flow Delta:** `+8.00 L/min`
- **Exogenous Conditions Shared:** `True`
- **Pre-Intervention History Identical:** `True`
- **Counterfactual Safety Transition:** `MARGINAL -> SAFE`

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `0.9250`
- **Second-Best Alternative:** `cand_throttle_close_0.5`
- **Decision Margin over Second-Best:** `0.0750`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `82aa85d229813a7359fbbec966eb7bea4edc3e8dbc26d29a699c2c0551acb7de`
- **Decision Evidence Hash:** `dec_hash_s1_safe`
- **Safety Evidence Hash:** `4e3ff314515851ee34414e9419818af790a7853988ba96105ec1ef9f287af1e6`
- **Abstention Evidence Hash:** `f8a7efc67b838dfe7301029b80e1144c9a5bf9ff80b0963d400872762811f2dd`
- **Counterfactual Hash:** `cf_hash_s1_001`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-11T12:00:00Z`