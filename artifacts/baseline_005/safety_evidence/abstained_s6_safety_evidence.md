# 🛡️ PRISM Safety Audit & Constraint Evidence
**Overall Safety Posture:** `ABSTAIN_REQUIRED` | **Candidate Safe:** `False`
**Uncertainty Adjustment:** `Active (k=2.0)`

## 1. Physical Safety Constraints

| Constraint | Symbol | Raw (μ) | Uncertainty (σ) | Effective Value | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | N/A | N/A | N/A | < 95.00 °C | N/A | ❌ FAIL |
| **Pressure** | `P_sys` | N/A | N/A | N/A | < 5.50 bar | N/A | ❌ FAIL |
| **Flow** | `F_cool` | N/A | N/A | N/A | > 8.00 L/min | N/A | ❌ FAIL |

## 2. Model-Support Constraint

| Constraint | Symbol | Observed Novelty | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Latent Novelty Support** | `D_latent` | N/A | <= 15.00 d_M | N/A | ❌ FAIL |

## 3. Limiting Constraint Analysis
- **Limiting Boundary:** `Model_Trust_Gateway` (R_T)
- **Buffer / Headroom:** N/A
- **Assessment:** Planning aborted due to model abstention: Thermal reconstruction residual R_T = 33.88°C exceeded threshold 6.00°C.

## 4. Detected Boundary Violations
- **`ABSTAIN_REQUIRED`** [ABSTAIN_REQUIRED]: Model Trust Gateway Aborted: Thermal reconstruction residual R_T = 33.88°C exceeded threshold 6.00°C

---
**Safety Hash:** `fbab4777e12bf08df82464ef8ca79c9c08dcd28cbc074d26aa38f4e7b3c1e4bf` | **Engine Version:** `v1.2_authoritative_gate`