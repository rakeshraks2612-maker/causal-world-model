# 🛡️ PRISM Safety Audit & Constraint Evidence
**Overall Safety Posture:** `UNSAFE` | **Candidate Safe:** `False`
**Uncertainty Adjustment:** `Active (k=2.0)`

## 1. Physical Safety Constraints

| Constraint | Symbol | Raw (μ) | Uncertainty (σ) | Effective Value | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 94.50 °C | 1.80 °C | 98.10 °C | < 95.00 °C | -3.10 °C | ❌ FAIL |
| **Pressure** | `P_sys` | 4.10 bar | 0.20 bar | 4.50 bar | < 5.50 bar | +1.00 bar | ✅ PASS |
| **Flow** | `F_cool` | 18.00 L/min | 1.00 L/min | 16.00 L/min | > 8.00 L/min | +8.00 L/min | ✅ PASS |

## 2. Model-Support Constraint

| Constraint | Symbol | Observed Novelty | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Latent Novelty Support** | `D_latent` | 3.80 d_M | <= 15.00 d_M | +11.20 d_M | ✅ PASS |

## 3. Limiting Constraint Analysis
- **Limiting Boundary:** `Thermal` (T_core)
- **Buffer / Headroom:** -3.10 °C (-4.1% headroom)
- **Assessment:** Hard safety boundary breached: Thermal exceeds limit with dimensionless normalized violation score 0.041 (4.1% over span, raw breach: 3.10 °C).

## 4. Detected Boundary Violations
- **`THERMAL_LIMIT_EXCEEDED`** [UNSAFE]: T_core effective (98.10°C) exceeds thermal limit (95.0°C) by 3.10°C.

---
**Safety Hash:** `fc14fd23e32497d7e3745ebd5c4f8e016047d205fef21473178c25294fa40469` | **Engine Version:** `v1.2_authoritative_gate`