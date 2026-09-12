# 🛡️ PRISM Safety Audit & Constraint Evidence
**Overall Safety Posture:** `SAFE` | **Candidate Safe:** `True`
**Uncertainty Adjustment:** `Active (k=2.0)`

## 1. Physical Safety Constraints

| Constraint | Symbol | Raw (μ) | Uncertainty (σ) | Effective Value | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | 78.20 °C | 1.10 °C | 80.40 °C | < 95.00 °C | +14.60 °C | ✅ PASS |
| **Pressure** | `P_sys` | 3.40 bar | 0.12 bar | 3.64 bar | < 5.50 bar | +1.86 bar | ✅ PASS |
| **Flow** | `F_cool` | 28.50 L/min | 0.90 L/min | 26.70 L/min | > 8.00 L/min | +18.70 L/min | ✅ PASS |

## 2. Model-Support Constraint

| Constraint | Symbol | Observed Novelty | Threshold | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Latent Novelty Support** | `D_latent` | 2.40 d_M | <= 15.00 d_M | +12.60 d_M | ✅ PASS |

## 3. Limiting Constraint Analysis
- **Limiting Boundary:** `Thermal` (T_core)
- **Buffer / Headroom:** +14.60 °C (+19.5% headroom)
- **Assessment:** Thermal has the tightest operating buffer (+14.60 °C, 19.5% headroom).

---
**Safety Hash:** `0285eb06735f9daf4f78a75e0bb8a7965b89537bfa4e684b27c8414c5ef9fe96` | **Engine Version:** `v1.2_authoritative_gate`