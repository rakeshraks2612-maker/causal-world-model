# 🛡️ PRISM Causal Decision Explanation
**Headline:** Increase pump stage to 4
**Trust Status:** MODEL_TRUSTED (Validated: Operating within calibrated in-distribution latent support)

## 1. Causal Mechanism & Pathway
Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the semiconductor junction, reducing peak core temperature.

```text
A_pump [INCREASE]
      ↓
F_cool [INCREASE (Δ = +6.10)]
      ↓
T_cool [DECREASE]
      ↓
T_core [DECREASE (Δ = -9.80)]
```

**All Affected Descendants:** `F_cool, P_sys, T_cool, T_core, Vib_pump`

## 2. Counterfactual Impact
Under the modeled intervention, `cand_pump_4` is predicted to increase coolant flow by +6.10 L/min and reduce peak core temperature by 9.80°C relative to the factual baseline trajectory.
- **Peak Core Temp ($T_{\text{core}}$):** 104.15°C → 94.35°C (Net effect: -9.80°C)
- **Coolant Flow ($F_{\text{cool}}$):** 9.95 → 15.90 L/min (Net effect: +6.10 L/min)

## 3. Alternative Action Analysis
- **`cand_do_nothing`** [SAFETY_VIOLATION]: Rejected due to physical safety boundary violation: Peak core temperature (104.15°C) exceeds hard threshold (95.00°C)..
- **`cand_valve_85`** [SAFETY_VIOLATION]: Rejected due to physical safety boundary violation: Peak core temperature (105.02°C) exceeds hard threshold (95.00°C)..
- **`cand_valve_15`** [SAFETY_VIOLATION]: Rejected due to physical safety boundary violation: Peak core temperature (105.67°C) exceeds hard threshold (95.00°C)..
- **`cand_throttle_30`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.3222) compared to recommended action; higher operational/thermal cost.

---
**Evidence Hash:** `da2d64c84117414fd63d9050549b00af05fdfed21f826e370fe1ad661e25d7de` | **Model:** `baseline_005`