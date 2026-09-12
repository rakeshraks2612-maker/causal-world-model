# 🛡️ PRISM Causal Decision Explanation
**Headline:** Increase pump stage to 4
**Trust Status:** MODEL_TRUSTED (Validated: Operating within calibrated in-distribution latent support)

## 1. Causal Mechanism & Pathway
Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the semiconductor junction, reducing peak core temperature.

```text
A_pump [INCREASE]
      ↓
F_cool [INCREASE (Δ = +8.28)]
      ↓
T_cool [DECREASE]
      ↓
T_core [DECREASE (Δ = -6.32)]
```

**All Affected Descendants:** `F_cool, P_sys, T_cool, T_core, Vib_pump`

## 2. Counterfactual Impact
Under the modeled intervention, `cand_pump_4_only` is predicted to increase coolant flow by +8.28 L/min and reduce peak core temperature by 6.32°C relative to the factual baseline trajectory.
- **Peak Core Temp ($T_{\text{core}}$):** 97.89°C → 91.58°C (Net effect: -6.32°C)
- **Coolant Flow ($F_{\text{cool}}$):** 12.54 → 19.95 L/min (Net effect: +8.28 L/min)

## 3. Alternative Action Analysis
- **`cand_do_nothing`** [SAFETY_VIOLATION]: Rejected due to physical safety boundary violation: Peak core temperature (97.89°C) exceeds hard threshold (95.00°C)..
- **`cand_valve_85_only`** [LATENT_NOVELTY]: Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = 15.62 > 15.00).
- **`cand_throttle_40_only`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.7627) compared to recommended action; higher operational/thermal cost.
- **`cand_combined_valve80_pump3`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.0071) compared to recommended action; higher operational/thermal cost.

---
**Evidence Hash:** `04544fd96086975bbecaf799dde79bbc97896d81533e71884fa47d3cb1b057d1` | **Model:** `baseline_005`