# 🛡️ PRISM Causal Decision Explanation
**Headline:** Throttle CPU workload to 50.0%
**Trust Status:** MODEL_TRUSTED (Validated: Operating within calibrated in-distribution latent support)

## 1. Causal Mechanism & Pathway
Throttling CPU compute workload directly lowers electrical power consumption, diminishing internal Joule heating generation and stabilizing core temperature.

```text
A_throttle [DECREASE]
      ↓
L_cpu [DECREASE]
      ↓
P_elec [DECREASE]
      ↓
T_core [DECREASE (Δ = -6.70)]
```

**All Affected Descendants:** `L_cpu, P_elec, T_core`

## 2. Counterfactual Impact
Under the modeled intervention, `cand_throttle_50` is predicted to increase coolant flow by +5.67 L/min and reduce peak core temperature by 6.70°C relative to the factual baseline trajectory.
- **Peak Core Temp ($T_{\text{core}}$):** 88.56°C → 81.86°C (Net effect: -6.70°C)
- **Coolant Flow ($F_{\text{cool}}$):** 39.07 → 45.36 L/min (Net effect: +5.67 L/min)

## 3. Alternative Action Analysis
- **`cand_do_nothing`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.3244) compared to recommended action; higher operational/thermal cost.
- **`cand_throttle_20`** [LATENT_NOVELTY]: Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = 15.03 > 15.00).
- **`cand_valve_100`** [LATENT_NOVELTY]: Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = 16.03 > 15.00).
- **`cand_pump_4`** [LATENT_NOVELTY]: Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = 15.47 > 15.00).

---
**Evidence Hash:** `ce10b604f2a6befcff31c3458c434e3c12e7b10d6605842c75138c42de9dde82` | **Model:** `baseline_005`