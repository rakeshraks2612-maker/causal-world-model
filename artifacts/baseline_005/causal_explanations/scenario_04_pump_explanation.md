# 🛡️ PRISM Causal Decision Explanation
**Headline:** Increase pump stage to 3
**Trust Status:** MODEL_TRUSTED (Validated: Operating within calibrated in-distribution latent support)

## 1. Causal Mechanism & Pathway
Increasing pump stage increases coolant circulation, which accelerates convective heat extraction from the semiconductor junction, reducing peak core temperature.

```text
A_pump [INCREASE]
      ↓
F_cool [INCREASE (Δ = +10.81)]
      ↓
T_cool [DECREASE]
      ↓
T_core [DECREASE (Δ = -2.87)]
```

**All Affected Descendants:** `F_cool, P_sys, T_cool, T_core, Vib_pump`

## 2. Counterfactual Impact
Under the modeled intervention, `cand_pump_3` is predicted to increase coolant flow by +10.81 L/min and reduce peak core temperature by 2.87°C relative to the factual baseline trajectory.
- **Peak Core Temp ($T_{\text{core}}$):** 89.10°C → 86.22°C (Net effect: -2.87°C)
- **Coolant Flow ($F_{\text{cool}}$):** 35.31 → 44.31 L/min (Net effect: +10.81 L/min)

## 3. Alternative Action Analysis
- **`cand_do_nothing`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.1911) compared to recommended action; higher operational/thermal cost.
- **`cand_pump_4`** [LATENT_NOVELTY]: Rejected because the predicted trajectory entered a region outside the model's validated latent support (D_M = 15.22 > 15.00).
- **`cand_valve_85`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.2212) compared to recommended action; higher operational/thermal cost.
- **`cand_throttle_50`** [LOWER_UTILITY]: Sub-optimal utility (ΔU = -0.7965) compared to recommended action; higher operational/thermal cost.

---
**Evidence Hash:** `13888997244686c603ff2e1647b49c7674c88d5bfe149f89143b4c1f70fc6ab9` | **Model:** `baseline_005`