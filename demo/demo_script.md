# PRISM — 5-Minute Final Demo Script & Recording Blueprint

**Video Target Duration:** 5:00 minutes (300 seconds)  
**Tone:** Confident, scientific, direct, high-signal, zero fluff.  
**Core Thesis:** *"Industrial cyber-physical systems don't just need predictions. They need an AI that determines whether it knows enough to act, evaluates interventions causally, enforces hard safety constraints under uncertainty, and abstains when it cannot be trusted."*

---

## Storyboard & Timing Overview

```text
0:00 ─── 0:25   ACT 0: The Problem & Thesis ("Prediction is Not Decision")
0:25 ─── 1:35   ACT 1: S4 — PRISM Knows Enough to Act (Autonomous Decision)
1:35 ─── 2:35   ACT 2: S2 — Uncertainty-Aware Safety Catch (The "Aha" Moment)
2:35 ─── 3:35   ACT 3: S6 — Fail-Closed Abstention ("PRISM Refuses to Guess")
3:35 ─── 4:20   ACT 4: Architecture & Causal Engine Under the Hood
4:20 ─── 4:50   ACT 5: Frozen Benchmark Evidence (Quantitative Proof)
4:50 ─── 5:00   ACT 6: Closing Summary & Takeaway
```

---

## Second-by-Second Demo Script

### Act 0: The Problem & Product Definition (0:00 – 0:25)
* **Screen / Visual**: Title Slide & Concept Graphic: "Prediction $\neq$ Decision". Transition to live dashboard hero screen on `http://localhost:8501`.
* **Voiceover**:
  > *"When controlling high-consequence cyber-physical systems—like data center liquid cooling loops—traditional predictive AI is dangerous. Standard models confuse correlation with causation, output deceptively safe point predictions while ignoring uncertainty, and blindly hallucinate answers when sensors fail."*
  >
  > *"This is **PRISM**—an uncertainty-aware causal world model and decision intelligence engine that evaluates interventions, enforces hard physical safety bounds under uncertainty, and fails closed when it cannot be trusted."*

---

### Act 1: Scenario S4 — PRISM Knows Enough to Act (0:25 – 1:35)
* **Screen / Visual**: Select **Scenario 4: Pump Head Modulating (S4)** in the dashboard sidebar.
* **On-Screen Elements**:
  1. Top Hero Card: `🎯 AUTONOMOUS INTERVENTION RECOMMENDED` | `PUMP → SETTING 3` | `🛡️ MODEL TRUSTED` | `✅ PHYSICALLY SAFE`.
  2. Structural Causal DAG: `A_pump → F_cool (+10.81 L/min) → T_cool ↓ → T_core (-2.87°C)`.
  3. Thermal Forecast Chart: Temperature curve dropping comfortably below the $95^\circ\text{C}$ red safety ceiling with $+5.48^\circ\text{C}$ margin.
  4. Cryptographic Provenance panel displaying SHA-256 hash.
* **Voiceover**:
  > *"Let's start with Scenario 4: a cooling deficit is emerging in the server rack. PRISM ingests 8 channels of physical telemetry, projects the state into its 16-dimensional recurrent latent world model, and verifies model trust."*
  >
  > *"Notice the Trust Badge: `MODEL_TRUSTED`. The thermal residual is just 1.35°C, well within our calibrated 6.08°C boundary."*
  >
  > *"Rather than relying on observational correlation, PRISM performs Pearl structural interventions—$do(A)$—simulating candidate pump and valve modulations across a 10-step forward horizon. It identifies that stepping Pump Speed from 2 to 3 creates a causal flow surge of +10.81 L/min, driving core temperature down by -2.87°C."*
  >
  > *"The action is verified safe with a +5.48°C thermal margin. PRISM recommends Pump 3 and seals the entire decision stack with a deterministic SHA-256 evidence fingerprint."*

---

### Act 2: Scenario S2 — Uncertainty-Aware Safety Catch (1:35 – 2:35)
* **Screen / Visual**: Switch dropdown to **Scenario 2: Valve Intervention (S2)**.
* **On-Screen Elements**:
  1. Hero Card: `❌ CANDIDATE REJECTED — UNSAFE AFTER UNCERTAINTY ADJUSTMENT` | `REJECTED: PUMP → SETTING 4 (UNSAFE)` | `❌ PHYSICAL LIMIT BREACHED`.
  2. Thermal Trajectory Chart: Point prediction curve ($\mu = 94.35^\circ\text{C}$) in blue, but upper $2\sigma$ uncertainty band crossing the dashed red line to $T_{\text{eff}} = 97.16^\circ\text{C}$.
  3. Safety Breakdown Table: Negative margin of $-2.16^\circ\text{C}$.
* **Voiceover**:
  > *"Now, here is the critical difference between forecasting and decision intelligence. In Scenario 2, the system evaluates candidate Pump 4."*
  >
  > *"Look at the point forecast: mean peak temperature is predicted at **94.35°C**. A naive AI would say: '94.35 is less than the 95°C limit—this action is safe!'*
  >
  > *"PRISM refuses to make that gamble. Our conservative safety layer propagates predictive trajectory uncertainty using a $k=2.0$ multiplier: $T_{\text{eff}} = \mu + 2\sigma$.*
  >
  > *"With a standard deviation of 1.405°C, the upper effective boundary reaches **97.16°C**—breaching the physical safety limit by -2.16°C. PRISM flags the candidate as UNSAFE and rejects it from autonomous execution. That is uncertainty-aware safety gating in action."*

---

### Act 3: Scenario S6 — Model Distrust & Abstention (2:35 – 3:35)
* **Screen / Visual**: Switch dropdown to **Scenario 6: All-Unsafe Abstention (S6)**.
* **On-Screen Elements**:
  1. Full Red Hero Screen: `🛑 PRISM ABSTAINED — NO AUTONOMOUS ACTION RECOMMENDED`.
  2. Metrics Grid: `Observed Residual (R_T) = 33.88°C` | `Allowed Threshold (τ) = 6.0827°C` | `Excess = +27.80°C`.
  3. Decision Status: `PLANNING BLOCKED (FAIL-CLOSED)`.
  4. Counterfactual & Planning panels: `NOT EXECUTED (SUPPRESSED)`.
  5. Operator Recommendation: `Immediate manual inspection required`.
* **Voiceover**:
  > *"This is PRISM's ultimate differentiator: Scenario 6. An acute sensor malfunction and unmodeled thermal runaway occur simultaneously."*
  >
  > *"Before any candidate planner or optimization runs, PRISM's upstream trust gateway evaluates the incoming telemetry against our 98th-percentile calibrated baseline."*
  >
  > *"The reconstruction residual $R_T$ explodes to **33.88°C**, blowing past the calibrated 6.0827°C trust ceiling by +27.80°C."*
  >
  > *"Look at what happens: PRISM enters `MODEL_ABSTAIN`. Planning is strictly blocked. Counterfactuals are suppressed. No actions are fabricated. The system fails closed and alerts the human operator with exact diagnostic telemetry."*
  >
  > *"When PRISM doesn't trust its world model, it doesn't guess. It abstains."*

---

### Act 4: Architecture & The Causal Stack (3:35 – 4:20)
* **Screen / Visual**: Transition to clean system architecture slide & Pearl causal graph.
* **On-Screen Elements**:
  * Decoupled architecture flow: Ingestion $\to$ 16D RSSM GRU World Model $\to$ Upstream Trust Gate $\to$ Pearl SCM Interventions $\to$ Multi-Step Planner ($K=10$) $\to$ Conservative Safety Gate ($k=2.0\sigma$) $\to$ Unified Evidence Record.
* **Voiceover**:
  > *"Under the hood, PRISM decouples learned representation from deterministic safety."*
  >
  > *"A 16-dimensional recurrent state-space world model forecasts open-loop physical dynamics up to 40 steps ahead without autoregressive compounding. Exogenous noise abduction allows Pearl Level-3 counterfactual replay under identical historical disturbances."*
  >
  > *"Crucially, safety gating, trust thresholds, and evidence assembly are 100% deterministic—guaranteeing zero hidden LLM hallucinations or unconstrained policy drift."*

---

### Act 5: Frozen Benchmark Evidence (4:20 – 4:50)
* **Screen / Visual**: Benchmark Results Summary slide / terminal runner output (`reports/PRISM_Final_Benchmark_Report.md`).
* **On-Screen Highlights**:
  * $H=40$ World Model Normalized MAE: `0.2121` ($-56.9\%$ vs MLP baseline `0.4920`).
  * Long-Horizon Error Growth: `+24.4%` (vs MLP `+220.3%`).
  * Causal Intervention Directional Accuracy: `86.1%`.
  * Counterfactual Directional Accuracy: `77.49%` ($347/448$ test cases, $0$ false-safe errors).
  * 100% Abstention Recall on model distrust cases with 0% false trust.
  * Full Regression Suite: `420/420 tests passing`.
* **Voiceover**:
  > *"Our quantitative benchmarks prove this architecture works across the board:"*
  >
  > *"In 40-step open-loop rollouts, PRISM reduces prediction error by 56.9% compared to feed-forward MLP dynamics, maintaining a tight 24.4% error growth versus the MLP's 220% compounding drift."*
  >
  > *"Across 448 Level-3 counterfactual test cases, PRISM achieved 77.49% directional accuracy with zero false-safe classifications, while maintaining 100% abstention recall on out-of-model regimes."*

---

### Act 6: Closing Summary (4:50 – 5:00)
* **Screen / Visual**: Closing Title Slide with GitHub repository link, live demo URL, and Apache 2.0 badge.
* **Voiceover**:
  > *"PRISM doesn't just forecast the future. It evaluates interventions, tests whether predictions are trustworthy enough to act on, and produces auditable, tamper-evident decisions—or refuses to decide."*
  >
  > *"Thank you."*

---

## Pre-Recording Checklist & Verification

* [x] Dashboard running locally on `http://localhost:8501`.
* [x] S4 displays `PUMP → SETTING 3`, `MODEL_TRUSTED`, `SAFE` ($+5.48^\circ\text{C}$ margin).
* [x] S2 displays `REJECTED: PUMP → SETTING 4 (UNSAFE)`, `T_eff = 97.16°C > 95.00°C`.
* [x] S6 displays `MODEL_ABSTAIN`, `Observed R_T = 33.88°C`, `Allowed τ = 6.0827°C`, `Excess = +27.80°C`.
* [x] SHA-256 evidence fingerprints visible and formatted consistently.
* [x] All 420 automated unit and integration tests passing.
