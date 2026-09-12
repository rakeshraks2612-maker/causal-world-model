# 🏛️ PRISM Unified Decision Record & Audit Dossier
**Scenario ID:** `scenario_06_all_unsafe` | **Decision Status:** `BLOCKED`
**Recommendation:** `NONE (ABSTAINED)` | **Safe:** `False`
**Trust State:** `MODEL_ABSTAIN` | **Overall Safety State:** `ABSTAIN_REQUIRED`

---

## 1. Executive Summary & Causal Reasoning
- **Headline:** Abstain from intervention: Observation inconsistency violates model trust
- **Primary Mechanism:** The observed telemetry contradicts the learned world model state representation (reconstruction residual R_T = 33.88°C exceeded calibrated threshold of 6.08°C by +27.80°C). Autonomous simulation and intervention planning are blocked to prevent unsafe hallucinations.
- **Dominant Pathway:** `None`
- **Net Direction of Effect:** `STABLE`

## 2. Upfront Model Trust & Telemetry Integrity
- **Trust Classification:** `MODEL_ABSTAIN`
- **Thermal Reconstruction Residual (R_T):** `33.88 °C` (Threshold: `6.08 °C`)
- **Multivariate 8D Residual (R_8D):** `1.82 z-norm` (Threshold: `1.90 z-norm`)
- **Latent Manifold Novelty (D_latent):** `7.73 d_M` (Threshold: `15.00 d_M`)
- **Trust Assessment:** Reconstruction inconsistency: Telemetry contradicts internal world-model dynamics (R_T = 33.88°C > 6.1°C (R_8D = 1.82 <= 1.90)). Acute sensor/physics mismatch detected; triggering hard safety abstention.

## 3. Physical Safety & Support Constraint Audit
- **Limiting Constraint:** `Model_Trust_Gateway` (R_T)
- **Limiting Buffer / Headroom:** N/A
- **Safety Assessment:** Planning aborted due to model abstention: Model Trust Layer Aborted: Reconstruction inconsistency: Telemetry contradicts internal world-model dynamics (R_T = 33.88°C > 6.1°C (R_8D = 1.82 <= 1.90)). Acute sensor/physics mismatch detected; triggering hard safety abstention..

| Constraint | Symbol | Effective (k=2) | Limit | Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thermal** | `T_core` | N/A | < 95.00 °C | N/A | ❌ FAIL |
| **Pressure** | `P_sys` | N/A | < 5.50 bar | N/A | ❌ FAIL |
| **Flow** | `F_cool` | N/A | > 8.00 L/min | N/A | ❌ FAIL |
| **Latent_Support** | `D_latent` | 7.73 d_M | <= 15.00 d_M | +7.27 d_M | ✅ PASS |

### Detected Safety Violations
- **`ABSTAIN_REQUIRED`** [ABSTAIN_REQUIRED]: Model Trust Gateway Aborted: Model Trust Layer Aborted: Reconstruction inconsistency: Telemetry contradicts internal world-model dynamics (R_T = 33.88°C > 6.1°C (R_8D = 1.82 <= 1.90)). Acute sensor/physics mismatch detected; triggering hard safety abstention.

## 4. Abstention & Trust Boundary Audit
- **Abstention Type:** `MODEL_ABSTAIN`
- **Primary Reason:** Model Trust Gateway Triggered: Thermal reconstruction residual R_T = 33.88°C exceeds trust threshold 6.08°C by 27.80°C.
- **Blocked Actions:** `candidate_intervention_selection`, `counterfactual_recommendation_emission`
- **Safety Posture:** World model forecasts are untrusted due to high reconstruction residual. Autonomous control is withheld to prevent ungrounded actuation.
- **Recommended Operator Action:** Hold current actuator states, flag telemetry anomaly to operator, and perform human-in-the-loop diagnostic.

## 6. Decision Quality & Optimization Gap
- **Utility Score:** `N/A (Abstained)`
- **Second-Best Alternative:** `None`
- **Decision Margin:** `N/A`

---
## 7. Cryptographic Provenance Chain
- **Unified Record Hash (SHA-256):** `fcd96e34fe67b88c3949f9a6cc75edebb6db44e5f58f67efbf5d2738b2815d6d`
- **Decision Evidence Hash:** `7632e5f9b0434462d44fcfa372a93c655dfaaea756a8a066f8bbe945e65dc2b2`
- **Safety Evidence Hash:** `ea9047f3c6352d1c0c6985db0155fe0d2d7a1b469f1bed857a05f81cd08c6174`
- **Abstention Evidence Hash:** `dfd9b076916bfadcddb23683ae2c54f40220262035beb3cd544011caf77a0438`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-12T13:42:39.938827+00:00`