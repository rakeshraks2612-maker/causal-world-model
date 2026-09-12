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
- **Unified Record Hash (SHA-256):** `478264c7abd343aaaa1d6955388fa75bf98f5432f2d62dcdb66f2d8c08b558b5`
- **Decision Evidence Hash:** `2c161f510fab2f3a78955929fa580cbff705b0372efef55e7c6f06d5cf8ae4c0`
- **Safety Evidence Hash:** `f06e5d6f5503168190c573b78232a5703294f6ebc787b35443356d6b5484f9f7`
- **Abstention Evidence Hash:** `37cc144abd6b57239b7851c29621e8b4fd7bd9f66befb3529aa0d88cf58180bc`
- **Counterfactual Hash:** `N/A`
- **Model / Planner:** `baseline_005` / `v1.2` | **Timestamp (UTC):** `2026-09-12T13:38:08.451073+00:00`