# PRISM Uncertainty & Trust Architecture

This document describes PRISM's uncertainty quantification framework, the mathematical separation between predictive uncertainty and model trust, and the fail-closed abstention mechanisms that prevent unsafe autonomous actuation.

---

## 1. The Core Scientific Distinction: Prediction $\neq$ Trust

A fundamental contribution of PRISM is the strict separation between **predictive uncertainty** and **model trust**:

```text
Predictive uncertainty (σ)
        │
        └── "How much variance is expected in the forward trajectory?"
            (Aleatoric / particle spread in known dynamics)

Latent support / novelty (D_latent)
        │
        └── "Has the world model ever seen this latent state regime?"
            (Mahalanobis distance on the latent manifold)

Reconstruction residual (R_T, R_8D)
        │
        └── "Does the current physical telemetry contradict the world model?"
            (Immediate sensor fault, unmodeled physics, or acute runaway)

Trust Gateway (Decision)
        │
        └── "Should PRISM rely on this model to recommend physical actions?"

Abstention (Fail-Closed)
        │
        └── "If trust is insufficient, block planning and refuse to act."
```

> [!IMPORTANT]
> **Key Benchmark Finding**: In empirical out-of-distribution (OOD) experiments, predictive observation variance ($\sigma$) remained deceptively narrow ($0.34$ to $0.47$) even when forecast errors surged by $10\times$. Therefore, **predictive uncertainty $\sigma$ is NEVER treated as a proxy for epistemic certainty**.

---

## 2. Mathematical Formalisms

### 2.1 Predictive Uncertainty Quantification
* **Decoder Output**: For each channel $i$, the decoder outputs both a predictive mean $\hat{y}_i$ and an observation log-variance $s_i$:
  $$\mu_i(z_t) = g_{\phi, i}^{\mu}(z_t), \quad \sigma_i(z_t) = \text{softplus}(g_{\phi, i}^{\sigma}(z_t))$$
* **Monte Carlo Particle Rollouts ($N=50$)**:
  During multi-step rollout ($H=1 \dots K$), PRISM samples $N=50$ stochastic latent particles:
  $$z_{t+h}^{(n)} \sim p(z_{t+h} \mid z_{t+h-1}^{(n)}, A_{t+h-1}), \quad n = 1, \dots, N$$
  The total predictive trajectory variance for channel $i$ combines particle spread and observation noise:
  $$\sigma_{\text{total}, i}^2(t+h) = \frac{1}{N}\sum_{n=1}^N \left(\mu_i(z_{t+h}^{(n)}) - \bar{\mu}_i(t+h)\right)^2 + \frac{1}{N}\sum_{n=1}^N \sigma_i^2(z_{t+h}^{(n)})$$

---

### 2.2 Latent Support Manifold Distance ($D_{\text{latent}}$)
To quantify whether a state $z_t$ falls inside the support of the training data distribution, PRISM computes the Mahalanobis distance with respect to the empirical latent mean $\mu_z \in \mathbb{R}^{16}$ and covariance $\Sigma_z \in \mathbb{R}^{16 \times 16}$:

$$D_{\text{latent}}(z_t) = \sqrt{(z_t - \mu_z)^T \Sigma_z^{-1} (z_t - \mu_z)}$$

* **Calibrated Threshold**: $\tau_{D_{\text{latent}}} = 15.00\text{ d}_M$
* **Interpretation**: If $D_{\text{latent}} > 15.00$, the system is in an unmodeled region of the latent state space where neural dynamics are non-generalizable.

---

### 2.3 Dynamic Telemetry Reconstruction Residuals ($R_T, R_{8D}$)
When acute sensor faults, physical component breaks, or severe thermal runaways occur, the observed observation $O_t$ diverges sharply from the world model's reconstruction $\hat{O}_t = g(\text{enc}(O_{1:t}))$.

1. **Thermal Residual ($R_T$)**:
   $$R_T = |O_{t, T_{\text{core}}} - \hat{O}_{t, T_{\text{core}}}|$$
   $$\text{Threshold: } \tau_{R_T} = 6.0827^\circ\text{C} \quad (\text{98th percentile calibrated on 3,872 validation samples})$$

2. **Full 8D Normalized Residual ($R_{8D}$)**:
   $$R_{8D} = \|\tilde{O}_t - \hat{\tilde{O}}_t\|_2$$
   $$\text{Threshold: } \tau_{R_{8D}} = 1.8960 \quad (\text{98th percentile calibrated on 3,872 validation samples})$$

---

## 3. Dual-Gate Trust Decision Logic

Before candidate planning commences, the upstream trust gateway evaluates all three indicators:

```text
               CURRENT TELEMETRY (O_t) & LATENT STATE (z_t)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                 TRUST GATE EVALUATION                  │
       │                                                        │
       │   Condition 1: R_T ≤ 6.0827°C                          │
       │   Condition 2: R_8D ≤ 1.8960                           │
       │   Condition 3: D_latent ≤ 15.00 d_M                    │
       └────────────────────────────┬───────────────────────────┘
                                    │
                    All True? ──────┴────── Any False?
                       │                         │
                       ▼                         ▼
            ┌─────────────────────┐   ┌─────────────────────┐
            │    MODEL_TRUSTED    │   │    MODEL_ABSTAIN    │
            │  Proceed to Planner │   │     FAIL-CLOSED     │
            │  Evaluate Candidates│   │   Planning Blocked  │
            │  Recommend Action   │   │ Recommendation=None │
            └─────────────────────┘   └─────────────────────┘
```

---

## 4. Uncertainty-Aware Safety Constraints

During candidate trajectory rollouts, PRISM enforces conservative safety boundaries using an uncertainty multiplier $k = 2.0$:

### 4.1 Thermal Limit Evaluation
$$\text{Effective Peak Temperature: } T_{\text{eff}} = \max_{h \in [1, K]} \left[ \mu_{T_{\text{core}}}(t+h) + k \cdot \sigma_{T_{\text{core}}}(t+h) \right]$$
$$\text{Safety Constraint: } T_{\text{eff}} < 95.0^\circ\text{C}$$
$$\text{Thermal Margin: } \text{Margin}_{\text{thermal}} = 95.0^\circ\text{C} - T_{\text{eff}}$$

* If $\text{Margin}_{\text{thermal}} < 0$, the action is classified as **UNSAFE** and rejected.

### 4.2 Pressure & Flow Constraints
$$\text{Effective Max Pressure: } P_{\text{eff}} = \max_{h \in [1, K]} \left[ \mu_{P_{\text{sys}}}(t+h) + k \cdot \sigma_{P_{\text{sys}}}(t+h) \right] < 5.5\text{ bar}$$
$$\text{Effective Min Flow: } F_{\text{eff}} = \min_{h \in [1, K]} \left[ \mu_{F_{\text{cool}}}(t+h) - k \cdot \sigma_{F_{\text{cool}}}(t+h) \right] > 8.0\text{ L/min}$$

---

## 5. Fail-Closed Abstention Semantics

When `MODEL_ABSTAIN` triggers (as demonstrated in Scenario S6):
1. **Planning Suppression**: No optimization or candidate search is executed.
2. **Counterfactual Suppression**: Retrospective twin-world simulation is bypassed to prevent fictitious projections.
3. **No Hallucinated Actions**: The decision engine returns `None (BLOCKED)`.
4. **Diagnostic Evidence Emission**: An `AbstentionExplanation` record is generated, detailing the exact excess threshold (e.g. $R_T = 33.88^\circ\text{C} > 6.0827^\circ\text{C}$, Excess $+27.80^\circ\text{C}$), blocked actuators, and the immediate requirement for manual operator inspection.
