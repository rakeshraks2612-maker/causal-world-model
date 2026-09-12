# PRISM System Architecture

This document describes the end-to-end system architecture of PRISM, detailing subsystem interactions, data representations, module boundaries, and the strict separation between learned and deterministic layers.

---

## 1. System Pipeline Overview

PRISM processes cyber-physical telemetry through an integrated, multi-stage pipeline designed for safety, causal fidelity, and cryptographic auditability:

```text
                    ┌──────────────────────┐
                    │   Industrial World   │
                    │  Telemetry / State   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Telemetry Ingestion  │
                    │ Normalization (O_t)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  PRISM World Model   │
                    │                      │
                    │ Encoder → Latent z   │
                    │ Transition Model     │
                    │ Decoder (μ, σ)       │
                    └───────┬───────┬──────┘
                            │       │
                prediction  │       │ latent support
                            │       │ residuals
                            ▼       ▼
                 ┌────────────┐  ┌───────────────┐
                 │ Forecasts  │  │ Trust Gateway │
                 │ + σ        │  │ Novelty       │
                 └─────┬──────┘  │ Residual      │
                       │         │ Uncertainty   │
                       │         └───────┬───────┘
                       │                 │
                       ▼                 ▼
                 ┌────────────────────────────┐
                 │ Causal Intervention Layer  │
                 │ Pearl graph surgery        │
                 │ Intervention semantics     │
                 └──────────────┬─────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Decision Planner     │
                    │ Candidate generation │
                    │ Multi-step rollout   │
                    │ Multi-obj utility    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Level-3 Counterfactual│
                    │ Abduction             │
                    │ Intervention          │
                    │ Twin-world replay     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Safety Evaluation    │
                    │ Tcore / Psys / Fcool │
                    │ Latent support       │
                    │ Uncertainty adjusted │
                    └──────────┬───────────┘
                               │
                         ┌─────┴─────┐
                         ▼           ▼
                    RECOMMEND     ABSTAIN
                         │           │
                         └─────┬─────┘
                               ▼
                    ┌──────────────────────┐
                    │ Unified Evidence     │
                    │ Causal + CF + Safety │
                    │ Trust + Provenance   │
                    │ SHA-256 fingerprint  │
                    └──────────────────────┘
```

---

## 2. Learned vs. Deterministic Components

A core design principle of PRISM is that **safety, trust gating, and evidence generation are strictly deterministic**, while **latent representation and dynamic state transitions are learned**.

| Subsystem | Execution Nature | Mechanism | Description |
| :--- | :--- | :--- | :--- |
| **Observation Normalization** | Deterministic | Z-score scaling / bounds | Scaled against training statistics |
| **Latent Encoder $q(z_t \mid O_{1:t}, A_{1:t})$** | **Learned** | MLP with residual skips | Projects 8D observation history to 16D latent manifold |
| **Latent Dynamics $p(z_{t+1} \mid z_t, A_t)$** | **Learned** | GRU with RSSM transition | Predicts forward latent state evolution |
| **Observation Decoder $p(O_t \mid z_t)$** | **Learned** | Mean $\mu_t$, Heteroscedastic $\sigma_t$ | Reconstructs 8-channel physical telemetry |
| **Exogenous Noise Abduction** | Deterministic | Inversion $U_t = O_t - \mu(z_t)$ | Captures historical environmental disturbances |
| **Trust Gateway** | **Deterministic** | Hard boundary checks | Gated against calibrated $\tau_{R_T}, \tau_{R_{8D}}, \tau_{D_{\text{latent}}}$ |
| **Causal Graph Surgery** | **Deterministic** | Structural graph surgery | Implements Pearl's $do(A)$ graph modifications |
| **Decision Planner** | **Deterministic** | Multi-objective optimization | Evaluates action candidates across $K=10$ horizons |
| **Safety Gate** | **Deterministic** | Conservative interval ($k=2.0$) | Evaluates $T_{\text{eff}} < 95^\circ\text{C}, P_{\text{eff}} < 5.5\text{ bar}, F_{\text{eff}} > 8\text{ L/min}$ |
| **Evidence Assembler** | **Deterministic** | Pure immutable projection | Assembles 8-domain `PrismDecisionRecord` |
| **Cryptographic Provenance** | **Deterministic** | Canonical JSON + SHA-256 | Computes immutable 256-bit tamper-evident hash |

---

## 3. Subsystem Breakdown

### 3.1 Observation Ingestion & Normalization
* **Input**: 8-channel physical observation vector $O_t \in \mathbb{R}^8$:
  $$O_t = [T_{\text{core}}, T_{\text{cool}}, P_{\text{sys}}, F_{\text{cool}}, L_{\text{cpu}}, V_{\text{pos}}, \text{Vib}_{\text{pump}}, P_{\text{elec}}]^T$$
* **Normalization**: Each channel $i$ is standardized via mean $\mu_i^{\text{train}}$ and standard deviation $\sigma_i^{\text{train}}$:
  $$\tilde{O}_{t, i} = \frac{O_{t, i} - \mu_i^{\text{train}}}{\sigma_i^{\text{train}}}$$

### 3.2 Latent Recurrent World Model (Baseline 005)
* **Latent State Space**: Continuous 16-dimensional vector $z_t \in \mathbb{R}^{16}$.
* **Encoder**: Ingests context window of $T_{\text{ctx}} = 10$ steps ($5.0\text{ s}$) to infer $z_t \sim q(z_t \mid O_{1:t}, A_{1:t})$.
* **Transition Dynamics**: Recurrent GRU dynamics update the latent state under action $A_t$:
  $$z_{t+1} = z_t + f_{\theta}(z_t, A_t)$$
  The residual transition parameterization $z_{t+1} = z_t + \Delta z$ prevents vanishing gradients over multi-step horizons up to $H=40$ ($20.0\text{ s}$).
* **Decoder**: Predicts both physical means and predictive uncertainties:
  $$\hat{O}_{t+h} = g_{\phi}^{\mu}(z_{t+h}), \quad \sigma_{t+h} = \text{softplus}(g_{\phi}^{\sigma}(z_{t+h}))$$

### 3.3 Upstream Trust Gateway
Before allowing the planner to evaluate actions, PRISM checks whether the current telemetry is trustworthy:
1. **Dynamic Thermal Residual**:
   $$R_T = |O_{t, T_{\text{core}}} - \hat{O}_{t, T_{\text{core}}}| \le 6.0827^\circ\text{C} \quad (\text{98th percentile})$$
2. **Dynamic 8D Normalized Residual**:
   $$R_{8D} = \|\tilde{O}_t - \hat{\tilde{O}}_t\|_2 \le 1.8960 \quad (\text{98th percentile})$$
3. **Latent Mahalanobis Manifold Distance**:
   $$D_{\text{latent}} = \sqrt{(z_t - \mu_z)^T \Sigma_z^{-1} (z_t - \mu_z)} \le 15.00\text{ d}_M$$

*If any threshold is violated*, the system enters `MODEL_ABSTAIN`. Downstream planning is strictly blocked, and zero hallucinated actions are produced.

### 3.4 Causal Intervention Layer
Applies Pearl's structural intervention operator $do(A_t = a)$:
* Replaces the natural endogenous structural equation of the actuator with a constant setpoint $a$.
* Severes upstream causal edges into the target node while preserving downstream physical mechanisms.

### 3.5 Multi-Step Candidate Planner
* **Horizon**: $K=10$ steps ($5.0\text{ s}$).
* **Candidate Set**: Evaluates discrete set of operational interventions (do-nothing, valve modulation, throttle limits, pump speed changes, compound actions).
* **Multi-Objective Cost Model**:
  $$\text{Utility}(a) = w_{\text{thermal}} \cdot \text{Margin}_{\text{thermal}}(a) - w_{\text{slew}} \cdot \text{Cost}_{\text{slew}}(a) - w_{\text{power}} \cdot \text{Cost}_{\text{power}}(a)$$

### 3.6 Conservative Safety Gate
To prevent false-safe recommendations under model uncertainty, physical constraints are evaluated using conservative bounds:
$$T_{\text{eff}} = \mu_{T_{\text{core}}} + k \cdot \sigma_{T_{\text{core}}} < 95.0^\circ\text{C} \quad (k=2.0)$$
$$P_{\text{eff}} = \mu_{P_{\text{sys}}} + k \cdot \sigma_{P_{\text{sys}}} < 5.5\text{ bar}$$
$$F_{\text{eff}} = \mu_{F_{\text{cool}}} - k \cdot \sigma_{F_{\text{cool}}} > 8.0\text{ L/min}$$

Any candidate where $T_{\text{eff}} \ge 95.0^\circ\text{C}$ is rejected from execution.

### 3.7 Level-3 Counterfactual Engine
Executes retrospective twin-world evaluation:
1. **Abduction**: Inverts historical episode observations to abduce the exogenous noise sequence $U_{1:t}$.
2. **Action Substitution**: Replaces the historical action $A_t^{\text{fact}}$ with counterfactual candidate $A_t^{\text{cf}}$.
3. **Twin-World Rollout**: Propagates dynamics under $U_{1:t}$ and $A_t^{\text{cf}}$ to compute true counterfactual deltas $\Delta T_{\text{core}}, \Delta P_{\text{sys}}, \Delta F_{\text{cool}}$.

### 3.8 Unified Evidence Assembler & Cryptographic Hash
* Compiles all decisions, trust metrics, causal DAG attributions, counterfactual trajectories, and safety margins into an immutable `PrismDecisionRecord`.
* Serializes the record to canonical JSON (sorted keys, stripped insignificant whitespace).
* Computes a deterministic SHA-256 evidence fingerprint for auditability and tamper-detection.
