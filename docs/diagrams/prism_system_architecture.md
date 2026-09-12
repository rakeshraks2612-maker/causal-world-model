# PRISM System Architecture Diagram

This document contains both ASCII and Mermaid representations of the complete end-to-end PRISM system architecture.

---

## 1. ASCII System Architecture

```text
                                INDUSTRIAL WORLD
                   (High-Density Server & Liquid Cooling Loop)
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      TELEMETRY INGESTION      │
                       │    8D Observations (O_t)      │
                       │      Z-Score Normalizer       │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │       PRISM WORLD MODEL       │
                       │   16D Latent State Space (z)  │
                       │   Recurrent GRU Dynamics      │
                       │   Decoders (Mean μ, Sigma σ)  │
                       └───────────────┬───────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
  ┌─────────────────────────────┐             ┌─────────────────────────────┐
  │     TRUST GATEWAY           │             │     PREDICTIVE FORECASTS    │
  │  • R_T ≤ 6.08°C (Thermal)   │             │  • Multi-horizon H=1..40    │
  │  • R_8D ≤ 1.90 (8D Norm)    │             │  • Heteroscedastic Sigma σ  │
  │  • D_latent ≤ 15.00 d_M     │             │  • Monte Carlo Particles    │
  └──────────────┬──────────────┘             └──────────────┬──────────────┘
                 │                                           │
        [Outside Trust Boundary]                             │
                 ├──► [MODEL_ABSTAIN] ──► (Fail-Closed)     │
                 │                                           │
         [Inside Trust]                                      │
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   CAUSAL INTERVENTION LAYER   │
                       │    Pearl Structural SCM       │
                       │    Action Surgery do(A=a)     │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │       DECISION PLANNER        │
                       │   Candidate Generation        │
                       │   Temporal Rollout (K=10)     │
                       │   Multi-Objective Utility     │
                       └───────────────┬───────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
  ┌─────────────────────────────┐             ┌─────────────────────────────┐
  │   LEVEL-3 COUNTERFACTUAL    │             │  CONSERVATIVE SAFETY GATE   │
  │  • Exogenous Noise Abduct.  │             │  • T_eff = μ + 2σ < 95.0°C  │
  │  • Action Replacement       │             │  • P_eff = μ + 2σ < 5.5 bar │
  │  • Twin-World Replay        │             │  • F_eff = μ - 2σ > 8.0 L/m │
  └──────────────┬──────────────┘             └──────────────┬──────────────┘
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │    UNIFIED DECISION RECORD    │
                       │    8-Domain Evidence Stack    │
                       │    Deterministic SHA-256      │
                       └───────────────────────────────┘
```

---

## 2. Mermaid System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Observation Layer"]
        World["Physical Plant (Telemetry)"] --> Obs["8D Observation Normalization"]
    end

    subgraph WorldModel["2. Latent World Model (Baseline 005)"]
        Obs --> Enc["Encoder q(z_t | O, A)"]
        Enc --> Latent["16D Latent State (z_t)"]
        Latent --> Trans["Recurrent GRU Dynamics"]
        Trans --> Dec["Decoder (Mean μ, Sigma σ)"]
    end

    subgraph TrustGate["3. Model Trust Gateway"]
        Latent --> Gate{"Trust Check<br/>R_T ≤ 6.08°C<br/>R_8D ≤ 1.90<br/>d_M ≤ 15.00"}
        Gate -- "Outside Boundary" --> Abstain["MODEL_ABSTAIN<br/>(Fail-Closed Blocking)"]
    end

    subgraph CausalPlanning["4. Causal Planning & Counterfactuals"]
        Gate -- "Trusted" --> SCM["Structural Causal Engine do(A)"]
        SCM --> Planner["Multi-Step Planner (K=10)"]
        Planner --> CF["Level-3 Counterfactual Twin Replay"]
        Planner --> Safety{"Safety Gate (k=2.0)<br/>T_eff < 95°C<br/>P_eff < 5.5 bar<br/>F_eff > 8 L/min"}
    end

    subgraph DecisionEvidence["5. Decision & Evidence"]
        Safety -- "Safe" --> Rec["RECOMMENDED Action"]
        Safety -- "Unsafe" --> Rej["UNSAFE (Candidate Rejected)"]
        Rec --> Assembly["Unified Evidence Assembler"]
        Rej --> Assembly
        Abstain --> Assembly
        Assembly --> SHA["Deterministic SHA-256 Fingerprint"]
    end
```
