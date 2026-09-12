# PRISM Decision & Gating Flow Diagram

This document details the control flow, trust triage, candidate rollout, and safety constraint arbitration logic within PRISM.

---

## 1. ASCII Decision & Gating Flow

```text
                     CURRENT TELEMETRY (O_t, z_t)
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   UPSTREAM TRUST GATE    │
                     │  • R_T ≤ 6.08°C          │
                     │  • R_8D ≤ 1.90           │
                     │  • D_latent ≤ 15.00 d_M  │
                     └─────────────┬────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
         [Any Check Violated]              [All Checks Passed]
                  │                                 │
                  ▼                                 ▼
       ┌─────────────────────┐           ┌─────────────────────┐
       │    MODEL_ABSTAIN    │           │    MODEL_TRUSTED    │
       │  • Fail-Closed Gate │           │  • Generate Actions │
       │  • Planning Blocked │           │  • Multi-step K=10  │
       │  • Rec = None       │           │  • MC Particles N=50│
       └──────────┬──────────┘           └──────────┬──────────┘
                  │                                 │
                  │                                 ▼
                  │                      ┌─────────────────────┐
                  │                      │ HARD SAFETY GATING  │
                  │                      │   (k = 2.0 Sigma)   │
                  │                      │ T_eff < 95.0°C      │
                  │                      │ P_eff < 5.5 bar     │
                  │                      │ F_eff > 8.0 L/min   │
                  │                      └──────────┬──────────┘
                  │                                 │
                  │               ┌─────────────────┴─────────────────┐
                  │               ▼                                   ▼
                  │      [Any Candidate Safe?]              [All Candidates Unsafe]
                  │               │                                   │
                  │               ▼                                   ▼
                  │    ┌─────────────────────┐             ┌─────────────────────┐
                  │    │ MULTI-OBJ UTILITY   │             │   CANDIDATE REJECT  │
                  │    │ Maximize U(a)       │             │ Find Limiting Viol. │
                  │    │ Select Optimal Action             │ Alert Operator      │
                  │    └──────────┬──────────┘             └──────────┬──────────┘
                  │               │                                   │
                  └───────────────┼───────────────────────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │  EMIT DECISION & AUDIT   │
                     └──────────────────────────┘
```

---

## 2. Mermaid Decision Flow

```mermaid
flowchart TD
    Start["Observation O_t & Latent State z_t"] --> TrustCheck{"Trust Evaluation<br/>R_T ≤ 6.08°C<br/>R_8D ≤ 1.90<br/>d_M ≤ 15.00"}

    TrustCheck -- "Violated" --> Abstain["MODEL_ABSTAIN<br/>Planning Blocked<br/>Recommendation: None"]
    TrustCheck -- "Satisfied" --> GenCandidates["Candidate Action Generation<br/>(Valve, Throttle, Pump, Compound)"]

    GenCandidates --> Rollout["Multi-Step Temporal Rollout (K=10)<br/>50 Monte Carlo Particles per Candidate"]
    Rollout --> DecTraj["Decode μ(t+1:t+K), σ(t+1:t+K)"]

    DecTraj --> SafetyCheck{"Safety Gate (k=2.0)<br/>T_eff = μ + 2σ < 95.0°C<br/>P_eff = μ + 2σ < 5.5 bar<br/>F_eff = μ - 2σ > 8.0 L/min"}

    SafetyCheck -- "At least one safe" --> MultiObj["Multi-Objective Utility Ranking<br/>Thermal Margin + Slew Penalty + Power Cost"]
    MultiObj --> Recommend["RECOMMENDED Optimal Candidate"]

    SafetyCheck -- "All unsafe" --> Reject["ALL UNSAFE<br/>Identify Limiting Bottleneck Constraint<br/>Retain Hold State / Alert"]

    Recommend --> Emit["Emit Immutable PrismDecisionRecord"]
    Reject --> Emit
    Abstain --> Emit
```
