# PRISM Technical Architecture Reference

Welcome to the comprehensive technical architecture documentation for **PRISM** (**P**redictive **R**ecurrent **I**ntervention & **S**afety **M**odel), an uncertainty-aware causal decision intelligence engine designed for industrial cyber-physical systems.

---

## 1. Architectural Levels of Understanding

To serve judges, engineers, and research scientists, PRISM's documentation is structured into three distinct conceptual levels:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: PRODUCT & MISSION                                                      │
│ What does PRISM do?                                                             │
│ • Replaces black-box forecasting with intervention-aware causal reasoning       │
│ • Evaluates uncertainty-adjusted hard safety constraints (k=2.0σ)               │
│ • Abstains when model trust is insufficient rather than fabricating actions     │
│ • Packages auditable evidence into deterministic SHA-256 decision records       │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: SYSTEM PIPELINE & DATA FLOW                                            │
│ How do subsystems connect?                                                      │
│ • Telemetry Ingestion → RSSM World Model → Upstream Trust Gateway               │
│ • Pearl Structural Causal Interventions → Level-3 Counterfactual Twin Replay    │
│ • Multi-Step Candidate Planner → Conservative Safety Gate → Evidence Assembly   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: SCIENTIFIC & MATHEMATICAL FORMALISMS                                   │
│ What is learned vs. deterministic? What is causal vs. predictive?               │
│ • 16D Continuous Latent RSSM GRU Dynamics with Exogenous Noise Abduction        │
│ • Predictive Variance vs. Latent Support Manifold Distance (Mahalanobis d_M)   │
│ • Dynamic Telemetry Residual Gating (R_T, R_8D) calibrated at 98th percentile   │
│ • Pearl Level-3 Retrospective Counterfactuals (Abduction → Action → Replay)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Modules Index

| Document | Primary Focus | Key Concepts Covered |
| :--- | :--- | :--- |
| **[System Architecture](system_architecture.md)** | End-to-end subsystem pipeline | Data flow, module boundaries, learned vs deterministic components |
| **[Causal Architecture](causal_architecture.md)** | Structural Causal Model (SCM) | Pearl do-calculus, Action vs State Interventions, Graph surgery, Invariants |
| **[Uncertainty & Trust Architecture](uncertainty_architecture.md)** | Uncertainty vs Trust distinction | Predictive variance ($\sigma$), Latent Mahalanobis ($d_M$), Dynamic residuals ($R_T, R_{8D}$), Abstention |
| **[Decision Pipeline](decision_pipeline.md)** | Planning & optimization engine | Multi-step candidate generation ($K=10$), Multi-objective utility, Slew costs, Safety boundaries |
| **[Evidence Architecture](evidence_architecture.md)** | Auditable decision record | 8-domain evidence structure, Pure projection layer, Canonical JSON, SHA-256 fingerprint |

---

## 3. Visual Diagrams Index

| Diagram Document | Format | Description |
| :--- | :--- | :--- |
| **[System Architecture Diagram](../diagrams/prism_system_architecture.md)** | ASCII & Mermaid | Complete end-to-end subsystem layout from telemetry to SHA-256 record |
| **[Causal Graph Topology](../diagrams/prism_causal_graph.md)** | ASCII & Mermaid | 12-variable ThermoHydro-Compute Structural Causal Model |
| **[Decision & Gating Flow](../diagrams/prism_decision_flow.md)** | ASCII & Mermaid | Upstream trust triage, candidate rollout, and safety gating logic |
| **[Evidence Assembly Flow](../diagrams/prism_evidence_flow.md)** | ASCII & Mermaid | 8-domain evidence aggregation into canonical SHA-256 fingerprint |
