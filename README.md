# PRISM
## AI Decision Intelligence for Cyber-Physical Systems

> **An uncertainty-aware causal world model that evaluates interventions, simulates counterfactual outcomes, applies hard safety constraints, and abstains when model trust is insufficient.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-causal--world--model.onrender.com-46e3b7?style=for-the-badge&logo=render&logoColor=white)](https://causal-world-model.onrender.com)

[![CI](https://github.com/rakeshraks2612-maker/causal-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rakeshraks2612-maker/causal-world-model/actions)
[![Tests](https://img.shields.io/badge/tests-421%20passed-brightgreen.svg)](#reproducibility)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](#quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](#quick-start)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](#docker--container-deployment)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b.svg)](#run-the-dashboard)
[![Provenance](https://img.shields.io/badge/audit-SHA--256%20deterministic-blueviolet.svg)](#evidence--auditability)

[[🌐 Live Web Dashboard](https://causal-world-model.onrender.com)] &nbsp;|&nbsp; [[Demo Walkthrough](demo/walkthrough.md)] &nbsp;|&nbsp; [[Architecture Documentation](docs/architecture/README.md)] &nbsp;|&nbsp; [[Final Benchmark Report](reports/PRISM_Final_Benchmark_Report.md)] &nbsp;|&nbsp; [[Render Guide](docs/DEPLOYMENT_RENDER.md)] &nbsp;|&nbsp; [[Quickstart Notebook](notebooks/01_quickstart_tutorial.ipynb)] &nbsp;|&nbsp; [[Reproducibility Guide](reproducibility/README.md)]

---

## The Problem

Industrial cyber-physical systems (data center liquid cooling loops, energy distribution networks, turbine governors) operate in high-consequence environments where incorrect actions cause physical destruction, runaway equipment damage, or multi-million-dollar outages.

Existing predictive AI systems suffer from three fatal flaws when applied to physical control:
1. **Correlation vs. Causation**: Pure observational models confuse correlation with causation, mistaking passive sensor drift for actuator control effects.
2. **Deceptive Point Estimates**: Forecasting models output single-point trajectories that appear safe (e.g. $94.35^\circ\text{C}$ against a $95.0^\circ\text{C}$ threshold) while completely ignoring predictive uncertainty that violates hard physical limits ($97.16^\circ\text{C}$).
3. **Hallucinatory Incompetence**: When sensors fail, telemetry corrupts, or dynamics transition into unmodeled emergency regimes, traditional ML models blindly fabricate recommendations rather than refusing to act.

---

## What PRISM Does

PRISM (**P**redictive **R**ecurrent **I**ntervention & **S**afety **M**odel) shifts physical control from black-box prediction to **auditable causal decision intelligence**:

* **Observes**: Ingests continuous multi-channel cyber-physical telemetry ($T_{\text{core}}, T_{\text{cool}}, P_{\text{sys}}, F_{\text{cool}}, L_{\text{cpu}}, V_{\text{pos}}, \text{Vib}_{\text{pump}}, P_{\text{elec}}$).
* **World-Models**: Forecasts open-loop multi-step dynamics over extended horizons ($H=40$ steps / $20.0\text{ s}$) using a 16D latent recurrent state space.
* **Intervenes**: Simulates explicit structural causal interventions ($do(A)$) on physical actuators (pumps, valves, throttles) rather than naive observational conditioning.
* **Counterfactualizes (Pearl Level-3)**: Abducts historical exogenous disturbances to answer: *"What would have happened under an alternative action under identical initial conditions?"*
* **Gates on Uncertainty**: Evaluates conservative confidence bounds ($\mu + 2\sigma$) against hard physical boundaries ($95^\circ\text{C}$ thermal cap, $5.5\text{ bar}$ pressure cap, $8\text{ L/min}$ flow floor).
* **Abstains When Untrusted**: Measures upfront dynamic residuals ($R_T$) and latent manifold novelty ($D_{\text{latent}}$). If telemetry violates trust boundaries ($R_T > 6.08^\circ\text{C}$), PRISM triggers `MODEL_ABSTAIN` and fails closed with zero hallucinated actions.
* **Proves Cryptographically**: Packages every decision, causal graph attribution, counterfactual trajectory, safety margin, and trust metric into a deterministic, tamper-evident `PrismDecisionRecord` with a SHA-256 fingerprint.

---

## Core Architecture

```text
                               TELEMETRY INPUT
                        (8-Channel Sensor Stream)
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │       UPSTREAM TRUST GATE     │
                   │   R_T ≤ 6.08°C, d_M ≤ 15.00   │
                   └───────┬───────────────┬───────┘
          [Outside Trust]  │               │  [Trusted]
                  ┌────────┘               └────────┐
                  ▼                                 ▼
         ┌─────────────────┐               ┌─────────────────┐
         │  MODEL_ABSTAIN  │               │   WORLD MODEL   │
         │   Fail-Closed   │               │ 16D RSSM Latent │
         │ Planning Blocked│               │ Dynamics & Dec. │
         └────────┬────────┘               └────────┬────────┘
                  │                                 │
                  │                 ┌───────────────┴───────────────┐
                  │                 ▼                               ▼
                  │        ┌─────────────────┐             ┌─────────────────┐
                  │        │   CAUSAL SCM    │             │ COUNTERFACTUAL  │
                  │        │ Structural DAG  │             │ Level-3 Abduct/ │
                  │        │  Interventions  │             │ Replay Traj.    │
                  │        └────────┬────────┘             └────────┬────────┘
                  │                 └───────────────┬───────────────┘
                  │                                 ▼
                  │                        ┌─────────────────┐
                  │                        │ UNCERTAINTY &   │
                  │                        │ SAFETY GATING   │
                  │                        │ T_eff = μ + 2σ  │
                  │                        └────────┬────────┘
                  │                                 │
                  │                                 ▼
                  │                        ┌─────────────────┐
                  │                        │ DECISION ENGINE │
                  │                        │ Multi-Objective │
                  │                        │ Action Ranking  │
                  │                        └────────┬────────┘
                  │                                 │
                  └─────────────────┬───────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   UNIFIED DECISION RECORD   │
                     │  8 Domains + SHA-256 Hash   │
                     └─────────────────────────────┘
```

---

## Why PRISM Is Different

| Capability | Traditional Predictive AI | PRISM Causal Decision Engine |
| :--- | :--- | :--- |
| **Paradigm** | Observation $\to$ Forecast $\to$ Action | Observation $\to$ World Model $\to$ Causal Interventions $\to$ Counterfactuals $\to$ Uncertainty Safety $\to$ Decision / Abstention |
| **Action Reasoning** | Correlational feature conditioning | Structural Causal Model ($do(A)$ interventions) |
| **Retrospective Analysis** | Re-run forward forecast (confounded) | Pearl Level-3 twin-world exogenous noise abduction |
| **Safety Evaluation** | Raw point-estimate ($\hat{y} < \text{threshold}$) | Uncertainty-aware conservative bound ($\mu + 2\sigma < \text{threshold}$) |
| **Out-of-Distribution** | Silent overconfident extrapolation | Dual-gate trust layer: Fail-closed `MODEL_ABSTAIN` |
| **Auditability** | Opaque black-box outputs | Machine-readable 8-domain evidence with deterministic SHA-256 fingerprint |

---

## Three Demo Scenarios

PRISM's core value proposition is demonstrated across three canonical scenarios:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. AUTONOMOUS DECISION (S4 — Pump Modulation)                                   │
│    Cooling deficit detected → PRISM evaluates candidates → Simulates Pump 3    │
│    Causal Chain: Pump ↑ → Flow +10.81 L/min → T_cool ↓ → T_core -2.87°C        │
│    Status: MODEL_TRUSTED | Safety: SAFE (+5.48°C margin) | Decision: RECOMMENDED│
├─────────────────────────────────────────────────────────────────────────────────┤
│ 2. UNCERTAINTY SAFETY CATCH (S2 — Valve / Thermal Boundary)                     │
│    Point Estimate: T_peak = 94.35°C < 95.00°C (Deceptively appears SAFE)       │
│    Uncertainty-Adjusted (k=2.0, σ=1.405°C): T_eff = 97.16°C > 95.00°C          │
│    Safety Gate: UNSAFE (Margin: -2.16°C) → Candidate REJECTED from execution    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 3. FAIL-CLOSED ABSTENTION (S6 — Emergency Runaway / Distrust)                   │
│    Telemetry shows acute sensor/physics contradiction: R_T = 33.88°C > 6.08°C   │
│    Trust Gate: MODEL_ABSTAIN (Excess +27.80°C) → Planning BLOCKED               │
│    Counterfactuals: NOT RUN | Recommendation: NONE (BLOCKED) | Zero Hallucination│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/rakeshraks2612-maker/causal-world-model.git
cd causal-world-model

python3 -m venv .venv
source .venv/bin/activate

# Install package & dependencies in editable mode
make install-dev
# or: pip install -e ".[dev]"
```

### 2. Run Test Suite (421 Invariant Proofs)
```bash
make test
# or: pytest -v tests/
```
*Expected: 421/421 tests passing (100% invariant & physics satisfaction).*

---

## Developer Ergonomics (`Makefile`)

A single unified `Makefile` streamlines all developer workflows:

```bash
make help          # View all available targets and descriptions
make test          # Run full 421-test suite with execution profiling
make cli           # Execute PRISM decision pipeline across all 6 benchmark scenarios
make dashboard     # Launch the interactive Streamlit Operator Console (port 8501)
make web           # Serve the lightweight PRISM Web SPA (port 8000)
make clean         # Purge build artifacts, pytest cache, and temporary files
```

---

## Docker & Container Deployment

Deploy PRISM instantly with a single command via Docker and Docker Compose:

```bash
# Build and run both Operator Console (:8501) and Web SPA (:8000)
docker-compose up -d

# Check service logs
docker-compose logs -f

# Teardown
docker-compose down
```

---

## ☁️ Live Cloud Deployment on Render

> 🚀 **Active Production Instance:** [**https://causal-world-model.onrender.com**](https://causal-world-model.onrender.com)
>
> The production PRISM Web Dashboard is live on Render, featuring zero-overhead interactive telemetry streams, scenario docks (`● S1`..`● S6`), Pearl Level-3 counterfactual rollouts, and cryptographic SHA-256 invariant audit dossiers.

PRISM is also configured for immediate 1-click deployment to [Render](https://render.com) via native Blueprints ([`render.yaml`](render.yaml)):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rakeshraks2612-maker/causal-world-model)

For detailed step-by-step instructions, see the [Render Deployment Guide](docs/DEPLOYMENT_RENDER.md).

---

## Interactive Tutorial Notebook

Explore PRISM step-by-step in an interactive Jupyter environment:
* Notebook location: [`notebooks/01_quickstart_tutorial.ipynb`](notebooks/01_quickstart_tutorial.ipynb)
* Demonstrates: World model loading, telemetry trust audit ($R_T$, $R_{8D}$, $D_{\text{lat}}$), Pearl Level-3 twin-world counterfactual simulation, $k=2$ uncertainty bounds, and SHA-256 cryptographic provenance hashing.

---

## Run the Dashboard

Launch the interactive Streamlit Decision Intelligence Dashboard:

```bash
make dashboard
# or: streamlit run scripts/run_dashboard.py --server.port 8501
```

Then open `http://localhost:8501` in your browser to inspect live scenario selectors, structural causal DAG projections, counterfactual twin-world overlays, thermal uncertainty bands, and cryptographic audit panels.

For the lightweight standalone Web SPA dashboard:
```bash
make web
# Then navigate to http://localhost:8000
```

---

## Run the CLI

Execute deterministic inference directly from your terminal using the installed `prism` CLI:

```bash
# Evaluate all 6 benchmark scenarios with formatted executive banners
prism --all

# S4: Autonomous Decision (Pump Modulation Recommended)
prism -s s4

# S2: Uncertainty-Aware Safety Catch (Unsafe Gated)
prism -s s2

# S6: Model Abstention (Distrust Blocked)
prism -s s6

# Output raw JSON or full Markdown audit dossier:
prism -s s4 --json
prism -s s4 --markdown
```

---

## Reproduce the Benchmark

Run the frozen benchmark aggregator and integrity verification engine:

```bash
python3 scripts/run_final_benchmark.py
```

Outputs:
* `reports/benchmark_results.json` (Machine-readable benchmark dataset)
* `reports/PRISM_Final_Benchmark_Report.md` (Publication-grade 16-section report)

---

## Evaluation Results

All metrics reflect frozen evaluation against authoritative benchmarks:

| Subsystem / Capability | Key Metric | Target / Baseline | PRISM Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **World Model ($H=40$ Rollout)** | Normalized MAE | MLP ($0.4920$) / Pers ($0.3230$) | **0.2121** ($-56.9\%$ vs MLP) | **PASS** |
| **Error Growth ($H=1 \to H=40$)** | Error Compounding | MLP ($+220.3\%$) | **+24.4%** | **PASS** |
| **Causal Interventions** | Directional Accuracy | Random ($50.0\%$) | **86.1%** | **PASS** |
| **Counterfactual Reasoning** | Directional Accuracy | Random ($50.0\%$) | **77.49%** ($347/448$) | **PASS** |
| **Counterfactual Accuracy** | Peak $T_{\text{core}}$ MAE | $\le 2.50^\circ\text{C}$ | **1.44°C** | **PASS** |
| **Counterfactual Safety** | False-Safe Rate | $\le 1.0\%$ | **0.00%** ($0/448$) | **PASS** |
| **Uncertainty Safety Catch** | S2 Effective Temp | $T_{\text{eff}} > 95.0^\circ\text{C}$ | **Caught at 97.16°C** | **PASS** |
| **Model Abstention Recall** | S6 Distrust Recall | $100.0\%$ | **100.0%** ($R_T = 33.88^\circ\text{C}$) | **PASS** |
| **Model False Trust Rate** | S6 False Trust | $0.0\%$ | **0.00%** | **PASS** |
| **Evidence Tamper Detection**| SHA-256 Mutation | 100% Detectable | **Deterministic / 100% Detectable** | **PASS** |
| **Regression Test Suite** | Total Tests | 100% Passing | **414 / 414 Passed** | **PASS** |

For complete methodology and forensic analyses, see [PRISM Final Benchmark Report](reports/PRISM_Final_Benchmark_Report.md).

---

## Evidence & Auditability

Every decision produces an immutable, machine-readable `PrismDecisionRecord` structured into 8 domains:

```json
{
  "decision": {
    "status": "RECOMMENDED",
    "recommended_action": "cand_pump_3",
    "action_name": "Pump Speed 3",
    "expected_utility": 0.0034
  },
  "trust": {
    "status": "MODEL_TRUSTED",
    "residual_t_core": 1.3548,
    "latent_mahalanobis_d": 5.6609
  },
  "causal_reasoning": {
    "primary_mechanism": "A_pump → F_cool → T_cool → T_core",
    "direct_effect": "+10.81 L/min coolant flow"
  },
  "counterfactual": {
    "factual_action": "A_pump_2",
    "counterfactual_action": "A_pump_3",
    "delta_t_core": -2.87
  },
  "safety": {
    "overall_safety": "SAFE",
    "effective_peak_t_core": 89.52,
    "thermal_margin_celsius": 5.48
  },
  "provenance": {
    "evidence_fingerprint": "18d2046b6c3b2a3f3f4fd0cfdc5997f9fc27d50aabe577e9735c28953bfaf304"
  }
}
```

---

## Repository Structure

```text
causal-world-model/
├── configs/
│   └── frozen_system_config.json        # Master frozen system specification
├── demo/
│   ├── demo_config.json                 # Golden demo configuration
│   ├── walkthrough.md                   # Step-by-step judge narrative walkthrough
│   ├── scenarios/                       # Scenario JSON inputs (S1 - S6)
│   └── expected/                        # Deterministic expected records
├── prism/
│   ├── world_model/                     # RSSM latent dynamics, encoders, decoders
│   ├── causal/                          # Structural causal graph & intervention ops
│   ├── counterfactual/                  # Level-3 exogenous abduction & twin-world replay
│   ├── uncertainty/                     # Particle estimator & calibration
│   ├── safety/                          # Conservative physical hard constraints
│   ├── planning/                        # Multi-step candidate evaluation & ranking
│   ├── explanation/                     # Causal, counterfactual, & abstention evidence
│   ├── pipeline/                        # End-to-end inference engine (PrismPipeline)
│   └── dashboard/                       # Streamlit decision intelligence app
├── reports/
│   ├── PRISM_Final_Benchmark_Report.md  # 16-section publication benchmark report
│   └── benchmark_results.json           # Machine-readable benchmark dataset
├── reproducibility/
│   ├── README.md                        # Hardware, setup & verification guide
│   └── environment.json                 # Environment library versions & metadata
├── scripts/
│   ├── run_prism.py                     # CLI decision entrypoint
│   ├── run_dashboard.py                 # Dashboard launcher
│   └── run_final_benchmark.py           # Benchmark runner & aggregator
├── tests/                               # 414 unit, integration, & contract tests
├── requirements.txt                     # Pinned reproducible dependencies
└── README.md                            # Primary project documentation
```

---

## Limitations

1. **Simulator-Based Evaluation**: PRISM was evaluated in a high-fidelity continuous-time numerical cooling simulator. Real-world deployment requires physical plant calibration and transfer validation.
2. **Defined Operational Envelope**: The causal graph topology is parameterized for defined closed-loop industrial topologies. Structural changes to plant piping require updating the causal graph specification.
3. **Epistemic Uncertainty Proxy**: True Bayesian epistemic uncertainty over neural network parameters is approximated via latent manifold distance ($D_{\text{latent}}$) and upfront dynamic residuals ($R_T$).
4. **Abstention Scope**: 100% abstention recall is verified on evaluated telemetry corruption and thermal runaway regimes; performance on arbitrary unobserved catastrophic sensor failure modes remains bounded by residual threshold sensitivity.
5. **Research Prototype Status**: PRISM is a research prototype demonstrating uncertainty-aware causal decision intelligence and is **not certified as safety-critical industrial control software**.

---

## Reproducibility

PRISM is designed for 100% deterministic reproducibility across platforms.

To verify the entire pipeline from scratch:
```bash
# Run all 414 test assertions
pytest

# Run final benchmark evaluation
python3 scripts/run_final_benchmark.py

# Test golden demo scenarios
python3 scripts/run_prism.py -s s4
python3 scripts/run_prism.py -s s2
python3 scripts/run_prism.py -s s6
```

Detailed hardware, library versions, and platform manifests are documented in [reproducibility/README.md](reproducibility/README.md).

---

## License & Citation

PRISM is released under the [Apache 2.0 License](LICENSE).
