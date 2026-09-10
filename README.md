# PRISM: Uncertainty-Aware Causal World Model & Intervention Engine

**PRISM** is an uncertainty-aware causal world model and counterfactual simulation framework designed for complex physical and cyber-physical systems (grounded on the **Thermal-Hydraulic-Compute Structural Causal Model, THC-SCM**).

PRISM enables autonomous agents to answer Pearl's Level 2 (Interventional) and Level 3 (Counterfactual) queries:
> *"Given partial, noisy observations up to time $t^*$, what will happen if we perform a graph-surgical intervention $do(X=x)$ or action control $A=a$ right now?"*

---

## Architecture Overview

```text
Observations O[0:t] + Actions A[0:t]
                │
                ▼
    Sequence Encoder qφ(Zt | O≤t, A≤t)
                │
                ▼
        Inferred Latent State Zt
                │
    ┌───────────┴───────────┐
    ▼                       ▼
Natural Action           do(X=x) / A=a
    │                       │
    └───────────┬───────────┘
                ▼
    Latent Transition pθ(Zt+1 | Zt, At)
                │
                ▼
      Observation Decoder pθ(Ot | Zt)
                │
                ▼
    Counterfactual & Intervened Trajectory
```

### Key Modules

- **`prism/simulator/`**: Ground-truth THC-SCM physics engine (12 state variables: 8 observable, 4 hidden latents, multi-timescale thermal/fluid dynamics, realistic sensor noise, and failure modes).
- **`prism/world_model/`**: Variational Causal World Model binding GRU Sequence Encoder, MLP Latent Transition Model, MLP Decoder, and Autoregressive Rollout Engine.
- **`prism/intervention/`**: Graph-surgical intervention operators ($do(X=x)$ state clamps vs $A=a$ action setpoints), persistent and pulse duration controls, and paired twin-world intervention simulator.
- **`prism/evaluation/`**: Multi-step open-loop forecasting, uncertainty calibration & OOD diagnostics, action branching sensitivity, and causal effect benchmark metrics ($E_{\text{causal}} = |\Delta Y_{\text{learned}} - \Delta Y_{\text{oracle}}|$).
- **`prism/dataset/`**: Paired twin-world dataset generators for observational, OOD regimes, confounding benchmarks, interventions, and counterfactuals.

---

## Project Status

- [x] **Task 1: Ground-Truth Causal Simulator** (Physics, SCM semantics, noise, failures)
- [x] **Task 2.1 - 2.8: Dataset & Benchmark Infrastructure** (Observational, OOD, Confounding, Interventions, Counterfactuals)
- [x] **Task 3.1: World Model Contract & Architecture Specification**
- [x] **Task 3.2 - 3.2B: Baseline World Model Training & Diagnostics** (`baseline_002` canonical checkpoint)
- [x] **Task 3.3 - 3.3B: Multi-Step Rollout & OOD Uncertainty Calibration**
- [x] **Task 3.4: Learned Intervention Simulator & Causal Benchmark** (220 paired intervention records, 10 causal invariants verified)

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/rakeshraks2612-maker/causal-world-model.git
cd causal-world-model

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

```bash
# Run complete test suite with coverage
PYTHONPATH=. pytest --cov=prism --cov-report=term-missing
```

### Evaluating Learned Interventions

```bash
# Generate intervention pilot dataset
PYTHONPATH=. python scripts/generate_intervention_pilot.py

# Benchmark learned PRISM against oracle ground truth
PYTHONPATH=. python scripts/evaluate_learned_interventions.py
```

---

## Citation & License

Apache-2.0 License. Developed as part of the PRISM Causal World Model Initiative.
