# PRISM Reproducibility Guide

This document provides the complete, deterministic recipe to reproduce all experimental findings, datasets, model checkpoints, multi-step rollout diagnostics, and causal intervention benchmarks reported in the repository.

---

## 1. System Requirements & Environment Setup

* **Python Version:** Supported on `Python >= 3.10, <= 3.12` (Tested with Python 3.12.13 on macOS / Linux).
* **Dependencies:** Strictly pinned in [`requirements.txt`](./requirements.txt).

```bash
# 1. Clone repository
git clone https://github.com/rakeshraks2612-maker/causal-world-model.git
cd causal-world-model

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Deterministic Pipeline Execution Order

Execute the following sequential pipeline:

### Step 1: Generate Ground-Truth Benchmark & Pilot Datasets
Generates observational, OOD regimes, confounding, and twin-world intervention/counterfactual datasets.

```bash
# Generate observational and OOD pilot datasets (Train: 400 ep, Val: 50 ep, Test: 50 ep)
PYTHONPATH=. python scripts/generate_pilot_dataset.py

# Generate paired intervention pilot records (220 paired episodes under frozen world)
PYTHONPATH=. python scripts/generate_intervention_pilot.py

# Generate twin-world counterfactual pilot records
PYTHONPATH=. python scripts/generate_counterfactual_pilot.py
```

---

### Step 2: Train the Canonical PRISM Baseline World Model (`baseline_002`)
Trains the variational causal world model ($d_z=64, \beta_{\text{KL}}=0.001$, context=40, seed=42):

```bash
PYTHONPATH=. python scripts/train_baseline_002.py
```

*Outputs stored in `artifacts/baseline_002/`:*
* `best.pt`: Canonical model weights checkpoint
* `config.yaml`: Frozen architectural hyperparameters
* `normalization.yaml`: Train-set-only z-score normalization statistics
* `metrics.json`: Epoch loss trajectories & probe scores

---

### Step 3: Run Multi-Step Rollout & OOD Uncertainty Diagnostics (Task 3.3 / 3.3B)

```bash
# Multi-step open-loop vs teacher-forced forecasting evaluation across horizons h in {1, 5, 10, 20, 40}
PYTHONPATH=. python scripts/evaluate_task_3_3.py

# Latent Mahalanobis OOD distance & predictive uncertainty calibration diagnostics
PYTHONPATH=. python scripts/diagnose_ood_uncertainty.py
```

---

### Step 4: Run Learned Causal Intervention Benchmark (Task 3.4A)
Executes graph surgery $do(X = x)$ vs action control $A = a$ across all 220 paired intervention records and computes $E_{\text{causal}}$, $E_{\text{rel}}$, and safety classification:

```bash
PYTHONPATH=. python scripts/evaluate_learned_interventions.py
```

*Outputs stored in `artifacts/intervention/`:*
* `intervention_benchmark_report.json`: Full metrics report
* `plots/causal_error_by_horizon.png`: Causal error curves
* `plots/intervention_ladder_curves.png`: Intervention ladder monotonicity

---

## 3. Automated Verification & Test Suite

Run the full test suite with coverage assertions:

```bash
PYTHONPATH=. pytest --cov=prism --cov-report=term-missing
```

### Expected Pass Criteria:
* **Total Tests:** 141+ tests passing.
* **Code Coverage:** $\ge 90\%$ total statement coverage.
* **Key Invariants Verified:**
  1. $do(V_{\text{pos}}=85)$ instantly clamps at $t^*+1$ ($\Delta V_{\text{pos}} = 0$), while $A_{\text{valve}}=85$ exhibits dynamic lag ($\Delta V_{\text{pos}} > 1.0\%$).
  2. $do(Vib_{\text{pump}}=v)$ produces strictly zero causal effect on thermal and hydraulic channels.
  3. Pre-intervention trajectories are identical ($t \le t^*$).
  4. Directionality: $V_{\text{pos}} \uparrow \implies F_{\text{cool}} \uparrow, T_{\text{core}} \downarrow$; $L_{\text{cpu}} \uparrow \implies T_{\text{core}} \uparrow$.

---

## 4. Expected Metric Tolerances

| Metric | Target Variable / Scope | Expected Value / Tolerance |
| :--- | :--- | :--- |
| **Open-Loop MAE ($h=40$)** | Global Observation Vector | $0.212 \pm 0.020$ |
| **Causal Error $E_{\text{causal}}$ ($h=10$)** | Hydraulic Pressure $P_{\text{sys}}$ | $0.157 \pm 0.030\text{ bar}$ |
| **Causal Error $E_{\text{causal}}$ ($h=10$)** | Coolant Flow $F_{\text{cool}}$ | $4.672 \pm 0.500\text{ L/min}$ |
| **Overall Directional Concordance** | All Channels & Horizons | $\ge 80.0\%$ |
| **Failure Classification Accuracy** | Safety Thresholds | $\ge 90.0\%$ |
| **Non-Descendant Invariance** | $do(Vib_{\text{pump}})$ on Thermal | $E_{\text{causal}} \le 0.05$ |
