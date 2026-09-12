# PRISM Reproducibility Guide

This guide details the exact environment, deterministic random seeds, configuration manifests, and step-by-step instructions to reproduce all empirical evaluations, demo scenarios, and test suites in the PRISM repository.

---

## 1. System Requirements

* **Operating System**: Linux (Ubuntu 22.04+ recommended) or macOS (Apple Silicon / Intel).
* **Python Runtime**: Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.13).
* **Hardware**: Standard x86_64 or ARM64 CPU. (GPU optional; CPU inference runs all benchmarks in < 15s).
* **Memory**: $\ge 4\text{ GB RAM}$.

---

## 2. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/causal-world-model.git
cd causal-world-model

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verification & Execution Steps

### Step 1: Run Full Regression Test Suite (410 Tests)
```bash
pytest
```
*Expected Result*: `410 passed in ~13s`

### Step 2: Run Master Benchmark Aggregator & Integrity Audit
```bash
python3 scripts/run_final_benchmark.py
```
*Expected Result*:
* Serializes machine-readable metrics to `reports/benchmark_results.json`
* Compiles 16-section publication report to `reports/PRISM_Final_Benchmark_Report.md`
* Confirms 100% tamper detection on mutated decision records.

### Step 3: Execute CLI Decision Scenarios
```bash
# Scenario 4: Pump Modulation (Autonomous Decision)
python3 scripts/run_prism.py -s s4

# Scenario 2: Valve Boundary (Uncertainty Safety Catch)
python3 scripts/run_prism.py -s s2

# Scenario 6: Thermal Runaway (Model Abstention)
python3 scripts/run_prism.py -s s6
```

### Step 4: Launch Decision Intelligence Dashboard
```bash
python3 scripts/run_dashboard.py
```
*Expected Result*: Launches local Streamlit dashboard on `http://localhost:8501`.

---

## 4. Frozen Assets & Manifest References

| Asset | Path | Description |
| :--- | :--- | :--- |
| **Model Checkpoint** | `artifacts/baseline_005/best.pt` | Frozen PyTorch state dict for RSSM-GRU world model |
| **Trust Calibration** | `artifacts/baseline_005/trust_calibration.json` | 98th percentile calibrated boundaries ($\tau_{R_T} = 6.0827^\circ\text{C}, \tau_{R_{8D}} = 1.8960, \tau_{D_{\text{latent}}} = 15.00$) |
| **System Config** | `configs/frozen_system_config.json` | Master configuration manifest |
| **Environment Specs** | `reproducibility/environment.json` | Exact library versions & platform metadata |
| **Demo Config** | `demo/demo_config.json` | Frozen inputs and expected outcomes for S4, S2, and S6 |
| **Final Report** | `reports/PRISM_Final_Benchmark_Report.md` | Comprehensive 16-section evaluation report |
