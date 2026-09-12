# 🏛️ PRISM Demo Suite & Quick-Start Guide

Welcome to the **PRISM Interactive Decision Intelligence Demo Suite**.

PRISM demonstrates:
1. **Autonomous Causal Decision-Making:** Evaluating intervention candidates via SCM do-calculus and optimizing multi-objective Pareto utility.
2. **Uncertainty-Aware Safety Gating ($k=2$):** Catching unsafe actions where point estimates appear compliant but conservative confidence bounds breach physical thresholds.
3. **Fail-Closed Model Abstention:** Detecting observational inconsistency and aborting autonomous actuation when internal world-model trust is compromised.
4. **Cryptographic Tamper-Evidence:** Generating deterministic SHA-256 evidence fingerprints for all decision records.

---

## ⚡ Quick Start

### 1. Launch Interactive Streamlit Dashboard
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch dashboard
python3 scripts/run_dashboard.py
```
Open your browser at **`http://localhost:8501`**.

### 2. Run Headless CLI Demo
```bash
# Demo 1: Autonomous Decision (S4 - Pump Modulation)
python3 scripts/run_prism.py -s s4

# Demo 2: Uncertainty Safety Catch (S2 - Valve Intervention)
python3 scripts/run_prism.py -s s2

# Demo 3: Model Distrust Abstention (S6 - Emergency Regime)
python3 scripts/run_prism.py -s s6

# Execute Full 6-Scenario Benchmark
python3 scripts/run_prism.py -a
```

---

## 📁 Demo Folder Structure

```text
demo/
├── README.md               # Quick-start guide
├── demo_config.json        # Frozen demo configuration & operating thresholds
├── walkthrough.md          # Complete 5-minute judge walkthrough & pitch script
├── scenarios/              # Pre-generated decision records & dossiers (.json / .md)
│   ├── demo_01_decision.json
│   ├── demo_01_decision.md
│   ├── demo_02_uncertainty.json
│   ├── demo_02_uncertainty.md
│   ├── demo_03_abstention.json
│   └── demo_03_abstention.md
└── expected/               # Frozen authoritative expected reference records
    ├── demo_01_decision_expected.json
    ├── demo_02_uncertainty_expected.json
    └── demo_03_abstention_expected.json
```

---

## 🧪 Verification Test
To verify demo reproducibility and pipeline integrity:
```bash
pytest tests/test_dashboard.py tests/test_product_pipeline.py
```
