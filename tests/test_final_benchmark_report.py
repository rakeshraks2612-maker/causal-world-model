"""
PRISM — Final Benchmark Report Contract and Reproducibility Test Suite
Verifies that Phase 7.4 benchmark generation, outputs, and integrity assertions hold.
"""

import json
from pathlib import Path
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_DIR / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "benchmark_results.json"
MD_REPORT_PATH = REPORTS_DIR / "PRISM_Final_Benchmark_Report.md"


def test_benchmark_files_exist():
    assert JSON_REPORT_PATH.exists(), "benchmark_results.json must exist"
    assert MD_REPORT_PATH.exists(), "PRISM_Final_Benchmark_Report.md must exist"


def test_benchmark_json_schema():
    with open(JSON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    required_keys = [
        "metadata",
        "multistep_forecasting",
        "ood_and_uncertainty",
        "causal_intervention",
        "counterfactual_reasoning",
        "baseline_005_acceptance",
        "trust_calibration",
        "decision_planning",
        "safety_evaluation",
        "scenario_records",
        "tamper_evidence_audit",
    ]
    for key in required_keys:
        assert key in data, f"Missing required top-level key: {key}"


def test_tamper_evidence_audit():
    with open(JSON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tamper = data.get("tamper_evidence_audit", {})
    assert tamper.get("tamper_detected") is True, "Tamper mutation must be detectable"
    assert tamper.get("original_fingerprint") != tamper.get("tampered_fingerprint")


def test_markdown_report_structure():
    with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 16 mandatory sections
    required_sections = [
        "## 1. Executive Summary",
        "## 2. Evaluation Objectives",
        "## 3. PRISM System Under Evaluation",
        "## 4. Experimental Protocol",
        "## 5. World-Model Evaluation",
        "## 6. Causal Intervention Evaluation",
        "## 7. Counterfactual Evaluation",
        "## 8. Uncertainty & Model-Trust Evaluation",
        "## 9. Decision-Planning Evaluation",
        "## 10. Safety-Gate Evaluation",
        "## 11. Abstention Evaluation",
        "## 12. Evidence & Auditability Evaluation",
        "## 13. Failure Analysis",
        "## 14. Limitations",
        "## 15. Reproducibility",
        "## 16. Final Results Summary",
    ]
    for sec in required_sections:
        assert sec in content, f"Missing required markdown section: {sec}"


def test_metric_fidelity():
    with open(JSON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Verify exact frozen values
    ms = data["multistep_forecasting"]["prism_open_loop"]["by_horizon"]["40"]
    assert pytest.approx(ms["train_normalized_aggregate_mae"], 1e-4) == 0.21206
    
    cf = data["counterfactual_reasoning"]["counterfactual_metrics"]
    assert cf["total_records"] == 448
    assert pytest.approx(cf["overall_directional_accuracy"], 1e-4) == 0.774906
    assert pytest.approx(cf["peak_t_core_mae"], 1e-2) == 1.44
    assert cf["safety_summary"]["false_safe"] == 0

    calib = data["trust_calibration"]
    assert pytest.approx(calib["tau_residual_t_core"], 1e-4) == 6.082679
