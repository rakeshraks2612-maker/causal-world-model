"""
PRISM — Architecture Documentation and Specification Consistency Tests
Phase 7.6: Validates presence, consistency, and mathematical fidelity of architecture docs.
"""

from pathlib import Path
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = WORKSPACE_DIR / "docs"
ARCH_DIR = DOCS_DIR / "architecture"
DIAG_DIR = DOCS_DIR / "diagrams"


def test_architecture_documents_exist():
    expected_arch_files = [
        ARCH_DIR / "README.md",
        ARCH_DIR / "system_architecture.md",
        ARCH_DIR / "causal_architecture.md",
        ARCH_DIR / "uncertainty_architecture.md",
        ARCH_DIR / "decision_pipeline.md",
        ARCH_DIR / "evidence_architecture.md",
    ]
    for p in expected_arch_files:
        assert p.exists(), f"Architecture doc missing: {p}"


def test_diagram_documents_exist():
    expected_diag_files = [
        DIAG_DIR / "prism_system_architecture.md",
        DIAG_DIR / "prism_causal_graph.md",
        DIAG_DIR / "prism_decision_flow.md",
        DIAG_DIR / "prism_evidence_flow.md",
    ]
    for p in expected_diag_files:
        assert p.exists(), f"Diagram file missing: {p}"


def test_causal_architecture_content():
    causal_doc = (ARCH_DIR / "causal_architecture.md").read_text(encoding="utf-8")
    assert "Structural Causal Model" in causal_doc
    assert "Action Intervention" in causal_doc
    assert "State Intervention" in causal_doc
    assert "do(A" in causal_doc or "do(X" in causal_doc
    assert "T_amb" in causal_doc
    assert "T_core" in causal_doc


def test_uncertainty_architecture_content():
    unc_doc = (ARCH_DIR / "uncertainty_architecture.md").read_text(encoding="utf-8")
    assert "Predictive uncertainty" in unc_doc
    assert "MODEL_ABSTAIN" in unc_doc
    assert "6.0827" in unc_doc
    assert "1.8960" in unc_doc
    assert "15.00" in unc_doc
    assert "k = 2.0" in unc_doc or "k=2.0" in unc_doc
    assert "95.0" in unc_doc


def test_evidence_architecture_content():
    ev_doc = (ARCH_DIR / "evidence_architecture.md").read_text(encoding="utf-8")
    domains = [
        "1. decision",
        "2. trust",
        "3. causal_reasoning",
        "4. counterfactual",
        "5. safety",
        "6. abstention",
        "7. decision_quality",
        "8. provenance",
    ]
    for d in domains:
        assert d in ev_doc, f"Evidence domain missing from doc: {d}"
    assert "SHA-256" in ev_doc
    assert "pure immutable projection" in ev_doc.lower()


def test_system_architecture_diagrams():
    sys_diag = (DIAG_DIR / "prism_system_architecture.md").read_text(encoding="utf-8")
    assert "flowchart TD" in sys_diag or "graph TD" in sys_diag
    assert "Trust Gateway" in sys_diag or "TrustCheck" in sys_diag
    assert "MODEL_ABSTAIN" in sys_diag
