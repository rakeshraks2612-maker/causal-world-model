"""Demo Suite Reproducibility and Parity Tests (Task 7.3).

Verifies that the frozen demo suite (demo/scenarios/) produces bit-identical
outputs matching demo/expected/ and reproduces the exact SHA-256 hashes.
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest

from prism.pipeline.engine import PrismPipeline, PrismPipelineConfig

FIXED_TIMESTAMP = "2026-09-11T12:00:00Z"


@pytest.fixture(scope="module")
def pipeline():
    return PrismPipeline(PrismPipelineConfig(model_dir=Path("artifacts/baseline_005")))


def test_demo_01_decision_reproducibility(pipeline):
    """Verify Demo 1 (S4 - Pump) produces expected decision and bit-identical hash."""
    record = pipeline.run_scenario("scenario_04_pump", timestamp_utc=FIXED_TIMESTAMP)
    
    with open("demo/expected/demo_01_decision_expected.json") as f:
        expected = json.load(f)

    assert record.decision.recommendation == "cand_pump_3"
    assert record.decision.decision_status == "RECOMMENDED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.provenance.unified_record_hash == expected["provenance"]["unified_record_hash"]


def test_demo_02_uncertainty_safety_catch_reproducibility(pipeline):
    """Verify Demo 2 (S2 - Safety Catch) produces expected UNSAFE evaluation and matching hash."""
    record = pipeline.run_scenario("scenario_02_valve", timestamp_utc=FIXED_TIMESTAMP)

    with open("demo/expected/demo_02_uncertainty_expected.json") as f:
        expected = json.load(f)

    assert record.decision.recommendation == "cand_pump_4"
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "UNSAFE"
    assert record.safety.limiting_constraint.constraint_name == "thermal"
    assert record.provenance.unified_record_hash == expected["provenance"]["unified_record_hash"]


def test_demo_03_abstention_reproducibility(pipeline):
    """Verify Demo 3 (S6 - Abstention) produces fail-closed BLOCKED record and matching hash."""
    record = pipeline.run_scenario("scenario_06_all_unsafe", timestamp_utc=FIXED_TIMESTAMP)

    with open("demo/expected/demo_03_abstention_expected.json") as f:
        expected = json.load(f)

    assert record.decision.recommendation is None
    assert record.decision.decision_status == "BLOCKED"
    assert record.trust.trust_state == "MODEL_ABSTAIN"
    assert record.safety.overall_state == "ABSTAIN_REQUIRED"
    assert record.provenance.unified_record_hash == expected["provenance"]["unified_record_hash"]


def test_demo_script_structure():
    """Verify demo_script.md exists and covers all 6 narrative acts."""
    script_path = Path("demo/demo_script.md")
    assert script_path.exists(), "demo/demo_script.md must exist"
    
    content = script_path.read_text(encoding="utf-8")
    assert "ACT 0:" in content
    assert "ACT 1:" in content
    assert "ACT 2:" in content
    assert "ACT 3:" in content
    assert "ACT 4:" in content
    assert "ACT 5:" in content
    assert "ACT 6:" in content
    assert "6.0827" in content
    assert "97.16" in content
