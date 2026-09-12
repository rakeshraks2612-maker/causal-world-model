"""Unit and Integration Tests for PRISM Product Pipeline & CLI (Task 7.1).

Verifies the end-to-end production pipeline:
1. Pipeline initialization with frozen baseline_005 model
2. Single scenario execution (S1 - Inaction Restraint)
3. Emergency abstention execution (S6 - Upfront Distrust)
4. Telemetry file execution (.npz)
5. Artifact serialization (JSON & Markdown)
6. Full 6-scenario benchmark execution
7. CLI terminal banner formatting
8. Scenario alias resolution
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np

from prism.pipeline.engine import PrismPipeline, PrismPipelineConfig
from prism.cli import format_terminal_banner, parse_scenario_alias
from prism.explanation.unified_record import PrismDecisionRecord


@pytest.fixture(scope="module")
def pipeline():
    """Shared pipeline instance loaded with frozen baseline_005."""
    cfg = PrismPipelineConfig(
        model_dir=Path("artifacts/baseline_005"),
        benchmark_dir=Path("data/decision_benchmark"),
    )
    return PrismPipeline(cfg)


def test_pipeline_initialization(pipeline):
    """Verify pipeline properly initializes all model, planner, and trust components."""
    assert pipeline.model is not None
    assert pipeline.normalizer is not None
    assert pipeline.trust_evaluator is not None
    assert pipeline.planner is not None
    assert pipeline.cf_engine is not None


def test_pipeline_run_scenario_s1(pipeline, tmp_path):
    """Verify pipeline runs S1 end-to-end and saves decision record + dossier."""
    out_dir = tmp_path / "s1_run"
    record = pipeline.run_scenario("scenario_01_do_nothing", output_dir=out_dir, timestamp_utc="2026-09-11T12:00:00Z")

    assert isinstance(record, PrismDecisionRecord)
    assert record.decision.recommendation == "cand_do_nothing"
    assert record.decision.decision_status == "NO_ACTION_REQUIRED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.safety.overall_state in ["SAFE", "MARGINAL"]
    assert len(record.provenance.unified_record_hash) == 64

    # Verify saved files
    json_path = out_dir / "scenario_01_do_nothing_decision_record.json"
    md_path = out_dir / "scenario_01_do_nothing_audit_dossier.md"
    assert json_path.exists()
    assert md_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert data["decision"]["recommendation"] == "cand_do_nothing"
    assert data["provenance"]["unified_record_hash"] == record.provenance.unified_record_hash


def test_pipeline_run_scenario_s6_abstention(pipeline, tmp_path):
    """Verify pipeline runs S6 emergency regime and enforces fail-closed abstention."""
    out_dir = tmp_path / "s6_run"
    record = pipeline.run_scenario("scenario_06_all_unsafe", output_dir=out_dir, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation is None
    assert record.decision.decision_status == "BLOCKED"
    assert record.trust.trust_state == "MODEL_ABSTAIN"
    assert record.trust.reconstruction_residual_t_core > 6.0
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "ABSTAIN_REQUIRED"
    assert record.abstention.abstained is True
    assert "33.88" in record.abstention.primary_reason


def test_pipeline_run_file(pipeline, tmp_path):
    """Verify pipeline runs directly from a standalone .npz telemetry file."""
    fpath = Path("data/decision_benchmark/scenarios/scenario_03_throttle/learner.npz")
    out_dir = tmp_path / "file_run"
    record = pipeline.run_file(fpath, output_dir=out_dir, timestamp_utc="2026-09-11T12:00:00Z")

    assert record.decision.recommendation == "cand_throttle_50"
    assert record.decision.decision_status == "RECOMMENDED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True


def test_pipeline_all_scenarios(pipeline, tmp_path):
    """Verify batch execution over all 6 benchmark scenarios."""
    out_dir = tmp_path / "all_runs"
    results = pipeline.run_all_benchmark_scenarios(output_dir=out_dir)

    assert len(results) == 6
    assert "scenario_01_do_nothing" in results
    assert "scenario_06_all_unsafe" in results
    assert results["scenario_01_do_nothing"].decision.recommendation == "cand_do_nothing"
    assert results["scenario_06_all_unsafe"].decision.recommendation is None


def test_cli_scenario_alias_parsing():
    """Verify alias mapping for user-friendly CLI invocation."""
    assert parse_scenario_alias("s1") == "scenario_01_do_nothing"
    assert parse_scenario_alias("s6") == "scenario_06_all_unsafe"
    assert parse_scenario_alias("1") == "scenario_01_do_nothing"
    assert parse_scenario_alias("scenario_02_valve") == "scenario_02_valve"


def test_cli_terminal_banner_formatting(pipeline):
    """Verify formatted ASCII banner renders all required fields."""
    record = pipeline.run_scenario("scenario_04_pump", timestamp_utc="2026-09-11T12:00:00Z")
    banner = format_terminal_banner(record)

    assert "PRISM DECISION INTELLIGENCE ENGINE" in banner
    assert "scenario_04_pump" in banner
    assert "cand_pump_3" in banner
    assert "MODEL_TRUSTED" in banner
    assert record.provenance.unified_record_hash in banner
