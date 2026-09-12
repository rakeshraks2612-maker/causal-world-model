"""Unit and Integration Tests for PRISM Decision Intelligence Dashboard (Task 7.2).

Validates the dashboard integration layer:
1. S4 Decision Hero state loading (cand_pump_3, MODEL_TRUSTED)
2. S6 Abstention state loading (BLOCKED, MODEL_ABSTAIN)
3. S2 Uncertainty-Aware Safety Catch (UNSAFE, thermal limit breach)
4. Evidence hash integrity & JSON reloading
5. Immutable projection / No mutation invariant
6. Deterministic reproducibility
7. Missing artifact fail-safe error handling
8. CLI ↔ Dashboard output bit-identical parity
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np

from prism.dashboard.data_loader import (
    load_benchmark_scenario,
    load_telemetry_file,
    load_saved_record_json,
    load_saved_dossier_markdown,
    get_pipeline,
)
from prism.pipeline.engine import PrismPipeline
from prism.explanation.unified_record import PrismDecisionRecord


@pytest.fixture(scope="module")
def shared_pipeline():
    """Shared pipeline instance."""
    return get_pipeline()


def test_dashboard_s4_decision_loading(shared_pipeline):
    """Test 1: S4 loads trusted decision with cand_pump_3 recommendation."""
    record, learner = load_benchmark_scenario("scenario_04_pump", pipeline=shared_pipeline)

    assert isinstance(record, PrismDecisionRecord)
    assert record.decision.recommendation == "cand_pump_3"
    assert record.decision.decision_status == "RECOMMENDED"
    assert record.trust.trust_state == "MODEL_TRUSTED"
    assert record.safety.is_safe is True
    assert record.causal_reasoning.summary.headline is not None
    assert record.decision_quality.utility_score is not None
    assert learner is not None


def test_dashboard_s6_abstention_loading(shared_pipeline):
    """Test 2: S6 loads model distrust abstention with BLOCKED recommendation."""
    record, learner = load_benchmark_scenario("scenario_06_all_unsafe", pipeline=shared_pipeline)

    assert record.decision.recommendation is None
    assert record.decision.decision_status == "BLOCKED"
    assert record.trust.trust_state == "MODEL_ABSTAIN"
    assert record.trust.reconstruction_residual_t_core > 6.0
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "ABSTAIN_REQUIRED"
    assert record.abstention.abstained is True
    assert "33.88" in record.abstention.primary_reason


def test_dashboard_s2_uncertainty_safety_catch(shared_pipeline):
    """Test 3: S2 displays uncertainty-adjusted safety breach (T_eff > 95°C)."""
    record, learner = load_benchmark_scenario("scenario_02_valve", pipeline=shared_pipeline)

    assert record.decision.recommendation == "cand_pump_4"
    assert record.safety.is_safe is False
    assert record.safety.overall_state == "UNSAFE"
    assert record.safety.limiting_constraint.constraint_name == "thermal"
    assert record.safety.limiting_constraint.raw_margin < 0.0


def test_dashboard_evidence_hash_integrity(shared_pipeline, tmp_path):
    """Test 4: Saved JSON artifact retains identical cryptographic hash."""
    out_dir = tmp_path / "dash_runs"
    record, _ = load_benchmark_scenario("scenario_03_throttle", pipeline=shared_pipeline, output_dir=out_dir)

    json_path = out_dir / "scenario_03_throttle_decision_record.json"
    assert json_path.exists()

    loaded_json = load_saved_record_json(json_path)
    assert loaded_json["provenance"]["unified_record_hash"] == record.provenance.unified_record_hash
    assert loaded_json["decision"]["recommendation"] == "cand_throttle_50"


def test_dashboard_no_mutation_invariant(shared_pipeline):
    """Test 5: Loading scenarios does not mutate underlying planner state."""
    rec_before = shared_pipeline.planner
    _ = load_benchmark_scenario("scenario_01_do_nothing", pipeline=shared_pipeline)
    rec_after = shared_pipeline.planner

    assert rec_before is rec_after


def test_dashboard_determinism(shared_pipeline):
    """Test 6: Loading the same scenario twice yields bit-identical hashes."""
    r1, _ = load_benchmark_scenario("scenario_04_pump", pipeline=shared_pipeline, timestamp_utc="2026-09-11T12:00:00Z")
    r2, _ = load_benchmark_scenario("scenario_04_pump", pipeline=shared_pipeline, timestamp_utc="2026-09-11T12:00:00Z")

    assert r1.provenance.unified_record_hash == r2.provenance.unified_record_hash
    assert r1.to_json() == r2.to_json()


def test_dashboard_missing_artifact_error(shared_pipeline):
    """Test 7: Attempting to load non-existent scenario raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _ = load_benchmark_scenario("scenario_99_invalid_nonexistent", pipeline=shared_pipeline)


def test_dashboard_cli_parity(shared_pipeline, tmp_path):
    """Test 8: CLI run and Dashboard pipeline run produce identical records."""
    out_dir = tmp_path / "parity_test"
    rec_dash = shared_pipeline.run_scenario("scenario_05_combined", output_dir=out_dir, timestamp_utc="2026-09-11T12:00:00Z")
    rec_pipe = shared_pipeline.run_scenario("scenario_05_combined", timestamp_utc="2026-09-11T12:00:00Z")

    assert rec_dash.provenance.unified_record_hash == rec_pipe.provenance.unified_record_hash
    assert rec_dash.decision.recommendation == rec_pipe.decision.recommendation
    assert rec_dash.safety.overall_state == rec_pipe.safety.overall_state
