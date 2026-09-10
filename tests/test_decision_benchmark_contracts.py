"""Tests for Task 5.1 Frozen Decision Benchmark Contracts and Firewall."""

from __future__ import annotations
import json
import pytest
import numpy as np
from pathlib import Path

from prism.dataset.schema import FORBIDDEN_LEARNER_KEYS
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario, DecisionClass


@pytest.fixture(scope="module")
def benchmark_dir():
    b_dir = Path("data/decision_benchmark")
    assert b_dir.exists(), "data/decision_benchmark must exist"
    return b_dir


def test_decision_benchmark_structure_and_manifest(benchmark_dir):
    """Verify that all 6 scenarios and manifest are present and valid."""
    manifest_path = benchmark_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["total_scenarios"] == 6
    assert len(manifest["scenarios"]) == 6

    expected_scenarios = [
        "scenario_01_do_nothing",
        "scenario_02_valve",
        "scenario_03_throttle",
        "scenario_04_pump",
        "scenario_05_combined",
        "scenario_06_all_unsafe",
    ]

    for scen_id in expected_scenarios:
        scen_path = benchmark_dir / "scenarios" / scen_id
        assert scen_path.exists(), f"Scenario directory {scen_id} must exist"
        assert (scen_path / "learner.npz").exists(), f"Learner record for {scen_id} must exist"
        assert (scen_path / "oracle.npz").exists(), f"Oracle record for {scen_id} must exist"


def test_learner_firewall_and_latent_isolation(benchmark_dir):
    """Verify strict firewall: learner record must contain zero unobserved oracle states."""
    scenarios_dir = benchmark_dir / "scenarios"
    for scen_folder in scenarios_dir.iterdir():
        if not scen_folder.is_dir():
            continue

        learner_file = scen_folder / "learner.npz"
        learner_scen = LearnerDecisionScenario.load_npz(learner_file)

        # 1. Check observations and action shapes
        t_pre = learner_scen.intervention_time + 1
        assert learner_scen.historical_observations.shape == (t_pre, 8)
        assert learner_scen.historical_actions.shape == (t_pre, 4)
        assert len(learner_scen.candidate_actions) >= 4

        # 2. Check support metadata for forbidden keys
        for k in learner_scen.support_metadata:
            assert k not in FORBIDDEN_LEARNER_KEYS, f"Forbidden key '{k}' found in learner support_metadata"

        # 3. Check direct keys inside npz
        data = np.load(learner_file, allow_pickle=True)
        for forbidden in FORBIDDEN_LEARNER_KEYS:
            assert forbidden not in data.files, f"Forbidden key '{forbidden}' found in learner npz archive"


def test_oracle_scenario_ground_truth_consistency(benchmark_dir):
    """Verify oracle records have ground truth optimal candidate and valid decision classes."""
    scenarios_dir = benchmark_dir / "scenarios"
    for scen_folder in scenarios_dir.iterdir():
        if not scen_folder.is_dir():
            continue

        oracle_file = scen_folder / "oracle.npz"
        orc_data = np.load(oracle_file, allow_pickle=True)

        assert "true_latent_state_t_star" in orc_data
        assert "candidate_outcomes" in orc_data
        assert "oracle_optimal_candidate_id" in orc_data
        assert "expected_decision_class" in orc_data
        assert orc_data["expected_decision_class"].item() in [DecisionClass.RECOMMEND.value, DecisionClass.CAUTION.value, DecisionClass.ABSTAIN.value]
