"""Tests for Task 3.4D-B: Intervention-Aware Dataset Integrity and Separation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from prism.dataset.schema import LearnerEpisode, OracleEpisode, FORBIDDEN_LEARNER_KEYS
from prism.dataset.validators import validate_learner_isolation, validate_oracle_episode_integrity


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    root = Path("data/intervention_aware")
    if not (root / "manifest.json").exists():
        pytest.skip("Intervention-aware dataset not generated yet")
    return root


def test_learner_and_oracle_file_counts(dataset_root: Path) -> None:
    """Ensure matched file counts across learner and oracle directories."""
    train_orc = list((dataset_root / "oracle" / "train").glob("*.npz"))
    train_lrn = list((dataset_root / "learner" / "train").glob("*.npz"))
    val_orc = list((dataset_root / "oracle" / "validation").glob("*.npz"))
    val_lrn = list((dataset_root / "learner" / "validation").glob("*.npz"))

    assert len(train_orc) == 100
    assert len(train_lrn) == 100
    assert len(val_orc) == 20
    assert len(val_lrn) == 20


def test_strict_learner_isolation(dataset_root: Path) -> None:
    """Validate that Learner episodes contain no hidden states, noise, or forbidden keys."""
    learner_files = list((dataset_root / "learner" / "train").glob("*.npz"))[:10]
    for f in learner_files:
        ep = LearnerEpisode.load_npz(f)
        validate_learner_isolation(ep)
        assert ep.observations.shape == (121, 8)
        assert ep.actions.shape == (121, 4)
        for k in ep.metadata:
            assert k not in FORBIDDEN_LEARNER_KEYS


def test_oracle_episode_integrity(dataset_root: Path) -> None:
    """Validate complete 12-dim state and noise in Oracle episodes."""
    oracle_files = list((dataset_root / "oracle" / "train").glob("*.npz"))[:10]
    for f in oracle_files:
        orc = OracleEpisode.load_npz(f)
        validate_oracle_episode_integrity(orc)
        assert orc.ground_truth_states.shape == (121, 12)
        assert orc.exogenous_noise.shape == (121, 12)


def test_high_load_and_runaway_representation(dataset_root: Path) -> None:
    """Verify that dataset contains high load (>= 70%) and thermal runaways."""
    oracle_files = list((dataset_root / "oracle" / "train").glob("*.npz"))
    runaways = 0
    all_l_cpu = []

    for f in oracle_files:
        orc = OracleEpisode.load_npz(f)
        if orc.failure_latched:
            runaways += 1
        all_l_cpu.extend(orc.ground_truth_states[:, 4].tolist())

    l_arr = np.array(all_l_cpu)
    assert runaways >= 20, f"Expected >= 20 runaway episodes, got {runaways}"
    assert np.mean(l_arr >= 70.0) > 0.30, f"Expected > 30% of steps with L_cpu >= 70%, got {np.mean(l_arr >= 70.0):.2%}"


def test_held_out_benchmark_isolation(dataset_root: Path) -> None:
    """Verify zero overlap between training episodes and 220-record test benchmark."""
    train_files = {f.stem for f in (dataset_root / "oracle" / "train").glob("*.npz")}
    test_bench_files = {f.stem for f in Path("data/pilot/oracle/intervention").glob("*.npz")}

    # No overlapping names
    intersection = train_files.intersection(test_bench_files)
    assert len(intersection) == 0, f"Contamination detected! Overlapping files: {intersection}"
