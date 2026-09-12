"""
PRISM — Reproducibility Manifest and Environment Verification Tests
Phase 7.5: Validates configuration manifests, checkpoint assets, and environment documentation.
"""

import json
from pathlib import Path
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_DIR / "configs" / "frozen_system_config.json"
ENV_PATH = WORKSPACE_DIR / "reproducibility" / "environment.json"
REPRO_README_PATH = WORKSPACE_DIR / "reproducibility" / "README.md"
ROOT_README_PATH = WORKSPACE_DIR / "README.md"


def test_frozen_system_config_structure():
    assert CONFIG_PATH.exists(), "frozen_system_config.json must exist"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    assert config["system_name"] == "PRISM"
    assert config["system_version"] == "1.0.0-frozen"
    
    arch = config.get("architecture", {})
    assert arch.get("model_name") == "baseline_005"
    assert arch.get("latent_dimension") == 16
    assert arch.get("planning_horizon_steps") == 10
    
    safety = config.get("safety_constraints", {})
    assert safety.get("t_core_max_celsius") == 95.0
    assert safety.get("p_sys_max_bar") == 5.5
    assert safety.get("f_cool_min_lpm") == 8.0
    assert safety.get("uncertainty_multiplier_k") == 2.0
    
    trust = config.get("trust_gate_thresholds", {})
    assert pytest.approx(trust.get("tau_residual_t_core_celsius"), 1e-4) == 6.082679
    assert pytest.approx(trust.get("tau_residual_8d_normalized"), 1e-4) == 1.896005
    assert trust.get("tau_novelty_mahalanobis") == 15.0


def test_manifest_referenced_files_exist():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    checkpoint_rel = config["architecture"]["checkpoint_path"]
    checkpoint_path = WORKSPACE_DIR / checkpoint_rel
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} must exist"
    
    report_rel = config["benchmark_reference"]["report_path"]
    report_path = WORKSPACE_DIR / report_rel
    assert report_path.exists(), f"Report {report_path} must exist"


def test_reproducibility_environment_metadata():
    assert ENV_PATH.exists(), "environment.json must exist"
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        env = json.load(f)
    
    assert "system" in env
    assert "runtime" in env
    assert "core_libraries" in env
    assert "frozen_system" in env
    assert env["core_libraries"]["torch"] is not None


def test_documented_scripts_exist():
    scripts = [
        WORKSPACE_DIR / "scripts" / "run_prism.py",
        WORKSPACE_DIR / "scripts" / "run_dashboard.py",
        WORKSPACE_DIR / "scripts" / "run_final_benchmark.py",
    ]
    for s in scripts:
        assert s.exists(), f"Documented script {s} must exist on disk"
