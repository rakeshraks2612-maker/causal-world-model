"""Unit tests for SafetyConstraintEngine, SafetyResult, and Canonical Safety Contracts (Task 5.3).

Verifies:
1. Canonical constraint boundary semantics:
   - T_core in {94.999999, 95.000000, 95.000001}
   - P_sys in {5.499999, 5.500000, 5.500001}
   - F_cool in {8.000001, 8.000000, 7.999999}
   - D_latent in {15.000000, 15.000001}
2. Margin calculation fidelity (thermal, pressure, flow, support).
3. Severity classification (SAFE, MARGINAL, UNSAFE, ABSTAIN_REQUIRED).
4. Multi-violation tracking (all violations reported, not just first).
5. NaN/Inf fail-closed behavior (returns ABSTAIN_REQUIRED / unsafe).
6. Uncertainty-aware safety envelopes (k * sigma expansion).
7. Candidate-level safety filtering prior to ranking.
8. All-unsafe scenario empty feasible set and abstention handling.
"""

import pytest
import numpy as np

from prism.planning.safety_constraints import (
    SafetySeverity,
    SafetyViolationReason,
    DecisionSafetyConfig,
    SafetyResult,
    SafetyConstraintEngine,
)
from prism.planning.cost_model import DecisionCostConfig, ActionCostModel
from prism.planning.objectives import PlanningObjective, CandidateEvaluation
from prism.intervention.spec import InterventionSpec, InterventionType


@pytest.fixture
def safety_engine() -> SafetyConstraintEngine:
    config = DecisionSafetyConfig(
        thermal_hard_limit=95.0,
        pressure_hard_limit=5.50,
        flow_hard_limit=8.00,
        latent_novelty_hard_limit=15.0,
        thermal_marginal_threshold=85.0,
        pressure_marginal_threshold=4.80,
        flow_marginal_threshold=12.00,
        latent_marginal_threshold=10.0,
        uncertainty_multiplier_k=2.0,
    )
    return SafetyConstraintEngine(config)


# =========================================================================
# 1. Exact High-Precision Boundary Semantics
# =========================================================================

@pytest.mark.parametrize("t_core,expected_safe,expected_thermal_safe", [
    (94.999999, True, True),
    (95.000000, False, False),
    (95.000001, False, False),
])
def test_thermal_boundary_high_precision(safety_engine: SafetyConstraintEngine, t_core: float, expected_safe: bool, expected_thermal_safe: bool):
    res = safety_engine.evaluate(peak_t_core=t_core, max_pressure=3.0, min_flow=20.0, latent_novelty=5.0)
    assert res.is_safe == expected_safe
    assert res.thermal_safe == expected_thermal_safe
    if not expected_safe:
        assert SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value in res.violation_reasons


@pytest.mark.parametrize("p_sys,expected_safe,expected_pressure_safe", [
    (5.499999, True, True),
    (5.500000, False, False),
    (5.500001, False, False),
])
def test_pressure_boundary_high_precision(safety_engine: SafetyConstraintEngine, p_sys: float, expected_safe: bool, expected_pressure_safe: bool):
    res = safety_engine.evaluate(peak_t_core=70.0, max_pressure=p_sys, min_flow=20.0, latent_novelty=5.0)
    assert res.is_safe == expected_safe
    assert res.pressure_safe == expected_pressure_safe
    if not expected_safe:
        assert SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value in res.violation_reasons


@pytest.mark.parametrize("f_cool,expected_safe,expected_flow_safe", [
    (8.000001, True, True),
    (8.000000, False, False),
    (7.999999, False, False),
])
def test_flow_boundary_high_precision(safety_engine: SafetyConstraintEngine, f_cool: float, expected_safe: bool, expected_flow_safe: bool):
    res = safety_engine.evaluate(peak_t_core=70.0, max_pressure=3.0, min_flow=f_cool, latent_novelty=5.0)
    assert res.is_safe == expected_safe
    assert res.flow_safe == expected_flow_safe
    if not expected_safe:
        assert SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value in res.violation_reasons


@pytest.mark.parametrize("novelty,expected_safe,expected_support_safe", [
    (15.000000, True, True),
    (15.000001, False, False),
])
def test_latent_support_boundary_high_precision(safety_engine: SafetyConstraintEngine, novelty: float, expected_safe: bool, expected_support_safe: bool):
    res = safety_engine.evaluate(peak_t_core=70.0, max_pressure=3.0, min_flow=20.0, latent_novelty=novelty)
    assert res.is_safe == expected_safe
    assert res.support_safe == expected_support_safe
    if not expected_safe:
        assert SafetyViolationReason.LATENT_NOVELTY_EXCEEDED.value in res.violation_reasons
        assert res.severity == SafetySeverity.ABSTAIN_REQUIRED


# =========================================================================
# 2. Safety Margin Calculations
# =========================================================================

def test_margin_calculations(safety_engine: SafetyConstraintEngine):
    """Verify exact numerical margins for thermal, pressure, flow, and latent support."""
    res = safety_engine.evaluate(
        peak_t_core=91.20,
        max_pressure=4.20,
        min_flow=22.50,
        latent_novelty=6.40,
    )
    assert res.is_safe is True
    assert pytest.approx(res.margin_thermal, abs=1e-5) == (95.0 - 91.20)  # 3.80 °C
    assert pytest.approx(res.margin_pressure, abs=1e-5) == (5.50 - 4.20)  # 1.30 bar
    assert pytest.approx(res.margin_flow, abs=1e-5) == (22.50 - 8.00)     # 14.50 L/min
    assert pytest.approx(res.margin_support, abs=1e-5) == (15.0 - 6.40)   # 8.60


def test_negative_margins_on_violation(safety_engine: SafetyConstraintEngine):
    """Negative margins indicate exact extent of safety envelope breach."""
    res = safety_engine.evaluate(
        peak_t_core=97.50,
        max_pressure=5.80,
        min_flow=6.50,
        latent_novelty=18.20,
    )
    assert res.is_safe is False
    assert pytest.approx(res.margin_thermal, abs=1e-5) == -2.50
    assert pytest.approx(res.margin_pressure, abs=1e-5) == -0.30
    assert pytest.approx(res.margin_flow, abs=1e-5) == -1.50
    assert pytest.approx(res.margin_support, abs=1e-5) == -3.20


# =========================================================================
# 3. Severity Classification
# =========================================================================

def test_severity_safe_posture(safety_engine: SafetyConstraintEngine):
    """SAFE: all constraints satisfied with ample margins beyond buffer thresholds."""
    res = safety_engine.evaluate(
        peak_t_core=78.0,
        max_pressure=3.8,
        min_flow=25.0,
        latent_novelty=4.0,
    )
    assert res.is_safe is True
    assert res.severity == SafetySeverity.SAFE


def test_severity_marginal_posture(safety_engine: SafetyConstraintEngine):
    """MARGINAL: strictly inside hard envelope, but within warning buffer."""
    # Thermal marginal (88°C in [85°C, 95°C))
    res_t = safety_engine.evaluate(peak_t_core=88.0, max_pressure=3.8, min_flow=25.0, latent_novelty=4.0)
    assert res_t.is_safe is True
    assert res_t.severity == SafetySeverity.MARGINAL

    # Pressure marginal (5.1 bar in [4.8 bar, 5.5 bar))
    res_p = safety_engine.evaluate(peak_t_core=75.0, max_pressure=5.1, min_flow=25.0, latent_novelty=4.0)
    assert res_p.is_safe is True
    assert res_p.severity == SafetySeverity.MARGINAL

    # Flow marginal (10.0 L/min in (8.0 L/min, 12.0 L/min])
    res_f = safety_engine.evaluate(peak_t_core=75.0, max_pressure=3.8, min_flow=10.0, latent_novelty=4.0)
    assert res_f.is_safe is True
    assert res_f.severity == SafetySeverity.MARGINAL

    # Support marginal (12.0 in [10.0, 15.0])
    res_s = safety_engine.evaluate(peak_t_core=75.0, max_pressure=3.8, min_flow=25.0, latent_novelty=12.0)
    assert res_s.is_safe is True
    assert res_s.severity == SafetySeverity.MARGINAL


def test_severity_unsafe_posture(safety_engine: SafetyConstraintEngine):
    """UNSAFE: physical hard constraint violated."""
    res = safety_engine.evaluate(peak_t_core=96.0, max_pressure=3.8, min_flow=25.0, latent_novelty=4.0)
    assert res.is_safe is False
    assert res.severity == SafetySeverity.UNSAFE


def test_severity_abstain_required_posture(safety_engine: SafetyConstraintEngine):
    """ABSTAIN_REQUIRED: latent support limit exceeded."""
    res = safety_engine.evaluate(peak_t_core=75.0, max_pressure=3.8, min_flow=25.0, latent_novelty=16.5)
    assert res.is_safe is False
    assert res.severity == SafetySeverity.ABSTAIN_REQUIRED


# =========================================================================
# 4. Multi-Violation Reporting
# =========================================================================

def test_multiple_simultaneous_violations(safety_engine: SafetyConstraintEngine):
    """All violations must be reported, not early-stopped on the first."""
    res = safety_engine.evaluate(
        peak_t_core=98.0,
        max_pressure=5.9,
        min_flow=5.5,
        latent_novelty=19.0,
    )
    assert res.is_safe is False
    assert len(res.violations) == 4
    assert len(res.violation_reasons) == 4
    assert SafetyViolationReason.THERMAL_LIMIT_EXCEEDED.value in res.violation_reasons
    assert SafetyViolationReason.PRESSURE_LIMIT_EXCEEDED.value in res.violation_reasons
    assert SafetyViolationReason.FLOW_MINIMUM_VIOLATED.value in res.violation_reasons
    assert SafetyViolationReason.LATENT_NOVELTY_EXCEEDED.value in res.violation_reasons


# =========================================================================
# 5. NaN / Inf Fail-Closed Protection
# =========================================================================

@pytest.mark.parametrize("t_core,p_sys,f_cool,novelty", [
    (float("nan"), 3.0, 20.0, 5.0),
    (75.0, float("inf"), 20.0, 5.0),
    (75.0, 3.0, float("-inf"), 5.0),
    (75.0, 3.0, 20.0, float("nan")),
])
def test_nan_inf_fail_closed(safety_engine: SafetyConstraintEngine, t_core: float, p_sys: float, f_cool: float, novelty: float):
    """Safety evaluation MUST fail closed (reject / abstain) on any NaN/Inf input."""
    res = safety_engine.evaluate(peak_t_core=t_core, max_pressure=p_sys, min_flow=f_cool, latent_novelty=novelty)
    assert res.is_safe is False
    assert res.severity == SafetySeverity.ABSTAIN_REQUIRED
    assert SafetyViolationReason.INVALID_NUMERICAL_STATE.value in res.violation_reasons


# =========================================================================
# 6. Uncertainty-Aware Safety Policy
# =========================================================================

def test_uncertainty_adjusted_thermal_rejection(safety_engine: SafetyConstraintEngine):
    """Mean T_core is safe (93°C), but 2*sigma (2*1.5°C = 3°C -> 96°C) violates 95°C limit."""
    # Point estimate mode: Safe
    res_point = safety_engine.evaluate(
        peak_t_core=93.0,
        max_pressure=3.5,
        min_flow=20.0,
        t_core_std=1.5,
        use_uncertainty_bounds=False,
    )
    assert res_point.is_safe is True

    # Uncertainty-aware mode: Rejected
    res_unc = safety_engine.evaluate(
        peak_t_core=93.0,
        max_pressure=3.5,
        min_flow=20.0,
        t_core_std=1.5,
        use_uncertainty_bounds=True,
    )
    assert res_unc.is_safe is False
    assert SafetyViolationReason.UNCERTAINTY_SAFETY_VIOLATION.value in res_unc.violation_reasons
    assert res_unc.margin_thermal == (95.0 - (93.0 + 2.0 * 1.5))  # -1.0 °C


def test_uncertainty_adjusted_flow_rejection(safety_engine: SafetyConstraintEngine):
    """Mean flow is safe (9.0 L/min), but lower bound (9.0 - 2*1.0 = 7.0 L/min) violates 8.0 limit."""
    res_unc = safety_engine.evaluate(
        peak_t_core=75.0,
        max_pressure=3.5,
        min_flow=9.0,
        f_cool_std=1.0,
        use_uncertainty_bounds=True,
    )
    assert res_unc.is_safe is False
    assert SafetyViolationReason.UNCERTAINTY_SAFETY_VIOLATION.value in res_unc.violation_reasons


# =========================================================================
# 7. Candidate-Level Safety Filtering & Lexicographical Dominance
# =========================================================================

def test_candidate_safety_filtering_prior_to_ranking():
    """Unsafe candidates must be rejected from ranking rather than outbidding safe actions on utility."""
    cost_model = ActionCostModel()
    base_actions = [50.0, 100.0, 2.0, 0.0]

    # Candidate 1: 100% compute load, but unsafe T_core = 97°C
    breakdown_unsafe = cost_model.evaluate_cost(
        peak_t_core=97.0,
        max_pressure=3.5,
        min_flow=20.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[50.0, 100.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )

    # Candidate 2: Safe, throttled to 70% load, T_core = 80°C
    breakdown_safe = cost_model.evaluate_cost(
        peak_t_core=80.0,
        max_pressure=3.5,
        min_flow=20.0,
        mean_cpu_load=70.0,
        actions_at_t_star=[50.0, 70.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )

    assert not breakdown_unsafe.is_safe
    assert breakdown_safe.is_safe

    # Filter safe candidates: only Candidate 2 passes
    all_candidates = [breakdown_unsafe, breakdown_safe]
    feasible = [c for c in all_candidates if c.is_safe]

    assert len(feasible) == 1
    assert feasible[0] is breakdown_safe


def test_all_unsafe_abstention_semantics():
    """When all candidates violate safety envelopes, feasible set is empty and system must abstain."""
    cost_model = ActionCostModel()
    base_actions = [50.0, 100.0, 2.0, 0.0]

    evals = []
    # Create 5 candidates that all violate thermal or pressure constraints
    for t_val in [96.0, 97.5, 99.0, 102.0, 105.0]:
        bd = cost_model.evaluate_cost(
            peak_t_core=t_val,
            max_pressure=5.8,
            min_flow=6.0,
            mean_cpu_load=100.0,
            actions_at_t_star=[50.0, 100.0, 2.0, 0.0],
            baseline_actions_at_t_star=base_actions,
        )
        evals.append(bd)

    feasible = [c for c in evals if c.is_safe]
    assert len(feasible) == 0, "No candidates should be feasible"


def test_safety_result_serialization(safety_engine: SafetyConstraintEngine):
    """SafetyResult and DecisionSafetyConfig serialize cleanly to dictionary."""
    config_dict = safety_engine.config.to_dict()
    assert config_dict["thermal_hard_limit"] == 95.0
    assert config_dict["pressure_hard_limit"] == 5.50

    res = safety_engine.evaluate(peak_t_core=82.0, max_pressure=4.0, min_flow=20.0, latent_novelty=5.0)
    res_dict = res.to_dict()
    assert res_dict["is_safe"] is True
    assert res_dict["severity"] == "SAFE"
    assert "margin_thermal" in res_dict
    assert "violation_reasons" in res_dict
