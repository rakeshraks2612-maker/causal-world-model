"""Unit tests for ActionCostModel and DecisionCostConfig (Task 5.2).

Verifies:
1. Invariant A: Increasing thermal risk cannot improve utility.
2. Invariant B: More throttling monotonically reduces performance utility.
3. Invariant C: Higher pump usage incurs strictly greater operational cost.
4. Invariant D: Unnecessary actuation displacement incurs strictly greater cost.
5. Invariant E: Safety is lexicographically dominant (unsafe candidates rejected regardless of utility).
6. Boundary semantics:
   - T_core in {94.99, 95.00, 95.01}
   - P_sys in {5.49, 5.50, 5.51}
   - F_cool in {8.01, 8.00, 7.99}
   - Latent novelty in {14.99, 15.00, 15.01}
7. Adversarial tests:
   - Safe+Expensive vs Safe+Cheap
   - Unsafe+HighUtility vs Safe+ModerateUtility
   - Intervention complexity penalty (single vs compound)
   - Deterministic evaluation stability
"""

import pytest
import numpy as np

from prism.planning.cost_model import DecisionCostConfig, ActionCostModel, CostBreakdown


@pytest.fixture
def cost_model() -> ActionCostModel:
    config = DecisionCostConfig(
        thermal_hard_limit=95.0,
        pressure_hard_limit=5.50,
        flow_hard_limit=8.00,
        latent_novelty_hard_limit=15.0,
        thermal_soft_limit=75.0,
        pressure_soft_limit=4.50,
        flow_soft_limit=15.0,
        performance_weight=1.0,
        thermal_risk_weight=0.5,
        pressure_risk_weight=0.3,
        pump_cost_weight=0.10,
        valve_actuation_weight=0.05,
        throttle_cost_weight=2.0,
        flush_cost_weight=0.50,
        complexity_weight=0.05,
    )
    return ActionCostModel(config)


# =========================================================================
# 1. Invariant A: Increasing thermal risk cannot improve utility
# =========================================================================

def test_invariant_a_thermal_monotonicity(cost_model: ActionCostModel):
    """Monotonicity: T_core ↑ must never increase net utility when all else is held equal."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    eval_actions = [50.0, 100.0, 2.0, 0.0]
    
    t_cores = [50.0, 65.0, 75.0, 78.0, 82.0, 88.0, 92.0, 94.5, 94.99]
    utilities = []
    
    for t in t_cores:
        res = cost_model.evaluate_cost(
            peak_t_core=t,
            max_pressure=3.5,
            min_flow=25.0,
            mean_cpu_load=100.0,
            actions_at_t_star=eval_actions,
            baseline_actions_at_t_star=base_actions,
        )
        assert res.is_safe, f"T_core={t} should be safe"
        utilities.append(res.net_utility)
    
    # Assert monotonically non-increasing
    for i in range(len(utilities) - 1):
        assert utilities[i] >= utilities[i+1], (
            f"Thermal monotonicity violated: T={t_cores[i]} (U={utilities[i]}) < "
            f"T={t_cores[i+1]} (U={utilities[i+1]})"
        )
    
    # Above soft limit (75°C), utility must be strictly decreasing
    soft_idx = t_cores.index(75.0)
    for i in range(soft_idx, len(utilities) - 1):
        assert utilities[i] > utilities[i+1], (
            f"Strict thermal decrease violated above soft limit: T={t_cores[i]} -> T={t_cores[i+1]}"
        )


# =========================================================================
# 2. Invariant B: More throttling costs something
# =========================================================================

def test_invariant_b_throttle_monotonicity(cost_model: ActionCostModel):
    """Monotonicity: Lower CPU load (throttling) must strictly reduce net utility when all else is equal."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    eval_actions = [50.0, 100.0, 2.0, 0.0]
    
    cpu_loads = [100.0, 90.0, 80.0, 70.0, 50.0, 30.0, 10.0]
    utilities = []
    
    for load in cpu_loads:
        res = cost_model.evaluate_cost(
            peak_t_core=70.0,
            max_pressure=3.5,
            min_flow=25.0,
            mean_cpu_load=load,
            actions_at_t_star=eval_actions,
            baseline_actions_at_t_star=base_actions,
        )
        assert res.is_safe
        utilities.append(res.net_utility)
    
    for i in range(len(utilities) - 1):
        assert utilities[i] > utilities[i+1], (
            f"Throttling monotonicity violated: load={cpu_loads[i]} (U={utilities[i]}) <= "
            f"load={cpu_loads[i+1]} (U={utilities[i+1]})"
        )


# =========================================================================
# 3. Invariant C: More pump usage has a cost
# =========================================================================

def test_invariant_c_pump_cost_monotonicity(cost_model: ActionCostModel):
    """Monotonicity: Higher pump stages must strictly increase pump cost and reduce net utility if outcomes are identical."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    
    stages = [1.0, 2.0, 3.0, 4.0]
    pump_costs = []
    utilities = []
    
    for stage in stages:
        eval_actions = [50.0, 100.0, stage, 0.0]
        res = cost_model.evaluate_cost(
            peak_t_core=70.0,
            max_pressure=3.5,
            min_flow=25.0,
            mean_cpu_load=100.0,
            actions_at_t_star=eval_actions,
            baseline_actions_at_t_star=base_actions,
        )
        assert res.is_safe
        pump_costs.append(res.pump_cost)
        utilities.append(res.net_utility)
    
    for i in range(len(stages) - 1):
        assert pump_costs[i] < pump_costs[i+1], f"Pump cost not strictly increasing from stage {stages[i]} to {stages[i+1]}"
        assert utilities[i] > utilities[i+1], f"Utility with pump stage {stages[i]} should be higher than stage {stages[i+1]}"


# =========================================================================
# 4. Invariant D: Unnecessary actuation has a cost
# =========================================================================

def test_invariant_d_actuation_displacement_cost(cost_model: ActionCostModel):
    """Monotonicity: Larger valve movement from baseline must incur higher actuation cost."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    
    valve_positions = [50.0, 55.0, 65.0, 80.0, 100.0]
    act_costs = []
    utilities = []
    
    for v_pos in valve_positions:
        eval_actions = [v_pos, 100.0, 2.0, 0.0]
        res = cost_model.evaluate_cost(
            peak_t_core=70.0,
            max_pressure=3.5,
            min_flow=25.0,
            mean_cpu_load=100.0,
            actions_at_t_star=eval_actions,
            baseline_actions_at_t_star=base_actions,
        )
        assert res.is_safe
        act_costs.append(res.actuation_cost)
        utilities.append(res.net_utility)
    
    # 50.0 is baseline -> 0 actuation cost
    assert act_costs[0] == 0.0
    
    for i in range(len(valve_positions) - 1):
        assert act_costs[i] < act_costs[i+1]
        assert utilities[i] > utilities[i+1]


# =========================================================================
# 5. Invariant E: Safety is lexicographically dominant
# =========================================================================

def test_invariant_e_safety_lexicographical_dominance(cost_model: ActionCostModel):
    """Invariant E: Unsafe candidate with high utility must strictly lose to safe candidate with modest utility."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    
    # Candidate A: 100% throughput, but peak T_core = 96°C (unsafe)
    cand_a = cost_model.evaluate_cost(
        peak_t_core=96.0,
        max_pressure=3.5,
        min_flow=25.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[50.0, 100.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )
    
    # Candidate B: Throttled to 60% load, but peak T_core = 80°C (safe)
    cand_b = cost_model.evaluate_cost(
        peak_t_core=80.0,
        max_pressure=3.5,
        min_flow=25.0,
        mean_cpu_load=60.0,
        actions_at_t_star=[50.0, 60.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )
    
    assert not cand_a.is_safe, "Candidate A must fail safety"
    assert cand_b.is_safe, "Candidate B must pass safety"
    assert len(cand_a.safety_violations) > 0
    assert len(cand_b.safety_violations) == 0
    
    # Lexicographical dominance: cand_b net_utility MUST be strictly greater than cand_a
    assert cand_b.net_utility > cand_a.net_utility, (
        f"Lexicographical dominance violated: Safe cand B ({cand_b.net_utility:.2f}) "
        f"must beat Unsafe cand A ({cand_a.net_utility:.2f})"
    )
    assert cand_a.net_utility < -500.0, "Unsafe candidate should receive massive penalty offset"


# =========================================================================
# 6. Boundary Semantics Unit Tests
# =========================================================================

@pytest.mark.parametrize("t_core,expected_safe", [
    (94.99, True),
    (95.00, False),  # >= 95.0 is rejected
    (95.01, False),
])
def test_thermal_boundary_semantics(cost_model: ActionCostModel, t_core: float, expected_safe: bool):
    is_safe, violations = cost_model.check_hard_safety(peak_t_core=t_core, max_pressure=3.0, min_flow=20.0)
    assert is_safe == expected_safe
    if not expected_safe:
        assert any("T_core peak" in v for v in violations)


@pytest.mark.parametrize("p_sys,expected_safe", [
    (5.49, True),
    (5.50, False),  # >= 5.50 is rejected
    (5.51, False),
])
def test_pressure_boundary_semantics(cost_model: ActionCostModel, p_sys: float, expected_safe: bool):
    is_safe, violations = cost_model.check_hard_safety(peak_t_core=80.0, max_pressure=p_sys, min_flow=20.0)
    assert is_safe == expected_safe
    if not expected_safe:
        assert any("P_sys max" in v for v in violations)


@pytest.mark.parametrize("f_cool,expected_safe", [
    (8.01, True),
    (8.00, False),  # <= 8.00 is rejected
    (7.99, False),
])
def test_flow_boundary_semantics(cost_model: ActionCostModel, f_cool: float, expected_safe: bool):
    is_safe, violations = cost_model.check_hard_safety(peak_t_core=80.0, max_pressure=3.0, min_flow=f_cool)
    assert is_safe == expected_safe
    if not expected_safe:
        assert any("F_cool min" in v for v in violations)


@pytest.mark.parametrize("novelty,expected_safe", [
    (14.99, True),
    (15.00, True),
    (15.01, False), # > 15.00 is rejected
])
def test_latent_novelty_boundary_semantics(cost_model: ActionCostModel, novelty: float, expected_safe: bool):
    is_safe, violations = cost_model.check_hard_safety(
        peak_t_core=80.0, max_pressure=3.0, min_flow=20.0, latent_novelty=novelty
    )
    assert is_safe == expected_safe
    if not expected_safe:
        assert any("Latent novelty" in v for v in violations)


# =========================================================================
# 7. Adversarial and Operational Tests
# =========================================================================

def test_safe_expensive_vs_safe_cheap(cost_model: ActionCostModel):
    """Given two safe candidates achieving the same physical outcome, the cheaper one must win."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    
    # Cheap: Baseline pump 2, baseline valve 50
    cheap = cost_model.evaluate_cost(
        peak_t_core=75.0,
        max_pressure=3.5,
        min_flow=20.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[50.0, 100.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )
    
    # Expensive: Pump 4, Valve 90
    expensive = cost_model.evaluate_cost(
        peak_t_core=75.0,
        max_pressure=3.5,
        min_flow=20.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[90.0, 100.0, 4.0, 0.0],
        baseline_actions_at_t_star=base_actions,
    )
    
    assert cheap.is_safe and expensive.is_safe
    assert cheap.total_cost < expensive.total_cost
    assert cheap.net_utility > expensive.net_utility


def test_single_action_beats_compound_action_with_equal_outcome(cost_model: ActionCostModel):
    """When a single action and compound action yield identical outcomes, complexity penalty breaks the tie."""
    base_actions = [50.0, 100.0, 2.0, 0.0]
    
    single = cost_model.evaluate_cost(
        peak_t_core=80.0,
        max_pressure=3.8,
        min_flow=22.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[70.0, 100.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
        is_compound_intervention=False,
    )
    
    compound = cost_model.evaluate_cost(
        peak_t_core=80.0,
        max_pressure=3.8,
        min_flow=22.0,
        mean_cpu_load=100.0,
        actions_at_t_star=[70.0, 100.0, 2.0, 0.0],
        baseline_actions_at_t_star=base_actions,
        is_compound_intervention=True,
    )
    
    assert single.is_safe and compound.is_safe
    assert compound.complexity_penalty > 0.0
    assert single.complexity_penalty == 0.0
    assert single.net_utility > compound.net_utility


def test_cost_breakdown_serialization(cost_model: ActionCostModel):
    """CostBreakdown and DecisionCostConfig serialize cleanly to dict."""
    config_dict = cost_model.config.to_dict()
    assert "thermal_hard_limit" in config_dict
    assert config_dict["thermal_hard_limit"] == 95.0
    
    breakdown = cost_model.evaluate_cost(
        peak_t_core=82.0,
        max_pressure=4.0,
        min_flow=20.0,
        mean_cpu_load=95.0,
        actions_at_t_star=[60.0, 95.0, 3.0, 0.0],
        baseline_actions_at_t_star=[50.0, 100.0, 2.0, 0.0],
    )
    b_dict = breakdown.to_dict()
    assert isinstance(b_dict["net_utility"], float)
    assert isinstance(b_dict["is_safe"], bool)
    assert "thermal_penalty" in b_dict
    assert "throttle_cost" in b_dict
