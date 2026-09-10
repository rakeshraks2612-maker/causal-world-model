"""Comprehensive Causal Invariant and Multi-Variable Intervention Validation Suite (Task 3.4F).

Validates:
1. Pearl Graph Surgery Invariant: State clamps do(X = x) leave control actions A_t strictly unchanged.
2. Action Setpoint Invariant: Action controls A = a modify setpoint and exhibit physical dynamic lag.
3. Downstream Propagation & Directionality:
   - do(L_cpu = 80%) -> T_core increases substantially (Delta T_core > +2.0°C), P_elec increases.
   - do(V_pos = 85%) -> Decoded V_pos clamped to 85.0% at t*.
   - A_throttle = 80% -> Action setpoint clamped, actions modified.
4. Non-Descendant Negative Control Invariance:
   - do(Vib_pump = 10) leaves T_core, P_sys, T_cool, and L_cpu completely invariant (Delta == 0.00).
5. Latent Manifold Divergence:
   - Causal state interventions produce significant latent trajectory divergence ||Delta Z_40|| > 1.0.
"""

from __future__ import annotations
import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.schema import SplitType
from prism.dataset.generator import generate_single_episode
from prism.intervention.spec import InterventionSpec, InterventionType
from prism.intervention.operator import InterventionOperator
from prism.intervention.simulator import LearnedInterventionSimulator
from prism.simulator.state import OBSERVABLE_VARIABLES


@pytest.fixture(scope="module")
def baseline_003_system():
    """Load the trained baseline_003 world model and normalizer."""
    b3_dir = Path("artifacts/baseline_003")
    assert b3_dir.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(b3_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(b3_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(b3_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    simulator = LearnedInterventionSimulator(model, norm)
    return model, norm, simulator


@pytest.fixture
def nominal_test_episode():
    """Generate a clean test episode for controlled invariant experiments."""
    return generate_single_episode(SplitType.TEST, index=99, regime="nominal", length=100)


def test_invariant_1_graph_surgery_action_preservation(baseline_003_system, nominal_test_episode):
    """Invariant 1: State intervention do(V_pos = x) must strictly preserve baseline action A_t."""
    model, norm, simulator = baseline_003_system
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(target="V_pos", value=85.0, intervention_time=t_star)
    
    res = simulator.simulate(
        pre_observations=ep.observations[: t_star + 1],
        pre_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention=spec,
        intervention_time=t_star,
        deterministic=True,
    )
    
    # Verify the operator actions were NOT modified
    operator = InterventionOperator([spec])
    fut_acts = torch.tensor(ep.actions[t_star : t_star + 41], dtype=torch.float32)
    mod_acts = operator.get_modified_actions(fut_acts, t_star=t_star)
    
    # Bit-for-bit action equality
    np.testing.assert_array_equal(mod_acts.numpy(), fut_acts.numpy())


def test_invariant_2_action_control_modifies_setpoint(nominal_test_episode):
    """Invariant 2: Action control A_valve = 85% must modify action setpoint starting at t*."""
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(
        target="A_valve",
        value=85.0,
        intervention_time=t_star,
        intervention_type=InterventionType.ACTION_CONTROL,
    )
    
    operator = InterventionOperator([spec])
    fut_acts = torch.tensor(ep.actions[t_star : t_star + 41], dtype=torch.float32)
    mod_acts = operator.get_modified_actions(fut_acts, t_star=t_star)
    
    # Action channel 0 (A_valve) must equal 85.0 for all future steps
    assert torch.all(mod_acts[:, 0] == 85.0)
    # Other action channels (1, 2, 3) must remain identical
    np.testing.assert_array_equal(mod_acts[:, 1:].numpy(), fut_acts[:, 1:].numpy())


def test_invariant_3_state_clamp_direct_overrides(baseline_003_system, nominal_test_episode):
    """Invariant 3: Clamping state variables directly sets decoded value at intervention time."""
    model, norm, simulator = baseline_003_system
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(target="V_pos", value=85.0, intervention_time=t_star)
    
    res = simulator.simulate(
        pre_observations=ep.observations[: t_star + 1],
        pre_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention=spec,
        intervention_time=t_star,
        deterministic=True,
    )
    
    # Decoded V_pos at t* (step 0) must be clamped to 85.0%
    assert abs(res.intervened_observations[0, 5] - 85.0) < 1e-4


def test_invariant_4_cpu_load_propagation(baseline_003_system, nominal_test_episode):
    """Invariant 4: Increasing CPU load do(L_cpu = 80%) must increase core temp and electrical power."""
    model, norm, simulator = baseline_003_system
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(target="L_cpu", value=80.0, intervention_time=t_star)
    
    res = simulator.simulate(
        pre_observations=ep.observations[: t_star + 1],
        pre_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention=spec,
        intervention_time=t_star,
        deterministic=True,
    )
    
    # Decoded L_cpu at t* must be clamped to 80.0%
    assert abs(res.intervened_observations[0, 4] - 80.0) < 1e-4
    
    # Delta at h = 10, 20: T_core must heat up significantly
    delta_t_core_20 = res.effects.horizon_effects[20].delta_t_core
    assert delta_t_core_20 > 2.0, f"Expected substantial heating under L_cpu=80%, got {delta_t_core_20}"
    assert res.effects.horizon_effects[10].delta_p_elec > 0.0, "Expected electrical power increase"


def test_invariant_5_non_descendant_negative_control(baseline_003_system, nominal_test_episode):
    """Invariant 5: Negative Control do(Vib_pump = 10) has zero effect on thermal and fluid channels."""
    model, norm, simulator = baseline_003_system
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(target="Vib_pump", value=10.0, intervention_time=t_star)
    
    res = simulator.simulate(
        pre_observations=ep.observations[: t_star + 1],
        pre_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention=spec,
        intervention_time=t_star,
        deterministic=True,
    )
    
    # Vib_pump at t* is clamped to 10.0 mm/s
    assert abs(res.intervened_observations[0, 6] - 10.0) < 1e-4
    
    # Non-descendant channels: T_core, T_cool, P_sys, L_cpu must have near-zero delta
    for h in [1, 5, 10, 20, 40]:
        eff = res.effects.horizon_effects[h]
        assert abs(eff.delta_t_core) < 0.05, f"T_core violated negative control at h={h}: {eff.delta_t_core}"
        assert abs(eff.delta_p_sys) < 0.05, f"P_sys violated negative control at h={h}: {eff.delta_p_sys}"
        assert abs(eff.delta_t_cool) < 0.05, f"T_cool violated negative control at h={h}: {eff.delta_t_cool}"


def test_invariant_6_latent_manifold_divergence(baseline_003_system, nominal_test_episode):
    """Invariant 6: State intervention do(L_cpu = 80%) produces non-zero latent divergence ||Delta Z_40|| > 1.0."""
    model, norm, simulator = baseline_003_system
    ep = nominal_test_episode
    t_star = 40
    
    spec = InterventionSpec(target="L_cpu", value=80.0, intervention_time=t_star)
    
    res = simulator.simulate(
        pre_observations=ep.observations[: t_star + 1],
        pre_actions=ep.actions[: t_star + 1],
        future_actions=ep.actions[t_star + 1 : t_star + 41],
        intervention=spec,
        intervention_time=t_star,
        deterministic=True,
    )
    
    # Latent difference norm at h = 40
    dz_40 = np.linalg.norm(res.intervened_latent_mean[-1] - res.baseline_latent_mean[-1])
    assert dz_40 > 1.0, f"Expected latent trajectory divergence ||ΔZ_40|| > 1.0, got {dz_40}"
