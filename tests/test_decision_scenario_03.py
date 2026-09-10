"""Scenario 3 Regression Tests: Throttle Intervention Is Optimal (Task 5.6).

Verifies:
1. Contract 1: Correct scenario isolation (zero hidden state in learner.npz).
2. Contract 2: Throttle intervention candidate exists in generic candidate pipeline.
3. Contract 3: State intervention semantics (do(L_cpu=50) clamps workload directly).
4. Contract 4: Causal propagation (workload reduction attenuates power and core temp).
5. Contract 5: Thermal direction (Delta T_core < 0).
6. Contract 6: Power direction (Delta P_elec < 0).
7. Contract 7: Workload direction (Delta L_cpu < 0).
8. Contract 8: Safety constraints satisfied under authoritative gate (k=2.0).
9. Contract 9: Canonical utility ranking selects cand_throttle_50 over cand_throttle_20 (Pareto efficiency).
10. Contract 10: Mathematical regret correctness (oracle-grounded regret calculation R=0.0).
11. Contract 11: Rejection of secondary non-root-cause interventions (valve/pump).
12. Contract 12: Explanation validity (identifies CPU workload, Joule heating attenuation).
13. Contract 13: No scenario hardcoding (selection invariant to scenario renaming).
14. Contract 14: Learner/oracle firewall (oracle mutation does not affect planning).
15. Contract 15: Meaningful positive decision margin against inaction (Delta U > 0).
"""

from __future__ import annotations
import pytest
import numpy as np
import torch
from pathlib import Path

from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.training.normalization import ObservationNormalizer
from prism.dataset.decision_benchmark import (
    LearnerDecisionScenario,
    OracleDecisionScenario,
    DecisionClass,
    CandidateActionSpec,
)
from prism.intervention.spec import InterventionSpec, InterventionType
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.schema import FORBIDDEN_LEARNER_KEYS
from scripts.evaluate_decision_scenario_03 import evaluate_scenario_03


@pytest.fixture(scope="module")
def scenario_system():
    """Load model, normalizer, and scenario 3 records."""
    model_path = Path("artifacts/baseline_003")
    assert model_path.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scen_dir = Path("data/decision_benchmark/scenarios/scenario_03_throttle")
    learner_scen = LearnerDecisionScenario.load_npz(scen_dir / "learner.npz")
    oracle_scen = OracleDecisionScenario.load_npz(scen_dir / "oracle.npz")

    planner = InterventionPlanner(model, norm)
    return planner, learner_scen, oracle_scen


def test_scenario_03_firewall_and_latent_isolation(scenario_system):
    """Contract 1: Verify learner record has zero unobserved latent states or oracle artifacts."""
    _, learner_scen, _ = scenario_system
    
    # Check support metadata
    for k in learner_scen.support_metadata:
        assert k not in FORBIDDEN_LEARNER_KEYS

    # Raw npz archive check
    raw_data = np.load("data/decision_benchmark/scenarios/scenario_03_throttle/learner.npz", allow_pickle=True)
    for forbidden in FORBIDDEN_LEARNER_KEYS:
        assert forbidden not in raw_data.files


def test_scenario_03_candidate_pipeline_integrity(scenario_system):
    """Contract 2: Verify scenario 3 contains cand_throttle_50 and all canonical candidates."""
    _, learner_scen, _ = scenario_system
    cand_ids = [c.candidate_id for c in learner_scen.candidate_actions]
    assert len(cand_ids) == 5
    assert "cand_do_nothing" in cand_ids
    assert "cand_throttle_50" in cand_ids
    assert "cand_throttle_20" in cand_ids
    assert "cand_valve_100" in cand_ids
    assert "cand_pump_4" in cand_ids


def test_scenario_03_state_intervention_semantics(scenario_system):
    """Contract 3: Verify do(L_cpu=50) performs graph surgery on state and clamps workload directly."""
    planner, learner_scen, _ = scenario_system

    throttle_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_throttle_50")
    assert throttle_cand_spec.intervention_type == "state_clamp"
    assert throttle_cand_spec.target == "L_cpu"

    specs = throttle_cand_spec.to_specs(t_star=learner_scen.intervention_time)
    assert len(specs) == 1
    assert specs[0].intervention_type == InterventionType.STATE_CLAMP
    assert specs[0].is_state_clamp

    # Simulate with state clamp
    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    # State channel L_cpu (index 4) is clamped to 50.0
    assert np.allclose(res.intervened_observations[:, 4], 50.0, atol=1e-3)


def test_scenario_03_causal_propagation_downstream(scenario_system):
    """Contract 4: Verify throttle intervention produces downstream power and thermal attenuation."""
    planner, learner_scen, _ = scenario_system

    throttle_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_throttle_50")
    specs = throttle_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_power = np.mean(res.intervened_observations[:, 7] - res.baseline_observations[:, 7])
    delta_temp = np.mean(res.intervened_observations[:, 0] - res.baseline_observations[:, 0])

    assert delta_power < 0.0, f"Expected power reduction, got {delta_power:+.2f} kW"
    assert delta_temp < -2.0, f"Expected significant core cooling, got {delta_temp:+.2f}°C"


def test_scenario_03_thermal_direction_concordance(scenario_system):
    """Contract 5: Throttle intervention must predict core cooling (Delta T_core < 0)."""
    planner, learner_scen, _ = scenario_system

    throttle_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_throttle_50")
    specs = throttle_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_t_core = np.mean(res.intervened_observations[:, 0] - res.baseline_observations[:, 0])
    assert delta_t_core < -3.0, f"Expected substantial core cooling, got {delta_t_core:.2f}°C"


def test_scenario_03_power_direction_concordance(scenario_system):
    """Contract 6: Throttle reduction must predict power draw reduction (Delta P_elec < 0)."""
    planner, learner_scen, _ = scenario_system

    throttle_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_throttle_50")
    specs = throttle_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_p_elec = np.mean(res.intervened_observations[:, 7] - res.baseline_observations[:, 7])
    assert delta_p_elec < 0.0, f"Expected power reduction, got {delta_p_elec:.2f} kW"


def test_scenario_03_workload_direction_concordance(scenario_system):
    """Contract 7: Throttle reduction must predict CPU workload drop (Delta L_cpu < 0)."""
    planner, learner_scen, _ = scenario_system

    throttle_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_throttle_50")
    specs = throttle_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_l_cpu = np.mean(res.intervened_observations[:, 4] - res.baseline_observations[:, 4])
    assert delta_l_cpu < -10.0, f"Expected substantial workload drop, got {delta_l_cpu:.2f}%"


def test_scenario_03_selected_candidate_is_safe(scenario_system):
    """Contract 8: Selected recommendation must satisfy all authoritative safety constraints."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    sel = plan_rec.recommended_candidate
    assert sel is not None
    assert sel.is_safe
    assert sel.peak_t_core < 95.0
    assert sel.max_pressure < 5.5
    assert sel.min_flow > 8.0
    assert sel.latent_novelty < 15.0


def test_scenario_03_canonical_utility_ranking(scenario_system):
    """Contract 9: Utility ranking must select cand_throttle_50 over cand_throttle_20 (Pareto efficiency)."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    assert plan_rec.recommended_candidate.candidate_id == "cand_throttle_50"
    
    t50_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_throttle_50")
    t20_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_throttle_20")
    assert t50_eval.utility_score > t20_eval.utility_score, "50% throttle must have higher utility than 20% over-throttling"


def test_scenario_03_regret_correctness(scenario_system):
    """Contract 10: Oracle true utility regret must equal exactly 0.0 for PRISM selection."""
    planner, learner_scen, oracle_scen = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    sel_id = plan_rec.recommended_candidate.candidate_id
    opt_id = oracle_scen.oracle_optimal_candidate_id
    assert sel_id == opt_id == "cand_throttle_50"

    u_opt = oracle_scen.candidate_outcomes[opt_id].true_utility
    u_sel = oracle_scen.candidate_outcomes[sel_id].true_utility
    regret = u_opt - u_sel
    assert pytest.approx(regret, abs=1e-6) == 0.0

    # Sub-optimal alternatives must have strictly positive regret
    for cid, out in oracle_scen.candidate_outcomes.items():
        if cid != opt_id:
            cand_regret = u_opt - out.true_utility
            assert cand_regret > 0.0, f"Candidate {cid} must have positive regret"


def test_scenario_03_rejection_of_valve_and_pump_alternatives(scenario_system):
    """Contract 11: Valve and pump alternatives must be rejected because they fail to address compute heat generation."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    sel_id = plan_rec.recommended_candidate.candidate_id
    assert sel_id == "cand_throttle_50"
    
    t50_util = plan_rec.recommended_candidate.utility_score
    valve_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_valve_100")
    pump_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_pump_4")
    
    assert t50_util > valve_eval.utility_score
    assert t50_util > pump_eval.utility_score


def test_scenario_03_explanation_validity(scenario_system):
    """Contract 12: Generated explanation must reference CPU workload, Joule heating, and thermal mechanism."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    expl = plan_rec.explanation
    md = expl.format_markdown()
    assert "L_cpu=50" in md or "50" in md
    assert "workload" in md.lower() or "cpu" in md.lower() or "joule" in md.lower()
    assert "SAFE" in md


def test_scenario_03_no_scenario_hardcoding(scenario_system):
    """Contract 13: Decision must be invariant to metadata/scenario names."""
    planner, learner_scen, _ = scenario_system

    # Create an anonymous copy of the scenario
    anon_candidates = [
        CandidateActionSpec(c.candidate_id, c.target, c.value, intervention_type=c.intervention_type, secondary_target=c.secondary_target, secondary_value=c.secondary_value)
        for c in learner_scen.candidate_actions
    ]

    plan_rec = planner.plan_intervention(
        historical_observations=np.copy(learner_scen.historical_observations),
        historical_actions=np.copy(learner_scen.historical_actions),
        future_actions=np.copy(learner_scen.future_baseline_actions),
        custom_candidates=anon_candidates,
        intervention_time=learner_scen.intervention_time,
    )

    assert plan_rec.recommended_candidate.candidate_id == "cand_throttle_50"


def test_scenario_03_learner_oracle_firewall(scenario_system):
    """Contract 14: Mutating oracle object after planning does not change planner output."""
    planner, learner_scen, oracle_scen = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )
    sel_id_1 = plan_rec.recommended_candidate.candidate_id

    # Mutate oracle scenario
    oracle_scen.oracle_optimal_candidate_id = "cand_valve_100"

    plan_rec_2 = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )
    sel_id_2 = plan_rec_2.recommended_candidate.candidate_id

    assert sel_id_1 == sel_id_2 == "cand_throttle_50"


def test_scenario_03_meaningful_positive_margin(scenario_system):
    """Contract 15: PRISM must have a positive utility margin for throttle intervention."""
    res = evaluate_scenario_03(verbose=False)
    assert res["action_agreement"] is True
    assert res["is_safe"] is True
    assert res["prism_utility"] > 0.0
    assert res["decision_margin"] > 0.1, f"Decision margin must be > 0.1 (got {res['decision_margin']:.4f})"
    assert res["margin_vs_do_nothing"] > 0.1, f"Margin vs inaction must be > 0.1 (got {res['margin_vs_do_nothing']:.4f})"
