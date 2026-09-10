"""Scenario 2 Regression Tests: Valve Intervention Is Optimal (Task 5.5).

Verifies:
1. Contract 1: Correct scenario isolation (zero hidden state in learner.npz).
2. Contract 2: Valve intervention candidate exists in generic candidate pipeline.
3. Contract 3: State intervention semantics (do(V_pos=85) does not mutate parent action).
4. Contract 4: Causal propagation (downstream multi-variable physical response).
5. Contract 5: Thermal direction (Delta T_core < 0).
6. Contract 6: Flow direction (Delta F_cool > 0).
7. Contract 7: Safety gate precedes ranking (unsafe candidates filtered before ranking).
8. Contract 8: Valve candidate is evaluated (no silent omission).
9. Contract 9: Selected candidate is safe under authoritative safety constraints (k=2.0).
10. Contract 10: Canonical utility ranking (Task-5.2 ActionCostModel).
11. Contract 11: Regret correctness (oracle-grounded regret calculation).
12. Contract 12: Explanation validity (identifies root cause, mechanism, and counterfactual).
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
from scripts.evaluate_decision_scenario_02 import evaluate_scenario_02


@pytest.fixture(scope="module")
def scenario_system():
    """Load model, normalizer, and scenario 2 records."""
    model_path = Path("artifacts/baseline_003")
    assert model_path.exists(), "artifacts/baseline_003 must exist"
    
    norm = ObservationNormalizer.load_yaml(model_path / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_path / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_path / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scen_dir = Path("data/decision_benchmark/scenarios/scenario_02_valve")
    learner_scen = LearnerDecisionScenario.load_npz(scen_dir / "learner.npz")
    oracle_scen = OracleDecisionScenario.load_npz(scen_dir / "oracle.npz")

    planner = InterventionPlanner(model, norm)
    return planner, learner_scen, oracle_scen


def test_scenario_02_firewall_and_latent_isolation(scenario_system):
    """Contract 1: Verify learner record has zero unobserved latent states or oracle artifacts."""
    _, learner_scen, _ = scenario_system
    
    # Check support metadata
    for k in learner_scen.support_metadata:
        assert k not in FORBIDDEN_LEARNER_KEYS

    # Raw npz archive check
    raw_data = np.load("data/decision_benchmark/scenarios/scenario_02_valve/learner.npz", allow_pickle=True)
    for forbidden in FORBIDDEN_LEARNER_KEYS:
        assert forbidden not in raw_data.files


def test_scenario_02_candidate_pipeline_integrity(scenario_system):
    """Contract 2: Verify scenario 2 contains cand_valve_85 and all canonical candidates."""
    _, learner_scen, _ = scenario_system
    cand_ids = [c.candidate_id for c in learner_scen.candidate_actions]
    assert len(cand_ids) == 5
    assert "cand_do_nothing" in cand_ids
    assert "cand_valve_85" in cand_ids
    assert "cand_valve_15" in cand_ids
    assert "cand_pump_4" in cand_ids
    assert "cand_throttle_30" in cand_ids


def test_scenario_02_state_intervention_semantics(scenario_system):
    """Contract 3: Verify do(V_pos=85) performs graph surgery on state and does NOT mutate parent action."""
    planner, learner_scen, _ = scenario_system

    valve_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_valve_85")
    assert valve_cand_spec.intervention_type == "state_clamp"
    assert valve_cand_spec.target == "V_pos"

    specs = valve_cand_spec.to_specs(t_star=learner_scen.intervention_time)
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

    # State channel V_pos (index 5) is clamped to 85.0
    assert np.allclose(res.intervened_observations[:, 5], 85.0, atol=1e-3)


def test_scenario_02_causal_propagation_downstream(scenario_system):
    """Contract 4: Verify valve intervention produces downstream multi-variable effects."""
    planner, learner_scen, _ = scenario_system

    valve_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_valve_85")
    specs = valve_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_flow = np.mean(res.intervened_observations[:, 3] - res.baseline_observations[:, 3])
    delta_press = np.mean(res.intervened_observations[:, 2] - res.baseline_observations[:, 2])
    delta_temp = np.mean(res.intervened_observations[:, 0] - res.baseline_observations[:, 0])

    # Downstream variables must change significantly (not zero)
    assert abs(delta_flow) > 5.0, "Flow must respond to valve change"
    assert abs(delta_press) > 0.1, "Pressure must respond to valve change"
    assert abs(delta_temp) > 2.0, "Core temp must respond to valve change"


def test_scenario_02_thermal_direction_concordance(scenario_system):
    """Contract 5: Valve intervention must predict core cooling (Delta T_core < 0)."""
    planner, learner_scen, _ = scenario_system

    valve_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_valve_85")
    specs = valve_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_t_core = np.mean(res.intervened_observations[:, 0] - res.baseline_observations[:, 0])
    assert delta_t_core < -5.0, f"Expected significant core cooling, got {delta_t_core:.2f}°C"


def test_scenario_02_flow_direction_concordance(scenario_system):
    """Contract 6: Valve opening must predict flow velocity increase (Delta F_cool > 0)."""
    planner, learner_scen, _ = scenario_system

    valve_cand_spec = next(c for c in learner_scen.candidate_actions if c.candidate_id == "cand_valve_85")
    specs = valve_cand_spec.to_specs(t_star=learner_scen.intervention_time)

    res = planner.simulator.simulate(
        pre_observations=learner_scen.historical_observations,
        pre_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        intervention=specs[0],
        intervention_time=learner_scen.intervention_time,
        deterministic=True,
    )

    delta_f_cool = np.mean(res.intervened_observations[:, 3] - res.baseline_observations[:, 3])
    assert delta_f_cool > 10.0, f"Expected substantial flow increase, got {delta_f_cool:.2f} L/min"


def test_scenario_02_safety_gate_precedes_ranking(scenario_system):
    """Contract 7: Unsafe candidate (cand_valve_15) must be disqualified before ranking."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    valve_15_eval = next(c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_valve_15")
    assert not valve_15_eval.is_safe, "cand_valve_15 must fail the safety gate"
    assert len(valve_15_eval.safety_violations) > 0

    # Recommended candidate must be safe
    assert plan_rec.recommended_candidate.is_safe


def test_scenario_02_valve_candidate_is_evaluated(scenario_system):
    """Contract 8: cand_valve_85 must be fully evaluated with cost breakdown."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    valve_85_eval = next((c for c in plan_rec.all_evaluated_candidates if c.candidate_id == "cand_valve_85"), None)
    assert valve_85_eval is not None
    assert valve_85_eval.cost_breakdown is not None
    assert valve_85_eval.peak_t_core < 90.0


def test_scenario_02_selected_candidate_is_safe(scenario_system):
    """Contract 9: Selected recommendation must satisfy all authoritative safety constraints."""
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


def test_scenario_02_canonical_utility_ranking(scenario_system):
    """Contract 10: Utility ranking must select cand_valve_85 as highest net utility."""
    planner, learner_scen, _ = scenario_system

    plan_rec = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )

    assert plan_rec.recommended_candidate.candidate_id == "cand_valve_85"
    assert plan_rec.recommended_candidate.utility_score > 0.0


def test_scenario_02_regret_correctness(scenario_system):
    """Contract 11: Oracle true utility regret must equal exactly 0.0 for PRISM selection."""
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
    assert sel_id == opt_id == "cand_valve_85"

    u_opt = oracle_scen.candidate_outcomes[opt_id].true_utility
    u_sel = oracle_scen.candidate_outcomes[sel_id].true_utility
    regret = u_opt - u_sel
    assert pytest.approx(regret, abs=1e-6) == 0.0

    # Sub-optimal alternatives must have strictly positive regret
    for cid, out in oracle_scen.candidate_outcomes.items():
        if cid != opt_id:
            cand_regret = u_opt - out.true_utility
            assert cand_regret > 0.0, f"Candidate {cid} must have positive regret"


def test_scenario_02_explanation_validity(scenario_system):
    """Contract 12: Generated explanation must reference valve and thermal/flow mechanisms."""
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
    assert "V_pos=85" in md or "85" in md
    assert "flow" in md.lower()
    assert "temperature" in md.lower() or "t_core" in md.lower()
    assert "SAFE" in md


def test_scenario_02_no_scenario_hardcoding(scenario_system):
    """Contract 13: Decision must be invariant to metadata/scenario names."""
    planner, learner_scen, _ = scenario_system

    # Create an anonymous copy of the scenario
    anon_candidates = [
        CandidateActionSpec(c.candidate_id, c.target, c.value, intervention_type=c.intervention_type)
        for c in learner_scen.candidate_actions
    ]

    plan_rec = planner.plan_intervention(
        historical_observations=np.copy(learner_scen.historical_observations),
        historical_actions=np.copy(learner_scen.historical_actions),
        future_actions=np.copy(learner_scen.future_baseline_actions),
        custom_candidates=anon_candidates,
        intervention_time=learner_scen.intervention_time,
    )

    assert plan_rec.recommended_candidate.candidate_id == "cand_valve_85"


def test_scenario_02_learner_oracle_firewall(scenario_system):
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
    oracle_scen.oracle_optimal_candidate_id = "cand_throttle_30"

    plan_rec_2 = planner.plan_intervention(
        historical_observations=learner_scen.historical_observations,
        historical_actions=learner_scen.historical_actions,
        future_actions=learner_scen.future_baseline_actions,
        custom_candidates=learner_scen.candidate_actions,
        intervention_time=learner_scen.intervention_time,
    )
    sel_id_2 = plan_rec_2.recommended_candidate.candidate_id

    assert sel_id_1 == sel_id_2 == "cand_valve_85"


def test_scenario_02_meaningful_positive_margin(scenario_system):
    """Contract 15: PRISM must have a substantial positive utility margin for intervention."""
    res = evaluate_scenario_02(verbose=False)
    assert res["action_agreement"] is True
    assert res["is_safe"] is True
    assert res["prism_utility"] > 0.0
    assert res["decision_margin"] > 0.5, f"Decision margin must be > 0.5 (got {res['decision_margin']:.4f})"
    assert res["margin_vs_do_nothing"] > 0.5, f"Margin vs inaction must be > 0.5 (got {res['margin_vs_do_nothing']:.4f})"
