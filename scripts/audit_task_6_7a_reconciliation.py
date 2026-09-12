"""Forensic Audit Script for Task 6.7A: Benchmark/Evidence Reconciliation.

Inspects all candidates across scenarios S1 through S6:
- Learner candidate specifications
- Model raw forecast vs k=2 uncertainty-adjusted effective values
- Planner CandidateEvaluation vs SafetyEvidenceDetail
- Oracle candidate outcomes and true safety/utility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator
from prism.planning.planner import InterventionPlanner
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario
from prism.explanation.evidence import build_decision_evidence
from prism.explanation.safety_evidence import build_safety_evidence_from_decision_evidence
from prism.explanation.abstention_explanation import build_abstention_explanation_from_decision_evidence
from prism.explanation.unified_record import build_unified_decision_record


def run_forensic_reconciliation():
    model_dir = Path("artifacts/baseline_005")
    norm = ObservationNormalizer.load_yaml(model_dir / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(model_dir / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(model_dir / "trust_calibration.json") as f:
        calib = json.load(f)

    mu_id = np.array(calib["mu_id"], dtype=np.float32)
    inv_cov_id = np.array(calib["inv_cov_id"], dtype=np.float32)

    trust_evaluator = ModelTrustEvaluator(
        model=model,
        normalizer=norm,
        mu_id=mu_id,
        inv_cov_id=inv_cov_id,
        tau_residual_t=calib["tau_residual_t_core"],
        tau_residual_8d=calib["tau_residual_8d_norm"],
        tau_novelty=calib["tau_novelty_mahalanobis"],
    )

    planner = InterventionPlanner(model, norm)
    bpath = Path("data/decision_benchmark/scenarios")

    scenarios = [
        "scenario_01_do_nothing",
        "scenario_02_valve",
        "scenario_03_throttle",
        "scenario_04_pump",
        "scenario_05_combined",
        "scenario_06_all_unsafe",
    ]

    for s_id in scenarios:
        sdir = bpath / s_id
        learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")
        oracle = OracleDecisionScenario.load_npz(sdir / "oracle.npz")

        diag = trust_evaluator.evaluate_trust(
            historical_obs=learner.historical_observations,
            historical_mask=learner.historical_observation_mask,
            historical_act=learner.historical_actions,
            t_star=learner.intervention_time,
        )

        rec = planner.plan_intervention(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
            trust_evaluator=trust_evaluator,
        )

        print(f"\n{'='*80}")
        print(f"SCENARIO: {s_id}")
        print(f"Trust State: {diag.state.name} (R_T={diag.residual_t_core:.2f}°C, R_8D={diag.residual_8d_norm:.2f}, D_lat={diag.latent_mahalanobis_d:.2f})")
        print(f"Oracle Optimal Candidate: {oracle.oracle_optimal_candidate_id}")
        
        orc_opt = oracle.candidate_outcomes.get(oracle.oracle_optimal_candidate_id)
        if orc_opt:
            print(f"  Oracle Optimal True Outcome: is_safe={orc_opt.is_safe}, true_util={orc_opt.true_utility:.4f}, peak_T={orc_opt.peak_t_core:.2f}, max_P={orc_opt.max_pressure:.2f}, min_F={orc_opt.min_flow:.2f}")

        rec_cand = rec.recommended_candidate
        rec_id = rec_cand.candidate_id if rec_cand else "NONE"
        print(f"Planner Recommended: {rec_id}")

        print("\nAll Evaluated Candidates by Planner:")
        for c in rec.all_evaluated_candidates:
            orc_c = oracle.candidate_outcomes.get(c.candidate_id)
            orc_safe = orc_c.is_safe if orc_c else None
            orc_util = orc_c.true_utility if orc_c else None
            orc_peak_t = orc_c.peak_t_core if orc_c else None
            print(f"  - {c.candidate_id:25s} | PlannerSafe={str(c.is_safe):5s} | Util={c.utility_score:8.4f} | PeakT={c.peak_t_core:6.2f}°C | MaxP={c.max_pressure:5.2f} | MinF={c.min_flow:5.2f} | Nov={c.latent_novelty:5.2f} | Viols={c.safety_violations} || OracleSafe={str(orc_safe):5s}, OracleUtil={str(orc_util)}")

        # Now build Phase 6 evidence
        dec_ev = build_decision_evidence(rec, diag, scenario_id=s_id, timestamp_utc="2026-09-11T12:00:00Z")
        saf_detail = build_safety_evidence_from_decision_evidence(dec_ev)
        record = build_unified_decision_record(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")

        print("\nPhase 6 Unified Record Projection:")
        print(f"  Decision Status: {record.decision.decision_status}")
        print(f"  Recommendation: {record.decision.recommendation}")
        print(f"  Safety is_safe: {record.safety.is_safe}")
        print(f"  Safety overall_state: {record.safety.overall_state}")
        print(f"  Limiting Constraint: {record.safety.limiting_constraint.constraint_name} (margin={record.safety.limiting_constraint.raw_margin})")
        if record.safety.violations:
            print(f"  Violations: {[v.constraint_code + ' (' + v.message + ')' for v in record.safety.violations]}")


if __name__ == "__main__":
    run_forensic_reconciliation()
