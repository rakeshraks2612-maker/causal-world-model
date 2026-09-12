"""Generate Comprehensive Phase 6 Decision & Evidence Dossiers for Scenarios S1-S6.

Executes the frozen baseline_005 model across all 6 benchmark scenarios,
assembles the authoritative PRISM Decision Records via the pure projection layer,
and outputs machine-readable JSON records and human-auditable Markdown dossiers.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import Dict, Any, List
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator, ModelTrustState
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario

from prism.explanation.evidence import build_decision_evidence, DecisionEvidence
from prism.explanation.causal_explanation import build_causal_explanation
from prism.explanation.counterfactual_evidence import build_counterfactual_evidence
from prism.explanation.safety_evidence import build_safety_evidence_from_decision_evidence
from prism.explanation.abstention_explanation import build_abstention_explanation_from_decision_evidence
from prism.explanation.unified_record import (
    PrismDecisionRecord,
    build_unified_decision_record,
)


def run_phase6_dossier_generation(
    model_dir: str | Path = "artifacts/baseline_005",
    benchmark_dir: str | Path = "data/decision_benchmark",
    output_dir: str | Path = "artifacts/baseline_005/phase6_dossiers",
) -> Dict[str, Any]:
    mpath = Path(model_dir)
    bpath = Path(benchmark_dir)
    outpath = Path(output_dir)
    outpath.mkdir(parents=True, exist_ok=True)

    norm = ObservationNormalizer.load_yaml(mpath / "normalization.yaml")
    cfg = WorldModelConfig.from_yaml(mpath / "config.yaml")
    model = CausalWorldModel(cfg)
    ckpt = torch.load(mpath / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(mpath / "trust_calibration.json") as f:
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

    scenarios = [
        ("scenario_01_do_nothing", "Scenario 1: Inaction Restraint"),
        ("scenario_02_valve", "Scenario 2: Valve Intervention"),
        ("scenario_03_throttle", "Scenario 3: Workload Throttling"),
        ("scenario_04_pump", "Scenario 4: Pump Head Modulating"),
        ("scenario_05_combined", "Scenario 5: Multi-Variable Compound"),
        ("scenario_06_all_unsafe", "Scenario 6: All-Unsafe Abstention"),
    ]

    records: Dict[str, Any] = {}
    summary_table: List[Dict[str, Any]] = []

    for s_id, s_title in scenarios:
        sdir = bpath / "scenarios" / s_id
        learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")
        oracle = OracleDecisionScenario.load_npz(sdir / "oracle.npz")

        # 1. Trust Diagnostic
        diag = trust_evaluator.evaluate_trust(
            historical_obs=learner.historical_observations,
            historical_mask=learner.historical_observation_mask,
            historical_act=learner.historical_actions,
            t_star=learner.intervention_time,
        )

        # 2. Plan Recommendation
        t0 = time.perf_counter()
        rec: PlanRecommendation = planner.plan_intervention(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
            trust_evaluator=trust_evaluator,
        )
        plan_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Base Decision Evidence
        dec_ev = build_decision_evidence(
            recommendation=rec,
            trust_diagnostic=diag,
            scenario_id=s_id,
            timestamp_utc="2026-09-11T12:00:00Z",
        )

        # 4. Specialized Explanations & Evidence Engines
        causal_expl = build_causal_explanation(dec_ev, timestamp_utc="2026-09-11T12:00:00Z")
        safety_detail = build_safety_evidence_from_decision_evidence(dec_ev)
        abst_expl = build_abstention_explanation_from_decision_evidence(dec_ev)

        # Optional counterfactual evidence
        # In decision benchmarks, counterfactual is Optional; set to None unless twin-world CF result is abduced
        cf_ev = None

        # 5. Assemble Unified PRISM Decision Record
        unified_record = build_unified_decision_record(
            decision_evidence=dec_ev,
            causal_explanation=causal_expl,
            counterfactual_evidence=cf_ev,
            safety_evidence=safety_detail,
            abstention_explanation=abst_expl,
            scenario_id=s_id,
            timestamp_utc="2026-09-11T12:00:00Z",
        )

        # 6. Save JSON and Markdown Dossiers
        json_path = outpath / f"{s_id}_dossier.json"
        md_path = outpath / f"{s_id}_dossier.md"

        with open(json_path, "w") as f:
            f.write(unified_record.to_json(indent=2))

        with open(md_path, "w") as f:
            f.write(unified_record.format_markdown())

        rec_cand_id = rec.recommended_candidate.candidate_id if rec.recommended_candidate else "NONE"
        records[s_id] = {
            "record": unified_record.to_dict(),
            "json_path": str(json_path),
            "md_path": str(md_path),
        }

        summary_table.append({
            "scenario_id": s_id,
            "title": s_title,
            "decision_status": unified_record.decision.decision_status,
            "recommendation": rec_cand_id,
            "trust_state": unified_record.trust.trust_state,
            "is_safe": unified_record.safety.is_safe,
            "overall_safety_state": unified_record.safety.overall_state,
            "limiting_constraint": unified_record.safety.limiting_constraint.constraint_name,
            "limiting_margin": unified_record.safety.limiting_constraint.raw_margin,
            "limiting_unit": unified_record.safety.limiting_constraint.unit,
            "utility_score": unified_record.decision_quality.utility_score,
            "decision_margin": unified_record.decision_quality.decision_margin,
            "unified_record_hash": unified_record.provenance.unified_record_hash,
            "planning_time_ms": plan_ms,
        })

    # Save summary table
    summary_path_json = outpath / "phase6_summary_table.json"
    with open(summary_path_json, "w") as f:
        json.dump(summary_table, f, indent=2)

    # Format Markdown Summary
    summary_md_lines = [
        "# 📊 PRISM Phase 6 Benchmark Evaluation & Evidence Dossier Summary",
        "",
        "| Scenario | Title | Recommendation | Trust State | Safety State | Limiting Constraint | Margin | Utility | SHA-256 Fingerprint |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in summary_table:
        m_str = f"{row['limiting_margin']:+.2f} {row['limiting_unit']}" if row['limiting_margin'] is not None else "N/A"
        u_str = f"{row['utility_score']:.4f}" if row['utility_score'] is not None else "N/A"
        h_short = row['unified_record_hash'][:12] + "..."
        summary_md_lines.append(
            f"| `{row['scenario_id']}` | **{row['title']}** | `{row['recommendation']}` | `{row['trust_state']}` | `{row['overall_safety_state']}` | `{row['limiting_constraint']}` | `{m_str}` | `{u_str}` | `{h_short}` |"
        )
    
    summary_md_lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **Tamper-Evident Integrity:** PRISM produces a deterministic SHA-256 evidence fingerprint that makes post-hoc modification detectable when compared against the recorded provenance hash.",
    ])

    summary_path_md = outpath / "phase6_summary_table.md"
    with open(summary_path_md, "w") as f:
        f.write("\n".join(summary_md_lines))

    print(f"Generated 6 complete dossiers and summary table in {output_dir}")
    return records


if __name__ == "__main__":
    run_phase6_dossier_generation()
