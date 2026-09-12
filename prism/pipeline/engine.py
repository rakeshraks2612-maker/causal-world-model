"""PRISM End-to-End Decision & Evidence Pipeline Engine (Task 7.1).

Provides the unified, production-grade inference engine for PRISM:
1. Ingests raw telemetry or benchmark scenario files.
2. Performs upfront model trust & telemetry consistency audit.
3. Conducts causal intervention planning & Pareto multi-objective ranking.
4. Generates Pearl Level-3 twin-world counterfactual simulations & causal effect deltas.
5. Evaluates uncertainty-adjusted (k=2) physical safety constraints and headroom.
6. Assembles the machine-readable PrismDecisionRecord with deterministic SHA-256 fingerprint.
7. Emits canonical JSON artifacts and human-auditable executive Markdown dossiers.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch

from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.evaluation.model_trust import ModelTrustEvaluator, TrustDiagnostics, ModelTrustState
from prism.planning.planner import InterventionPlanner, PlanRecommendation
from prism.dataset.decision_benchmark import LearnerDecisionScenario, OracleDecisionScenario

from prism.counterfactual.engine import LearnedCounterfactualEngine
from prism.dataset.counterfactuals import LearnerCounterfactualRecord

from prism.explanation.evidence import (
    build_decision_evidence,
    DecisionEvidence,
)
from prism.explanation.causal_explanation import (
    build_causal_explanation,
    CausalExplanation,
)
from prism.explanation.counterfactual_evidence import (
    build_counterfactual_evidence,
    CounterfactualEvidence,
)
from prism.explanation.safety_evidence import (
    build_safety_evidence_from_decision_evidence,
    SafetyEvidenceDetail,
)
from prism.explanation.abstention_explanation import (
    build_abstention_explanation_from_decision_evidence,
    AbstentionExplanation,
)
from prism.explanation.unified_record import (
    PrismDecisionRecord,
    build_unified_decision_record,
)


@dataclass
class PrismPipelineConfig:
    """Configuration for PRISM Pipeline execution."""
    model_dir: Path = Path("artifacts/baseline_005")
    benchmark_dir: Path = Path("data/decision_benchmark")
    model_version: str = "baseline_005"
    dataset_version: str = "v2.1"
    planner_version: str = "v1.2"
    benchmark_version: str = "v1.0"
    uncertainty_multiplier_k: float = 2.0
    planning_horizon: int = 40
    device: str = "cpu"


class PrismPipeline:
    """Production End-to-End Decision & Auditable Evidence Pipeline."""

    def __init__(self, config: Optional[PrismPipelineConfig] = None):
        self.config = config or PrismPipelineConfig()
        self._load_system()

    def _load_system(self) -> None:
        """Load the frozen model, normalizer, trust calibration, and planner."""
        mdir = Path(self.config.model_dir)
        if not (mdir / "best.pt").exists():
            raise FileNotFoundError(f"Model checkpoint not found at {mdir / 'best.pt'}")

        self.normalizer = ObservationNormalizer.load_yaml(mdir / "normalization.yaml")
        self.model_config = WorldModelConfig.from_yaml(mdir / "config.yaml")
        self.model = CausalWorldModel(self.model_config)
        
        ckpt = torch.load(mdir / "best.pt", map_location=self.config.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Load Trust Calibration
        calib_path = mdir / "trust_calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)
            mu_id = np.array(calib["mu_id"], dtype=np.float32)
            inv_cov_id = np.array(calib["inv_cov_id"], dtype=np.float32)
            tau_rt = float(calib["tau_residual_t_core"])
            tau_r8d = float(calib["tau_residual_8d_norm"])
            tau_nov = float(calib["tau_novelty_mahalanobis"])
        else:
            mu_id = np.zeros(32, dtype=np.float32)
            inv_cov_id = np.eye(32, dtype=np.float32)
            tau_rt = 6.00
            tau_r8d = 1.90
            tau_nov = 15.00

        self.trust_evaluator = ModelTrustEvaluator(
            model=self.model,
            normalizer=self.normalizer,
            mu_id=mu_id,
            inv_cov_id=inv_cov_id,
            tau_residual_t=tau_rt,
            tau_residual_8d=tau_r8d,
            tau_novelty=tau_nov,
        )

        self.planner = InterventionPlanner(self.model, self.normalizer)
        self.cf_engine = LearnedCounterfactualEngine(self.model, self.normalizer)

    def run_telemetry(
        self,
        historical_observations: np.ndarray,
        historical_actions: np.ndarray,
        future_baseline_actions: np.ndarray,
        custom_candidates: Optional[List[Any]] = None,
        intervention_time: int = 20,
        historical_mask: Optional[np.ndarray] = None,
        scenario_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PrismDecisionRecord:
        """Execute end-to-end PRISM decision, counterfactual simulation, safety, and evidence projection."""
        t_start = time.perf_counter()
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

        # 1. Upfront Model Trust & Telemetry Integrity Audit
        trust_diag: TrustDiagnostics = self.trust_evaluator.evaluate_trust(
            historical_obs=historical_observations,
            historical_mask=historical_mask,
            historical_act=historical_actions,
            t_star=intervention_time,
        )

        # 2. Causal Intervention Planning with Upfront Trust Gateway
        recommendation: PlanRecommendation = self.planner.plan_intervention(
            historical_observations=historical_observations,
            historical_actions=historical_actions,
            future_actions=future_baseline_actions,
            custom_candidates=custom_candidates,
            intervention_time=intervention_time,
            trust_evaluator=self.trust_evaluator,
        )

        # 3. Base Decision Evidence Contract
        dec_evidence: DecisionEvidence = build_decision_evidence(
            recommendation=recommendation,
            trust_diagnostic=trust_diag,
            scenario_id=scenario_id,
            timestamp_utc=ts,
        )

        # 4. Specialized Explanations & Evidence Engines
        causal_expl: CausalExplanation = build_causal_explanation(dec_evidence, timestamp_utc=ts)
        safety_detail: SafetyEvidenceDetail = build_safety_evidence_from_decision_evidence(
            dec_evidence,
            k_multiplier=self.config.uncertainty_multiplier_k,
        )
        abst_expl: AbstentionExplanation = build_abstention_explanation_from_decision_evidence(dec_evidence)

        # 5. Pearl Level-3 Twin-World Counterfactual Abduction (if trusted & candidate selected)
        cf_evidence: Optional[CounterfactualEvidence] = None
        if (
            trust_diag.state != ModelTrustState.MODEL_ABSTAIN
            and recommendation.recommended_candidate is not None
            and recommendation.baseline_simulation is not None
        ):
            best_cand = recommendation.recommended_candidate
            tgt_action_name = "none"
            cf_val = 0.0
            if best_cand.spec is not None and not isinstance(best_cand.spec, list):
                tgt_action_name = best_cand.spec.target
                cf_val = float(best_cand.spec.value)
            elif isinstance(best_cand.spec, list) and len(best_cand.spec) > 0:
                tgt_action_name = "compound"
                cf_val = float(best_cand.spec[0].value)

            try:
                # Execute Twin-World Abduction & Replay
                cf_result = self.cf_engine.simulate_counterfactual(
                    historical_observations=historical_observations,
                    historical_actions=historical_actions,
                    future_factual_actions=future_baseline_actions,
                    target_action=tgt_action_name,
                    counterfactual_value=cf_val,
                    intervention_time=intervention_time,
                    parent_episode_id=scenario_id or "live_telemetry",
                )

                cf_evidence = build_counterfactual_evidence(
                    cf_result=cf_result,
                    source_evidence_hash=dec_evidence.provenance.decision_hash,
                    timestamp_utc=ts,
                )
            except Exception:
                # If CF engine simulation is skipped or not available for custom candidate, proceed gracefully
                cf_evidence = None

        # 6. Assemble Unified PRISM Decision Record
        unified_record: PrismDecisionRecord = build_unified_decision_record(
            decision_evidence=dec_evidence,
            causal_explanation=causal_expl,
            counterfactual_evidence=cf_evidence,
            safety_evidence=safety_detail,
            abstention_explanation=abst_expl,
            scenario_id=scenario_id,
            model_version=self.config.model_version,
            dataset_version=self.config.dataset_version,
            planner_version=self.config.planner_version,
            benchmark_version=self.config.benchmark_version,
            timestamp_utc=ts,
        )

        # 7. Write Artifacts if Output Directory is Specified
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fname_prefix = scenario_id or f"prism_record_{int(time.time())}"

            json_file = out_path / f"{fname_prefix}_decision_record.json"
            md_file = out_path / f"{fname_prefix}_audit_dossier.md"

            with open(json_file, "w") as f:
                f.write(unified_record.to_json(indent=2))

            with open(md_file, "w") as f:
                f.write(unified_record.format_markdown())

        return unified_record

    def run_scenario(
        self,
        scenario_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PrismDecisionRecord:
        """Load a frozen decision benchmark scenario and execute end-to-end pipeline."""
        sdir = Path(self.config.benchmark_dir) / "scenarios" / scenario_id
        if not sdir.exists():
            # Try matching scenario prefix/name
            all_dirs = list((Path(self.config.benchmark_dir) / "scenarios").glob(f"*{scenario_id}*"))
            if all_dirs:
                sdir = all_dirs[0]
            else:
                raise FileNotFoundError(f"Scenario directory not found for '{scenario_id}' in {self.config.benchmark_dir}")

        learner = LearnerDecisionScenario.load_npz(sdir / "learner.npz")
        actual_scenario_id = sdir.name

        return self.run_telemetry(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_baseline_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
            historical_mask=learner.historical_observation_mask,
            scenario_id=actual_scenario_id,
            output_dir=output_dir,
            timestamp_utc=timestamp_utc,
        )

    def run_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PrismDecisionRecord:
        """Load an NPZ telemetry file and execute end-to-end pipeline."""
        fpath = Path(file_path)
        if not fpath.exists():
            raise FileNotFoundError(f"Telemetry file not found: {fpath}")

        learner = LearnerDecisionScenario.load_npz(fpath)
        scen_id = fpath.stem if fpath.stem != "learner" else fpath.parent.name

        return self.run_telemetry(
            historical_observations=learner.historical_observations,
            historical_actions=learner.historical_actions,
            future_baseline_actions=learner.future_baseline_actions,
            custom_candidates=learner.candidate_actions,
            intervention_time=learner.intervention_time,
            historical_mask=learner.historical_observation_mask,
            scenario_id=scen_id,
            output_dir=output_dir,
            timestamp_utc=timestamp_utc,
        )

    def run_all_benchmark_scenarios(
        self,
        output_dir: Optional[Union[str, Path]] = "artifacts/phase7_dossiers",
    ) -> Dict[str, PrismDecisionRecord]:
        """Execute complete 6-scenario benchmark suite end-to-end."""
        scenarios = [
            "scenario_01_do_nothing",
            "scenario_02_valve",
            "scenario_03_throttle",
            "scenario_04_pump",
            "scenario_05_combined",
            "scenario_06_all_unsafe",
        ]
        results = {}
        for s in scenarios:
            results[s] = self.run_scenario(s, output_dir=output_dir)
        return results
