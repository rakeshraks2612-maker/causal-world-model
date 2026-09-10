"""Metrics and Quantitative Scoring for Learned Level-3 Counterfactual Reasoning.

Evaluates:
1. Counterfactual Causal Effect Error E_CF and Relative Error E_rel
2. Directional Concordance (Sign Agreement) on Counterfactual Deltas
3. Peak Metric Errors (Peak T_core, Max P_sys, Min F_cool)
4. Counterfactual Failure Transition Accuracy and Safety Classification
5. Latent Abduction Estimation Accuracy (MAE against true physical unobserved state)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from prism.counterfactual.engine import LearnedCounterfactualResult
from prism.dataset.counterfactuals import (
    OracleCounterfactualRecord,
    CounterfactualHorizonEffect,
    CounterfactualFailureMetrics,
    OracleLatentTruth,
)
from prism.evaluation.intervention_metrics import SafetyClassificationSummary
from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES


EPSILON_BY_VAR: Dict[str, float] = {
    "delta_t_core": 0.5,
    "delta_t_cool": 0.5,
    "delta_p_sys": 0.05,
    "delta_f_cool": 0.5,
    "delta_l_cpu": 1.0,
    "delta_v_pos": 1.0,
    "delta_vib_pump": 0.2,
    "delta_p_elec": 0.1,
}


@dataclass
class CounterfactualEvaluationResult:
    """Paired evaluation of a single learned counterfactual result against oracle ground truth."""

    counterfactual_id: str
    target_action: str
    counterfactual_value: float
    counterfactual_time: int
    parent_episode_id: str
    causal_errors: Dict[int, Dict[str, float]]
    relative_causal_errors: Dict[int, Dict[str, float]]
    oracle_deltas: Dict[int, Dict[str, float]]
    learned_deltas: Dict[int, Dict[str, float]]
    directional_concordance: Dict[int, Dict[str, bool]]
    peak_t_core_error: float
    max_pressure_error: float
    min_flow_error: float
    oracle_failed: bool
    predicted_failed: bool
    oracle_failure_time: Optional[int]
    predicted_failure_time: Optional[int]
    failure_time_error: Optional[float]
    abduction_latent_error: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "target_action": self.target_action,
            "counterfactual_value": self.counterfactual_value,
            "counterfactual_time": self.counterfactual_time,
            "parent_episode_id": self.parent_episode_id,
            "causal_errors": {str(h): errs for h, errs in self.causal_errors.items()},
            "relative_causal_errors": {str(h): errs for h, errs in self.relative_causal_errors.items()},
            "oracle_deltas": {str(h): d for h, d in self.oracle_deltas.items()},
            "learned_deltas": {str(h): d for h, d in self.learned_deltas.items()},
            "directional_concordance": {str(h): c for h, c in self.directional_concordance.items()},
            "peak_t_core_error": self.peak_t_core_error,
            "max_pressure_error": self.max_pressure_error,
            "min_flow_error": self.min_flow_error,
            "oracle_failed": self.oracle_failed,
            "predicted_failed": self.predicted_failed,
            "oracle_failure_time": self.oracle_failure_time,
            "predicted_failure_time": self.predicted_failure_time,
            "failure_time_error": self.failure_time_error,
            "abduction_latent_error": self.abduction_latent_error,
        }


@dataclass
class CounterfactualBenchmarkSummary:
    """Aggregated benchmark evaluation across an entire counterfactual dataset."""

    total_records: int
    horizons: List[int]
    mean_causal_error_by_horizon: Dict[int, Dict[str, float]]
    mean_relative_error_by_horizon: Dict[int, Dict[str, float]]
    overall_mean_causal_error: float
    overall_mean_relative_error: float
    directional_accuracy_by_var: Dict[str, float]
    overall_directional_accuracy: float
    peak_t_core_mae: float
    max_pressure_mae: float
    min_flow_mae: float
    safety_summary: SafetyClassificationSummary
    breakdown_by_target: Dict[str, Dict[str, Any]]
    evaluations: List[CounterfactualEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "horizons": self.horizons,
            "mean_causal_error_by_horizon": {str(h): e for h, e in self.mean_causal_error_by_horizon.items()},
            "mean_relative_error_by_horizon": {str(h): e for h, e in self.mean_relative_error_by_horizon.items()},
            "overall_mean_causal_error": self.overall_mean_causal_error,
            "overall_mean_relative_error": self.overall_mean_relative_error,
            "directional_accuracy_by_var": self.directional_accuracy_by_var,
            "overall_directional_accuracy": self.overall_directional_accuracy,
            "peak_t_core_mae": self.peak_t_core_mae,
            "max_pressure_mae": self.max_pressure_mae,
            "min_flow_mae": self.min_flow_mae,
            "safety_summary": self.safety_summary.to_dict(),
            "breakdown_by_target": self.breakdown_by_target,
        }


def evaluate_single_counterfactual(
    result: LearnedCounterfactualResult,
    oracle_file_or_record: str | Path | OracleCounterfactualRecord,
) -> CounterfactualEvaluationResult:
    """Evaluate a single learned counterfactual result against oracle ground truth."""
    if isinstance(oracle_file_or_record, (str, Path)):
        data = np.load(oracle_file_or_record, allow_pickle=True)
        t_star = int(data["counterfactual_time"])
        target = str(data["target_action"])
        val = float(data["counterfactual_value"])
        cf_id = str(data["counterfactual_id"])
        parent_id = str(data["parent_episode_id"])
        int_gt = data["counterfactual_ground_truth_states"]
        horizons_dict = data["horizon_effects"].item() if data["horizon_effects"].ndim == 0 else dict(data["horizon_effects"])
        fail_dict = data["failure_metrics"].item() if data["failure_metrics"].ndim == 0 else dict(data["failure_metrics"])
        orc_failed = bool(fail_dict.get("counterfactual_failed", False))
        orc_fail_time = fail_dict.get("counterfactual_failure_time")
        rec_horizons = horizons_dict
    else:
        rec = oracle_file_or_record
        t_star = rec.counterfactual_time
        target = rec.spec.target_action
        val = rec.spec.counterfactual_value
        cf_id = rec.counterfactual_id
        parent_id = rec.parent_episode_id
        int_gt = rec.counterfactual_oracle_ep.ground_truth_states
        rec_horizons = rec.horizon_effects
        orc_failed = rec.failure_metrics.counterfactual_failed
        orc_fail_time = rec.failure_metrics.counterfactual_failure_time
    # Peak metrics
    peak_t_core_oracle = float(np.max(int_gt[t_star:, 0]))
    max_p_sys_oracle = float(np.max(int_gt[t_star:, 2]))
    min_f_cool_oracle = float(np.min(int_gt[t_star:, 3]))

    peak_t_core_error = abs(result.effects.peak_t_core_int - peak_t_core_oracle)
    max_p_sys_error = abs(result.effects.max_pressure_int - max_p_sys_oracle)
    min_f_cool_error = abs(result.effects.min_flow_int - min_f_cool_oracle)

    var_keys = [
        "delta_t_core",
        "delta_t_cool",
        "delta_p_sys",
        "delta_f_cool",
        "delta_l_cpu",
        "delta_v_pos",
        "delta_vib_pump",
        "delta_p_elec",
    ]

    causal_errors: Dict[int, Dict[str, float]] = {}
    relative_causal_errors: Dict[int, Dict[str, float]] = {}
    oracle_deltas: Dict[int, Dict[str, float]] = {}
    learned_deltas: Dict[int, Dict[str, float]] = {}
    directional_concordance: Dict[int, Dict[str, bool]] = {}

    for h in result.horizons:
        orc_h = None
        if h in rec_horizons:
            orc_h = rec_horizons[h]
        elif str(h) in rec_horizons:
            orc_h = rec_horizons[str(h)]

        if orc_h is not None:
            if isinstance(orc_h, dict):
                orc_h_dict = orc_h
            elif hasattr(orc_h, "to_dict"):
                orc_h_dict = orc_h.to_dict()
            else:
                orc_h_dict = {vk: getattr(orc_h, vk, 0.0) for vk in var_keys}

            pred_h = result.effects.horizon_effects.get(h)

            h_errs: Dict[str, float] = {}
            h_rels: Dict[str, float] = {}
            h_orc: Dict[str, float] = {}
            h_prd: Dict[str, float] = {}
            h_sign: Dict[str, bool] = {}

            for vk in var_keys:
                orc_val = float(orc_h_dict.get(vk, 0.0))
                prd_val = float(getattr(pred_h, vk, 0.0)) if pred_h is not None else 0.0
                err_val = abs(prd_val - orc_val)

                eps = EPSILON_BY_VAR.get(vk, 0.5)
                rel_val = err_val / (abs(orc_val) + eps)

                if abs(orc_val) < 0.05 and abs(prd_val) < 0.05:
                    sign_match = True
                else:
                    sign_match = bool(np.sign(orc_val) == np.sign(prd_val))

                h_errs[vk] = err_val
                h_rels[vk] = rel_val
                h_orc[vk] = orc_val
                h_prd[vk] = prd_val
                h_sign[vk] = sign_match

            causal_errors[h] = h_errs
            relative_causal_errors[h] = h_rels
            oracle_deltas[h] = h_orc
            learned_deltas[h] = h_prd
            directional_concordance[h] = h_sign

    pred_failed = result.effects.failure_metrics.intervened_failed
    pred_fail_time = result.effects.failure_metrics.intervention_failure_time

    fail_time_err = None
    if orc_failed and pred_failed and orc_fail_time is not None and pred_fail_time is not None:
        fail_time_err = float(abs(pred_fail_time - orc_fail_time))

    return CounterfactualEvaluationResult(
        counterfactual_id=cf_id,
        target_action=target,
        counterfactual_value=val,
        counterfactual_time=t_star,
        parent_episode_id=parent_id,
        causal_errors=causal_errors,
        relative_causal_errors=relative_causal_errors,
        oracle_deltas=oracle_deltas,
        learned_deltas=learned_deltas,
        directional_concordance=directional_concordance,
        peak_t_core_error=peak_t_core_error,
        max_pressure_error=max_p_sys_error,
        min_flow_error=min_f_cool_error,
        oracle_failed=orc_failed,
        predicted_failed=pred_failed,
        oracle_failure_time=orc_fail_time,
        predicted_failure_time=pred_fail_time,
        failure_time_error=fail_time_err,
    )


def aggregate_counterfactual_benchmark(
    evaluations: List[CounterfactualEvaluationResult],
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> CounterfactualBenchmarkSummary:
    """Aggregate paired counterfactual evaluations into a summary report."""
    if not evaluations:
        raise ValueError("Cannot aggregate empty list of evaluations")

    total_records = len(evaluations)
    var_keys = [
        "delta_t_core",
        "delta_t_cool",
        "delta_p_sys",
        "delta_f_cool",
        "delta_l_cpu",
        "delta_v_pos",
        "delta_vib_pump",
        "delta_p_elec",
    ]

    mean_causal_error_by_horizon: Dict[int, Dict[str, float]] = {}
    mean_relative_error_by_horizon: Dict[int, Dict[str, float]] = {}
    all_causal_errors: List[float] = []
    all_relative_errors: List[float] = []

    for h in horizons:
        mean_causal_error_by_horizon[h] = {}
        mean_relative_error_by_horizon[h] = {}
        for vk in var_keys:
            vals = [e.causal_errors[h][vk] for e in evaluations if h in e.causal_errors and vk in e.causal_errors[h]]
            rels = [e.relative_causal_errors[h][vk] for e in evaluations if h in e.relative_causal_errors and vk in e.relative_causal_errors[h]]
            mean_val = float(np.mean(vals)) if vals else 0.0
            mean_rel = float(np.mean(rels)) if rels else 0.0
            mean_causal_error_by_horizon[h][vk] = mean_val
            mean_relative_error_by_horizon[h][vk] = mean_rel
            all_causal_errors.extend(vals)
            all_relative_errors.extend(rels)

    overall_mean_causal_error = float(np.mean(all_causal_errors)) if all_causal_errors else 0.0
    overall_mean_relative_error = float(np.mean(all_relative_errors)) if all_relative_errors else 0.0

    directional_accuracy_by_var: Dict[str, float] = {}
    all_signs: List[bool] = []
    for vk in var_keys:
        v_signs = []
        for e in evaluations:
            for h in horizons:
                if h in e.directional_concordance and vk in e.directional_concordance[h]:
                    v_signs.append(e.directional_concordance[h][vk])
        directional_accuracy_by_var[vk] = float(np.mean(v_signs)) if v_signs else 1.0
        all_signs.extend(v_signs)

    overall_directional_accuracy = float(np.mean(all_signs)) if all_signs else 1.0

    peak_t_core_mae = float(np.mean([e.peak_t_core_error for e in evaluations]))
    max_p_mae = float(np.mean([e.max_pressure_error for e in evaluations]))
    min_f_mae = float(np.mean([e.min_flow_error for e in evaluations]))

    tp = sum(1 for e in evaluations if e.oracle_failed and e.predicted_failed)
    fn = sum(1 for e in evaluations if e.oracle_failed and not e.predicted_failed)
    fp = sum(1 for e in evaluations if not e.oracle_failed and e.predicted_failed)
    tn = sum(1 for e in evaluations if not e.oracle_failed and not e.predicted_failed)

    false_safe_rate = fn / max(1, (tp + fn))

    safety_summary = SafetyClassificationSummary(
        true_critical=tp,
        false_safe=fn,
        false_alarm=fp,
        true_safe=tn,
        false_safe_rate=false_safe_rate,
    )

    breakdown_by_target: Dict[str, Dict[str, Any]] = {}
    targets = sorted(list({e.target_action for e in evaluations}))
    for tgt in targets:
        sub = [e for e in evaluations if e.target_action == tgt]
        sub_errs = [err for e in sub for h in horizons if h in e.causal_errors for err in e.causal_errors[h].values()]
        sub_signs = [s for e in sub for h in horizons if h in e.directional_concordance for s in e.directional_concordance[h].values()]
        breakdown_by_target[tgt] = {
            "count": len(sub),
            "mean_causal_error": float(np.mean(sub_errs)) if sub_errs else 0.0,
            "directional_accuracy": float(np.mean(sub_signs)) if sub_signs else 1.0,
            "peak_t_core_mae": float(np.mean([e.peak_t_core_error for e in sub])),
        }

    return CounterfactualBenchmarkSummary(
        total_records=total_records,
        horizons=list(horizons),
        mean_causal_error_by_horizon=mean_causal_error_by_horizon,
        mean_relative_error_by_horizon=mean_relative_error_by_horizon,
        overall_mean_causal_error=overall_mean_causal_error,
        overall_mean_relative_error=overall_mean_relative_error,
        directional_accuracy_by_var=directional_accuracy_by_var,
        overall_directional_accuracy=overall_directional_accuracy,
        peak_t_core_mae=peak_t_core_mae,
        max_pressure_mae=max_p_mae,
        min_flow_mae=min_f_mae,
        safety_summary=safety_summary,
        breakdown_by_target=breakdown_by_target,
        evaluations=evaluations,
    )
