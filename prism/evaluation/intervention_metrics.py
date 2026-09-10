"""Intervention Evaluation Metrics and Causal Error Benchmarking.

Implements evaluation protocols comparing learned PRISM interventions against Oracle ground truth:
- Causal error: E_causal = |Delta Y_learned - Delta Y_oracle|
- Directional sign concordance
- Peak outcome errors: peak T_core, max pressure, min flow
- Failure prediction accuracy and Brier score
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from prism.dataset.interventions import (
    OracleInterventionRecord,
    LearnerInterventionRecord,
    HorizonEffect,
)
from prism.intervention.simulator import (
    LearnedInterventionSimulator,
    LearnedInterventionResult,
)
from prism.intervention.effects import CausalEffectSummary
from prism.simulator.state import OBSERVABLE_VARIABLES


@dataclass
class PairedInterventionEvaluation:
    """Evaluation metrics for a single paired intervention episode."""

    intervention_id: str
    target: str
    value: float
    intervention_time: int
    category: str
    causal_errors: Dict[int, Dict[str, float]]       # horizon -> {var -> |Delta_pred - Delta_oracle|}
    oracle_deltas: Dict[int, Dict[str, float]]       # horizon -> {var -> Delta_oracle}
    learned_deltas: Dict[int, Dict[str, float]]      # horizon -> {var -> Delta_pred}
    directional_concordance: Dict[int, Dict[str, bool]] # horizon -> {var -> sign_match}
    peak_t_core_error: float
    max_pressure_error: float
    min_flow_error: float
    oracle_failed: bool
    predicted_failed: bool
    oracle_failure_time: Optional[int]
    predicted_failure_time: Optional[int]


@dataclass
class InterventionBenchmarkSummary:
    """Aggregated benchmark evaluation across all intervention records."""

    total_records: int
    horizons: List[int]
    mean_causal_error_by_horizon: Dict[int, Dict[str, float]]  # horizon -> {var -> mean E_causal}
    overall_mean_causal_error: float
    directional_accuracy_by_var: Dict[str, float]
    overall_directional_accuracy: float
    peak_t_core_mae: float
    max_pressure_mae: float
    min_flow_mae: float
    failure_accuracy: float
    failure_precision: float
    failure_recall: float
    failure_f1: float
    breakdown_by_target: Dict[str, Dict[str, Any]]
    evaluations: List[PairedInterventionEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "horizons": self.horizons,
            "mean_causal_error_by_horizon": {
                str(h): errs for h, errs in self.mean_causal_error_by_horizon.items()
            },
            "overall_mean_causal_error": self.overall_mean_causal_error,
            "directional_accuracy_by_var": self.directional_accuracy_by_var,
            "overall_directional_accuracy": self.overall_directional_accuracy,
            "peak_t_core_mae": self.peak_t_core_mae,
            "max_pressure_mae": self.max_pressure_mae,
            "min_flow_mae": self.min_flow_mae,
            "failure_accuracy": self.failure_accuracy,
            "failure_precision": self.failure_precision,
            "failure_recall": self.failure_recall,
            "failure_f1": self.failure_f1,
            "breakdown_by_target": self.breakdown_by_target,
        }

    def format_table(self, horizon: int = 10) -> str:
        """Format a summary markdown table of causal effect errors at a specific horizon."""
        lines = [
            f"| Variable | Oracle Mean Δ | Learned Mean Δ | Causal Error E_causal | Sign Concordance |",
            f"| :--- | :--- | :--- | :--- | :--- |",
        ]
        var_keys = [
            ("T_core", "delta_t_core", "°C"),
            ("T_cool", "delta_t_cool", "°C"),
            ("P_sys", "delta_p_sys", "bar"),
            ("F_cool", "delta_f_cool", "L/min"),
            ("L_cpu", "delta_l_cpu", "%"),
            ("V_pos", "delta_v_pos", "%"),
            ("Vib_pump", "delta_vib_pump", "mm/s"),
        ]
        for var_name, delta_key, unit in var_keys:
            oracle_vals = [e.oracle_deltas.get(horizon, {}).get(delta_key, 0.0) for e in self.evaluations]
            pred_vals = [e.learned_deltas.get(horizon, {}).get(delta_key, 0.0) for e in self.evaluations]
            err_vals = [e.causal_errors.get(horizon, {}).get(delta_key, 0.0) for e in self.evaluations]
            sign_vals = [e.directional_concordance.get(horizon, {}).get(delta_key, True) for e in self.evaluations]

            mean_orc = float(np.mean(oracle_vals)) if oracle_vals else 0.0
            mean_prd = float(np.mean(pred_vals)) if pred_vals else 0.0
            mean_err = float(np.mean(err_vals)) if err_vals else 0.0
            sign_acc = float(np.mean(sign_vals)) if sign_vals else 1.0

            lines.append(
                f"| **{var_name}** | {mean_orc:+.2f} {unit} | {mean_prd:+.2f} {unit} | {mean_err:.3f} {unit} | {sign_acc:.1%} |"
            )
        return "\n".join(lines)


def evaluate_single_intervention(
    result: LearnedInterventionResult,
    oracle_file_or_record: str | Path | OracleInterventionRecord,
) -> PairedInterventionEvaluation:
    """Evaluate a single learned intervention result against oracle ground truth."""
    if isinstance(oracle_file_or_record, (str, Path)):
        oracle_data = np.load(oracle_file_or_record, allow_pickle=True)
        t_star = int(oracle_data["intervention_time"])
        target = str(oracle_data["target"])
        val = float(oracle_data["value"])
        cat = str(oracle_data["category"])
        inv_id = str(oracle_data["intervention_id"])

        horizons_raw = oracle_data["horizon_effects"].item()
        base_gt = oracle_data["baseline_ground_truth_states"]
        int_gt = oracle_data["intervened_ground_truth_states"]

        # Parse oracle failure info
        fail_meta = oracle_data.get("failure_metrics")
        if fail_meta is not None:
            f_dict = fail_meta.item() if fail_meta.ndim == 0 else dict(fail_meta)
            orc_failed = bool(f_dict.get("intervened_failed", False))
            orc_fail_time = f_dict.get("intervention_failure_time")
        else:
            orc_failed = False
            orc_fail_time = None
    else:
        rec = oracle_file_or_record
        t_star = rec.intervention_time
        target = rec.target
        val = rec.value
        cat = rec.category.value if hasattr(rec.category, "value") else str(rec.category)
        inv_id = rec.intervention_id
        horizons_raw = {str(h): eff.to_dict() for h, eff in rec.horizon_effects.items()}
        base_gt = rec.baseline_oracle_ep.ground_truth_states
        int_gt = rec.intervened_oracle_ep.ground_truth_states
        orc_failed = rec.failure_metrics.intervened_failed
        orc_fail_time = rec.failure_metrics.intervention_failure_time

    # True oracle peaks
    peak_t_core_oracle = float(np.max(int_gt[t_star:, 0]))
    max_p_sys_oracle = float(np.max(int_gt[t_star:, 2]))
    min_f_cool_oracle = float(np.min(int_gt[t_star:, 3]))

    # Learned peaks
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
    oracle_deltas: Dict[int, Dict[str, float]] = {}
    learned_deltas: Dict[int, Dict[str, float]] = {}
    directional_concordance: Dict[int, Dict[str, bool]] = {}

    for h in result.horizons:
        str_h = str(h)
        if str_h in horizons_raw:
            orc_h = horizons_raw[str_h]
            if isinstance(orc_h, HorizonEffect):
                orc_h = orc_h.to_dict()

            pred_h = result.effects.horizon_effects.get(h)

            h_errs: Dict[str, float] = {}
            h_orc: Dict[str, float] = {}
            h_prd: Dict[str, float] = {}
            h_sign: Dict[str, bool] = {}

            for vk in var_keys:
                orc_val = float(orc_h.get(vk, 0.0))
                prd_val = float(getattr(pred_h, vk)) if pred_h is not None else 0.0
                err_val = abs(prd_val - orc_val)

                # Sign concordance check (with 0.05 threshold deadband for noise)
                if abs(orc_val) < 0.05 and abs(prd_val) < 0.05:
                    sign_match = True
                else:
                    sign_match = bool(np.sign(orc_val) == np.sign(prd_val))

                h_errs[vk] = err_val
                h_orc[vk] = orc_val
                h_prd[vk] = prd_val
                h_sign[vk] = sign_match

            causal_errors[h] = h_errs
            oracle_deltas[h] = h_orc
            learned_deltas[h] = h_prd
            directional_concordance[h] = h_sign

    pred_failed = result.effects.failure_metrics.intervened_failed
    pred_fail_time = result.effects.failure_metrics.intervention_failure_time

    return PairedInterventionEvaluation(
        intervention_id=inv_id,
        target=target,
        value=val,
        intervention_time=t_star,
        category=cat,
        causal_errors=causal_errors,
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
    )


def aggregate_intervention_benchmark(
    evaluations: List[PairedInterventionEvaluation],
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> InterventionBenchmarkSummary:
    """Aggregate paired evaluation records into a comprehensive benchmark summary."""
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
    all_causal_errors: List[float] = []

    for h in horizons:
        mean_causal_error_by_horizon[h] = {}
        for vk in var_keys:
            vals = [e.causal_errors[h][vk] for e in evaluations if h in e.causal_errors and vk in e.causal_errors[h]]
            mean_val = float(np.mean(vals)) if vals else 0.0
            mean_causal_error_by_horizon[h][vk] = mean_val
            all_causal_errors.extend(vals)

    overall_mean_causal_error = float(np.mean(all_causal_errors)) if all_causal_errors else 0.0

    # Directional accuracy per variable
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

    # Peak errors
    peak_t_core_mae = float(np.mean([e.peak_t_core_error for e in evaluations]))
    max_p_mae = float(np.mean([e.max_pressure_error for e in evaluations]))
    min_f_mae = float(np.mean([e.min_flow_error for e in evaluations]))

    # Failure metrics (classification)
    tp = sum(1 for e in evaluations if e.oracle_failed and e.predicted_failed)
    fp = sum(1 for e in evaluations if not e.oracle_failed and e.predicted_failed)
    tn = sum(1 for e in evaluations if not e.oracle_failed and not e.predicted_failed)
    fn = sum(1 for e in evaluations if e.oracle_failed and not e.predicted_failed)

    fail_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    fail_prec = tp / max(1, (tp + fp))
    fail_rec = tp / max(1, (tp + fn))
    fail_f1 = (2 * fail_prec * fail_rec) / max(1e-4, (fail_prec + fail_rec))

    # Breakdown by target variable
    breakdown_by_target: Dict[str, Dict[str, Any]] = {}
    targets = sorted(list({e.target for e in evaluations}))
    for tgt in targets:
        sub = [e for e in evaluations if e.target == tgt]
        sub_errs: List[float] = []
        for e in sub:
            for h in horizons:
                if h in e.causal_errors:
                    sub_errs.extend(e.causal_errors[h].values())
        breakdown_by_target[tgt] = {
            "count": len(sub),
            "mean_causal_error": float(np.mean(sub_errs)) if sub_errs else 0.0,
            "peak_t_core_mae": float(np.mean([e.peak_t_core_error for e in sub])),
        }

    return InterventionBenchmarkSummary(
        total_records=total_records,
        horizons=list(horizons),
        mean_causal_error_by_horizon=mean_causal_error_by_horizon,
        overall_mean_causal_error=overall_mean_causal_error,
        directional_accuracy_by_var=directional_accuracy_by_var,
        overall_directional_accuracy=overall_directional_accuracy,
        peak_t_core_mae=peak_t_core_mae,
        max_pressure_mae=max_p_mae,
        min_flow_mae=min_f_mae,
        failure_accuracy=fail_acc,
        failure_precision=fail_prec,
        failure_recall=fail_rec,
        failure_f1=fail_f1,
        breakdown_by_target=breakdown_by_target,
        evaluations=evaluations,
    )
