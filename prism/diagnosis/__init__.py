"""PRISM Diagnosis and Model Inspection Suite."""

from __future__ import annotations

from prism.diagnosis.domain_metrics import (
    DOMAINS,
    PHYSICAL_UNITS,
    DomainSummary,
    ComprehensiveEvaluationReport,
    compute_comprehensive_metrics,
)
from prism.diagnosis.reconstruction_vs_prediction import evaluate_reconstruction_vs_prediction
from prism.diagnosis.latent_probes import fit_and_evaluate_latent_probes, LatentProbeReport
from prism.diagnosis.action_sensitivity import evaluate_action_sensitivity, ActionSensitivityReport

__all__ = [
    "DOMAINS",
    "PHYSICAL_UNITS",
    "DomainSummary",
    "ComprehensiveEvaluationReport",
    "compute_comprehensive_metrics",
    "evaluate_reconstruction_vs_prediction",
    "fit_and_evaluate_latent_probes",
    "LatentProbeReport",
    "evaluate_action_sensitivity",
    "ActionSensitivityReport",
]
