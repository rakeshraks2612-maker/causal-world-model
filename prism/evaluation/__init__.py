"""PRISM Evaluation Subsystem: Multi-Step Rollouts, Calibration, OOD Generalization, and Action Branching."""

from __future__ import annotations

from prism.evaluation.rollout_metrics import (
    HorizonChannelMetric,
    HorizonEvaluationSummary,
    MultiStepEvaluationReport,
    compute_horizon_metrics,
)
from prism.evaluation.forecasting import (
    evaluate_prism_multistep,
    evaluate_mlp_multistep,
    evaluate_persistence_multistep,
)
from prism.evaluation.uncertainty_metrics import (
    ChannelUncertaintySummary,
    HorizonUncertaintySummary,
    MonteCarloUncertaintyReport,
    evaluate_monte_carlo_uncertainty,
)
from prism.evaluation.calibration import (
    CalibrationStatistics,
    compute_interval_calibration,
)
from prism.evaluation.ood import (
    RegimeUncertaintySummary,
    OODEvaluationReport,
    evaluate_ood_regimes,
)
from prism.evaluation.action_branching import (
    BranchTrajectory,
    ActionBranchingExperiment,
    ActionBranchingReport,
    evaluate_action_branching,
)

__all__ = [
    "HorizonChannelMetric",
    "HorizonEvaluationSummary",
    "MultiStepEvaluationReport",
    "compute_horizon_metrics",
    "evaluate_prism_multistep",
    "evaluate_mlp_multistep",
    "evaluate_persistence_multistep",
    "ChannelUncertaintySummary",
    "HorizonUncertaintySummary",
    "MonteCarloUncertaintyReport",
    "evaluate_monte_carlo_uncertainty",
    "CalibrationStatistics",
    "compute_interval_calibration",
    "RegimeUncertaintySummary",
    "OODEvaluationReport",
    "evaluate_ood_regimes",
    "BranchTrajectory",
    "ActionBranchingExperiment",
    "ActionBranchingReport",
    "evaluate_action_branching",
]
