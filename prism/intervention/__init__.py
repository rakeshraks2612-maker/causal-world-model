"""PRISM Learned Intervention Engine Package.

Provides:
- InterventionSpec, InterventionType, state_clamp, action_control
- InterventionOperator
- LearnedInterventionSimulator, LearnedInterventionResult
- CausalEffectSummary, HorizonCausalDelta, LearnedFailureMetrics, compute_causal_effects
- Validation routines and causal invariant assertions
"""

from __future__ import annotations

from prism.intervention.spec import (
    InterventionSpec,
    InterventionType,
    state_clamp,
    action_control,
)
from prism.intervention.operator import (
    InterventionOperator,
    OBSERVABLE_TO_IDX,
    ACTION_TO_IDX,
)
from prism.intervention.effects import (
    HorizonCausalDelta,
    LearnedFailureMetrics,
    CausalEffectSummary,
    compute_causal_effects,
)
from prism.intervention.simulator import (
    LearnedInterventionSimulator,
    LearnedInterventionResult,
)
from prism.intervention.validation import (
    check_no_intervention_equivalence,
    check_pre_intervention_identity,
    check_non_descendant_invariance,
    check_valve_directionality,
    check_throttle_directionality,
    check_state_clamp_vs_action_control,
)

__all__ = [
    "InterventionSpec",
    "InterventionType",
    "state_clamp",
    "action_control",
    "InterventionOperator",
    "OBSERVABLE_TO_IDX",
    "ACTION_TO_IDX",
    "HorizonCausalDelta",
    "LearnedFailureMetrics",
    "CausalEffectSummary",
    "compute_causal_effects",
    "LearnedInterventionSimulator",
    "LearnedInterventionResult",
    "check_no_intervention_equivalence",
    "check_pre_intervention_identity",
    "check_non_descendant_invariance",
    "check_valve_directionality",
    "check_throttle_directionality",
    "check_state_clamp_vs_action_control",
]
