"""PRISM Counterfactual Engine Package (Pearl Level-3 Causal Reasoning)."""

from prism.counterfactual.engine import (
    LearnedCounterfactualEngine,
    LearnedCounterfactualResult,
    AbducedLatentState,
)
from prism.counterfactual.metrics import (
    CounterfactualEvaluationResult,
    CounterfactualBenchmarkSummary,
    evaluate_single_counterfactual,
    aggregate_counterfactual_benchmark,
)

__all__ = [
    "LearnedCounterfactualEngine",
    "LearnedCounterfactualResult",
    "AbducedLatentState",
    "CounterfactualEvaluationResult",
    "CounterfactualBenchmarkSummary",
    "evaluate_single_counterfactual",
    "aggregate_counterfactual_benchmark",
]
