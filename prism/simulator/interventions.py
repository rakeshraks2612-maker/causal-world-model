"""Intervention Registry and Graph Surgery Operators for THC-SCM System.

Implements Pearl's Level 2 (do-calculus) atomic graph surgery:
- Severs incoming causal parents (PA(X) -> None)
- Replaces structural equations with constant or functional assignments
- Allows downstream causal descendants to update naturally
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np


@dataclass
class Intervention:
    """Represents an atomic intervention do(target = value) at step t."""

    target: str
    value: float
    start_step: int = 0
    duration: Optional[int] = None  # None = permanent for the episode
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, step: int) -> bool:
        """Check if this intervention is active at the given simulation step."""
        if step < self.start_step:
            return False
        if self.duration is None:
            return True
        return step < (self.start_step + self.duration)


class InterventionRegistry:
    """Manages active graph-surgical interventions during simulation."""

    def __init__(self) -> None:
        self.interventions: List[Intervention] = []

    def add(self, intervention: Intervention) -> None:
        """Register a new intervention."""
        self.interventions.append(intervention)

    def clear(self) -> None:
        """Clear all registered interventions."""
        self.interventions.clear()

    def get_intervened_value(self, target: str, step: int) -> Optional[float]:
        """Return forced value if variable is currently intervened, else None."""
        for inv in reversed(self.interventions):
            if inv.target == target and inv.is_active(step):
                return float(inv.value)
        return None

    def has_intervention(self, target: str, step: int) -> bool:
        """Check if target variable has an active graph-surgical override."""
        return self.get_intervened_value(target, step) is not None

    def active_targets(self, step: int) -> Set[str]:
        """Return set of all variable names currently subjected to do(X=x)."""
        return {inv.target for inv in self.interventions if inv.is_active(step)}

    @classmethod
    def create_single(cls, target: str, value: float, step: int = 0, duration: Optional[int] = None) -> InterventionRegistry:
        """Helper to create a registry with a single active intervention."""
        reg = cls()
        reg.add(Intervention(target=target, value=value, start_step=step, duration=duration))
        return reg
