"""Intervention Specification and Schema for Learned Causal World Models.

Defines:
- InterventionType: STATE_CLAMP (do(X=x)) vs ACTION_CONTROL (A=a)
- InterventionSpec: Complete specification of target, value, t*, duration, and metadata
- Invariant checking and validation
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np

from prism.simulator.state import ALL_STATE_VARIABLES, OBSERVABLE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES


class InterventionType(str, Enum):
    """Type of causal intervention."""
    STATE_CLAMP = "state_clamp"        # do(X_j = x): Graph surgery on state/observable variable
    ACTION_CONTROL = "action_control"  # A_j = a: Control action setpoint override


@dataclass
class InterventionSpec:
    """Specification of an intervention applied to the learned causal world model.
    
    Attributes:
        target: Name of target variable (e.g., 'V_pos', 'L_cpu', 'Vib_pump', 'A_valve', etc.)
        value: Numerical value to assign
        intervention_time: Timestep t* when intervention takes effect
        duration: Number of timesteps intervention remains active (None = permanent/persistent)
        intervention_type: STATE_CLAMP or ACTION_CONTROL (auto-inferred if omitted)
        metadata: Optional dictionary with context or provenance
    """

    target: str
    value: float
    intervention_time: int = 0
    duration: Optional[int] = None
    intervention_type: Optional[InterventionType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Infer type if not provided and validate specification."""
        if self.intervention_type is None:
            if self.target in ACTION_VARIABLES:
                self.intervention_type = InterventionType.ACTION_CONTROL
            else:
                self.intervention_type = InterventionType.STATE_CLAMP
        elif isinstance(self.intervention_type, str):
            self.intervention_type = InterventionType(self.intervention_type)

        self.value = float(self.value)
        self.intervention_time = int(self.intervention_time)
        if self.duration is not None:
            self.duration = int(self.duration)

        self.validate()

    def validate(self) -> None:
        """Validate bounds, types, and physical domains."""
        valid_targets = set(ALL_STATE_VARIABLES) | set(OBSERVABLE_VARIABLES) | set(ACTION_VARIABLES)
        if self.target not in valid_targets:
            raise ValueError(f"Unknown intervention target '{self.target}'. Valid targets: {sorted(valid_targets)}")

        if self.intervention_time < 0:
            raise ValueError(f"Intervention time t* must be non-negative, got {self.intervention_time}")

        if self.duration is not None and self.duration <= 0:
            raise ValueError(f"Intervention duration must be positive if specified, got {self.duration}")

        # Domain boundary checks
        if self.target in ("V_pos", "A_valve") and not (0.0 <= self.value <= 100.0):
            raise ValueError(f"{self.target} value must be in [0.0, 100.0]%, got {self.value}")

        if self.target == "A_throttle" and not (10.0 <= self.value <= 100.0):
            raise ValueError(f"A_throttle value must be in [10.0, 100.0]%, got {self.value}")

        if self.target == "L_cpu" and not (0.0 <= self.value <= 100.0):
            raise ValueError(f"L_cpu value must be in [0.0, 100.0]%, got {self.value}")

        if self.target == "A_pump" and int(self.value) not in (1, 2, 3, 4):
            raise ValueError(f"A_pump value must be in {{1, 2, 3, 4}}, got {self.value}")

        if self.target == "A_flush" and int(self.value) not in (0, 1):
            raise ValueError(f"A_flush value must be binary {{0, 1}}, got {self.value}")

        if self.target == "Vib_pump" and not (0.0 <= self.value <= 30.0):
            raise ValueError(f"Vib_pump value must be in [0.0, 30.0] mm/s, got {self.value}")

    def is_active(self, step: int) -> bool:
        """Check if this intervention is active at timestep step."""
        if step < self.intervention_time:
            return False
        if self.duration is None:
            return True
        return step < (self.intervention_time + self.duration)

    @property
    def is_state_clamp(self) -> bool:
        """True if this is a state clamp do(X=x)."""
        return self.intervention_type == InterventionType.STATE_CLAMP

    @property
    def is_action_control(self) -> bool:
        """True if this is an action control A=a."""
        return self.intervention_type == InterventionType.ACTION_CONTROL

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        d = asdict(self)
        d["intervention_type"] = self.intervention_type.value if self.intervention_type else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> InterventionSpec:
        """Construct InterventionSpec from dictionary."""
        itype = d.get("intervention_type")
        if itype is not None and not isinstance(itype, InterventionType):
            itype = InterventionType(str(itype))
        return cls(
            target=str(d["target"]),
            value=float(d["value"]),
            intervention_time=int(d.get("intervention_time", 0)),
            duration=int(d["duration"]) if d.get("duration") is not None else None,
            intervention_type=itype,
            metadata=dict(d.get("metadata", {})),
        )

    def __repr__(self) -> str:
        dur_str = f", dur={self.duration}" if self.duration is not None else ", perm"
        if self.is_state_clamp:
            return f"do({self.target}={self.value:g} @ t*={self.intervention_time}{dur_str})"
        return f"set({self.target}={self.value:g} @ t*={self.intervention_time}{dur_str})"


def state_clamp(
    target: str,
    value: float,
    intervention_time: int = 0,
    duration: Optional[int] = None,
    **metadata: Any,
) -> InterventionSpec:
    """Helper to construct an atomic state clamp do(X=x)."""
    return InterventionSpec(
        target=target,
        value=value,
        intervention_time=intervention_time,
        duration=duration,
        intervention_type=InterventionType.STATE_CLAMP,
        metadata=metadata,
    )


def action_control(
    target: str,
    value: float,
    intervention_time: int = 0,
    duration: Optional[int] = None,
    **metadata: Any,
) -> InterventionSpec:
    """Helper to construct an action control A=a."""
    return InterventionSpec(
        target=target,
        value=value,
        intervention_time=intervention_time,
        duration=duration,
        intervention_type=InterventionType.ACTION_CONTROL,
        metadata=metadata,
    )
