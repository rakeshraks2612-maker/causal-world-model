"""Action Schema and Validation for THC-SCM System.

Defines the 4-dimensional control action vector:
- A_valve: Continuous [0.0, 100.0]%
- A_throttle: Continuous [10.0, 100.0]%
- A_pump: Discrete {1, 2, 3, 4}
- A_flush: Binary {0, 1}
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import numpy as np

ACTION_VARIABLES: List[str] = [
    "A_valve",
    "A_throttle",
    "A_pump",
    "A_flush",
]


@dataclass(frozen=True)
class ActionVector:
    """Represents the control action executed at time step t.
    
    Attributes:
        A_valve: Target coolant valve opening setpoint [0.0, 100.0]%
        A_throttle: Upper bound limit on compute workload [10.0, 100.0]%
        A_pump: Discrete pump head stage setting {1, 2, 3, 4}
        A_flush: Emergency line purge command {0, 1}
    """

    A_valve: float
    A_throttle: float
    A_pump: int
    A_flush: int

    def __post_init__(self) -> None:
        """Strict validation of action bounds and types."""
        # A_valve validation
        if not isinstance(self.A_valve, (int, float, np.floating)):
            raise TypeError(f"A_valve must be numeric, got {type(self.A_valve)}")
        if not (0.0 <= float(self.A_valve) <= 100.0):
            raise ValueError(f"A_valve must be in [0.0, 100.0]%, got {self.A_valve}")

        # A_throttle validation
        if not isinstance(self.A_throttle, (int, float, np.floating)):
            raise TypeError(f"A_throttle must be numeric, got {type(self.A_throttle)}")
        if not (10.0 <= float(self.A_throttle) <= 100.0):
            raise ValueError(f"A_throttle must be in [10.0, 100.0]%, got {self.A_throttle}")

        # A_pump validation
        if int(self.A_pump) not in (1, 2, 3, 4) or not isinstance(self.A_pump, (int, np.integer)):
            raise ValueError(f"A_pump must be in {{1, 2, 3, 4}}, got {self.A_pump}")

        # A_flush validation
        if int(self.A_flush) not in (0, 1) or not isinstance(self.A_flush, (int, np.integer)):
            raise ValueError(f"A_flush must be binary in {{0, 1}}, got {self.A_flush}")

    @classmethod
    def default_nominal(cls) -> ActionVector:
        """Return standard nominal control actions."""
        return cls(
            A_valve=55.0,
            A_throttle=100.0,
            A_pump=2,
            A_flush=0,
        )

    def to_array(self) -> np.ndarray:
        """Convert action to 4-element numpy float64 array."""
        return np.array(
            [float(self.A_valve), float(self.A_throttle), float(self.A_pump), float(self.A_flush)],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ActionVector:
        """Construct ActionVector from 4-element array."""
        if len(arr) != 4:
            raise ValueError(f"Expected 4 elements for ActionVector, got {len(arr)}")
        return cls(
            A_valve=float(arr[0]),
            A_throttle=float(arr[1]),
            A_pump=int(round(arr[2])),
            A_flush=int(round(arr[3])),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ActionVector:
        """Construct ActionVector from dictionary."""
        missing = [k for k in ACTION_VARIABLES if k not in d]
        if missing:
            raise KeyError(f"Missing action variables in dict: {missing}")
        return cls(
            A_valve=float(d["A_valve"]),
            A_throttle=float(d["A_throttle"]),
            A_pump=int(d["A_pump"]),
            A_flush=int(d["A_flush"]),
        )
