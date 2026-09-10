"""State Schema and Container for the THC-SCM System.

Defines the exact 12-dimensional physical state vector (8 observable, 4 latent).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import numpy as np

OBSERVABLE_VARIABLES: List[str] = [
    "T_core",
    "T_cool",
    "P_sys",
    "F_cool",
    "L_cpu",
    "V_pos",
    "Vib_pump",
    "P_elec",
]

LATENT_VARIABLES: List[str] = [
    "T_amb",
    "W_wear",
    "Q_internal",
    "xi_leak",
]

ALL_STATE_VARIABLES: List[str] = OBSERVABLE_VARIABLES + LATENT_VARIABLES

STATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "T_core": (15.0, 135.0),
    "T_cool": (10.0, 105.0),
    "P_sys": (0.2, 7.0),
    "F_cool": (0.0, 65.0),
    "L_cpu": (0.0, 100.0),
    "V_pos": (0.0, 100.0),
    "Vib_pump": (0.0, 30.0),
    "P_elec": (0.1, 6.0),
    "T_amb": (5.0, 50.0),
    "W_wear": (0.0, 1.0),
    "Q_internal": (-1.0, 3.5),
    "xi_leak": (0.0, 60.0),
}


@dataclass(frozen=True)
class StateVector:
    """Represents the complete 12-dimensional physical state at step t.
    
    Observable state variables (8):
        T_core: Core silicon junction temperature (°C)
        T_cool: Coolant loop temperature (°C)
        P_sys: Hydraulic system pressure (bar)
        F_cool: Coolant volumetric flow rate (L/min)
        L_cpu: Compute CPU utilization (%)
        V_pos: Valve opening position (%)
        Vib_pump: Mechanical chassis vibration (mm/s)
        P_elec: Electrical power consumption (kW)
        
    Hidden/Latent state variables (4):
        T_amb: External ambient temperature (°C, confounder)
        W_wear: Cumulative mechanical/thermal fouling degradation ([0, 1])
        Q_internal: Micro-architectural hotspot thermal flux (kW)
        xi_leak: Latent fluid micro-leak rate (mL/hr)
    """

    # Observable
    T_core: float
    T_cool: float
    P_sys: float
    F_cool: float
    L_cpu: float
    V_pos: float
    Vib_pump: float
    P_elec: float

    # Latent
    T_amb: float
    W_wear: float
    Q_internal: float
    xi_leak: float

    def __post_init__(self) -> None:
        """Validate types and physical validity."""
        for var_name in ALL_STATE_VARIABLES:
            val = getattr(self, var_name)
            if not isinstance(val, (int, float, np.floating)):
                raise TypeError(f"State variable '{var_name}' must be float/int, got {type(val)}: {val}")
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"State variable '{var_name}' cannot be NaN or Inf: {val}")

    @classmethod
    def default_nominal(cls) -> StateVector:
        """Return a typical steady-state nominal operating point."""
        return cls(
            T_core=68.5,
            T_cool=34.0,
            P_sys=3.15,
            F_cool=32.0,
            L_cpu=50.0,
            V_pos=55.0,
            Vib_pump=4.2,
            P_elec=1.85,
            T_amb=25.0,
            W_wear=0.05,
            Q_internal=0.15,
            xi_leak=0.0,
        )

    def to_array(self) -> np.ndarray:
        """Convert full 12-dimensional state to a numpy float64 array."""
        return np.array([getattr(self, k) for k in ALL_STATE_VARIABLES], dtype=np.float64)

    def observable_array(self) -> np.ndarray:
        """Extract only the 8 observable state variables."""
        return np.array([getattr(self, k) for k in OBSERVABLE_VARIABLES], dtype=np.float64)

    def latent_array(self) -> np.ndarray:
        """Extract only the 4 latent state variables."""
        return np.array([getattr(self, k) for k in LATENT_VARIABLES], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> StateVector:
        """Construct StateVector from 12-element numpy array."""
        if len(arr) != 12:
            raise ValueError(f"Expected 12 elements for StateVector, got {len(arr)}")
        kwargs = {k: float(arr[i]) for i, k in enumerate(ALL_STATE_VARIABLES)}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StateVector:
        """Construct StateVector from dictionary, validating all keys."""
        missing = [k for k in ALL_STATE_VARIABLES if k not in d]
        if missing:
            raise KeyError(f"Missing state variables in dict: {missing}")
        return cls(**{k: float(d[k]) for k in ALL_STATE_VARIABLES})

    def validate_bounds(self) -> Tuple[bool, List[str]]:
        """Check if all state variables fall within safety physical bounds."""
        violations = []
        for var_name, (low, high) in STATE_BOUNDS.items():
            val = getattr(self, var_name)
            if val < low or val > high:
                violations.append(f"{var_name}={val:.2f} outside [{low}, {high}]")
        return len(violations) == 0, violations
