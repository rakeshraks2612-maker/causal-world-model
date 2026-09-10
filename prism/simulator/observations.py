"""Observation Schema and Sensor Model for THC-SCM System.

Exposes only the 8 observable variables with sensor noise and telemetry masking.
Latent variables (T_amb, W_wear, Q_internal, xi_leak) are strictly excluded from observations.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Any
import numpy as np
from prism.simulator.state import StateVector, OBSERVABLE_VARIABLES


@dataclass(frozen=True)
class ObservationVector:
    """Represents noisy, partial telemetry received by a monitoring agent at step t.
    
    Attributes:
        T_core: Noisy core temperature (°C) or NaN if missing
        T_cool: Noisy coolant temperature (°C) or NaN if missing
        P_sys: Noisy loop pressure (bar) or NaN if missing
        F_cool: Noisy flow rate (L/min) or NaN if missing
        L_cpu: Noisy CPU utilization (%) or NaN if missing
        V_pos: Noisy valve position (%) or NaN if missing
        Vib_pump: Noisy vibration (mm/s) or NaN if missing
        P_elec: Noisy electrical power (kW) or NaN if missing
        missing_channels: Set of channel names that are unobserved/masked
    """

    T_core: float
    T_cool: float
    P_sys: float
    F_cool: float
    L_cpu: float
    V_pos: float
    Vib_pump: float
    P_elec: float
    missing_channels: Set[str]

    def to_array(self) -> np.ndarray:
        """Return 8-element numpy float64 array (NaN for missing channels)."""
        return np.array([getattr(self, k) for k in OBSERVABLE_VARIABLES], dtype=np.float64)

    def is_missing(self, var_name: str) -> bool:
        """Check if a specific sensor telemetry channel is missing."""
        return var_name in self.missing_channels or np.isnan(getattr(self, var_name))

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        d = asdict(self)
        d["missing_channels"] = list(self.missing_channels)
        return d

    @classmethod
    def from_state(
        cls,
        state: StateVector,
        sensor_noise: Optional[Dict[str, float]] = None,
        missing_channels: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ObservationVector:
        """Generate a realistic observation from ground truth state.
        
        Args:
            state: True 12-dimensional StateVector
            sensor_noise: Mapping of standard deviations per observable sensor
            missing_channels: Set of observable variable names to mask with NaN
            rng: NumPy Random Generator for repeatable noise sampling
        """
        missing = set(missing_channels) if missing_channels is not None else set()

        if sensor_noise is None:
            # Default sensor noise levels from specification
            sensor_noise = {
                "T_core": 0.40,
                "T_cool": 0.30,
                "P_sys": 0.05,
                "F_cool": 0.20,
                "L_cpu": 0.50,
                "V_pos": 0.20,
                "Vib_pump": 0.15,
                "P_elec": 0.02,
            }

        obs_dict: Dict[str, float] = {}
        for var_name in OBSERVABLE_VARIABLES:
            if var_name in missing:
                obs_dict[var_name] = float("nan")
            else:
                true_val = getattr(state, var_name)
                sigma = sensor_noise.get(var_name, 0.0)
                noise_val = float(rng.normal(0.0, sigma)) if (rng is not None and sigma > 0.0) else 0.0
                obs_dict[var_name] = float(true_val + noise_val)

        return cls(
            T_core=obs_dict["T_core"],
            T_cool=obs_dict["T_cool"],
            P_sys=obs_dict["P_sys"],
            F_cool=obs_dict["F_cool"],
            L_cpu=obs_dict["L_cpu"],
            V_pos=obs_dict["V_pos"],
            Vib_pump=obs_dict["Vib_pump"],
            P_elec=obs_dict["P_elec"],
            missing_channels=missing,
        )
