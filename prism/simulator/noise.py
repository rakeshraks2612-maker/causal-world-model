"""Exogenous Stochastic Process and Deterministic RNG Manager for THC-SCM.

Handles:
- Exogenous noise vector U_t
- Ornstein-Uhlenbeck continuous ambient disturbance
- Dedicated independent RNG streams for bit-for-bit reproducible twin-simulation
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import numpy as np

NOISE_VARIABLES: List[str] = [
    "u_amb",
    "u_p_elec",
    "u_q",
    "u_q_int",
    "u_v",
    "u_p",
    "u_f",
    "u_vib",
    "u_t_core",
    "u_t_cool",
    "u_w",
    "u_xi",
]


@dataclass(frozen=True)
class NoiseVector:
    """Represents the realized exogenous disturbance innovations at step t."""

    u_amb: float
    u_p_elec: float
    u_q: float
    u_q_int: float
    u_v: float
    u_p: float
    u_f: float
    u_vib: float
    u_t_core: float
    u_t_cool: float
    u_w: float
    u_xi: float

    @classmethod
    def zeros(cls) -> NoiseVector:
        """Zero noise vector for deterministic ablation studies."""
        return cls(**{k: 0.0 for k in NOISE_VARIABLES})

    def to_array(self) -> np.ndarray:
        """Convert noise vector to 12-element numpy array."""
        return np.array([getattr(self, k) for k in NOISE_VARIABLES], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> NoiseVector:
        """Construct NoiseVector from 12-element numpy array."""
        if len(arr) != 12:
            raise ValueError(f"Expected 12 elements for NoiseVector, got {len(arr)}")
        return cls(**{k: float(arr[i]) for i, k in enumerate(NOISE_VARIABLES)})

    def to_dict(self) -> Dict[str, float]:
        """Convert noise vector to dictionary."""
        return asdict(self)


class RNGManager:
    """Manages isolated, deterministic pseudo-random number generator streams.
    
    Uses NumPy's SeedSequence to spawn independent random generators for:
    - Ambient environmental process
    - Physical process dynamics innovations
    - Observation sensor noise
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.seed_seq = np.random.SeedSequence(self.seed)
        # Spawn 3 distinct independent RNG streams
        child_seeds = self.seed_seq.spawn(3)
        self.rng_ambient = np.random.default_rng(child_seeds[0])
        self.rng_dynamics = np.random.default_rng(child_seeds[1])
        self.rng_sensors = np.random.default_rng(child_seeds[2])

    def sample_ambient_step(
        self,
        current_t_amb: float,
        theta_amb: float,
        mu_amb: float,
        sigma_amb: float,
        delta_t: float,
    ) -> Tuple[float, float]:
        """Perform one step of Ornstein-Uhlenbeck ambient temperature process.
        
        Returns:
            (next_t_amb, u_amb_realized)
        """
        drift = theta_amb * (mu_amb - current_t_amb) * delta_t
        diffusion_std = sigma_amb * np.sqrt(delta_t)
        zeta = float(self.rng_ambient.normal(0.0, 1.0))
        u_amb = float(diffusion_std * zeta)
        next_t_amb = float(current_t_amb + drift + u_amb)
        return next_t_amb, u_amb

    def sample_process_innovations(self, sigmas: Dict[str, float]) -> NoiseVector:
        """Sample all Gaussian physical innovations for step t."""
        return NoiseVector(
            u_amb=0.0,  # Computed directly via sample_ambient_step
            u_p_elec=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_p_elec", 0.02))),
            u_q=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_q", 0.04))),
            u_q_int=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_q_int", 0.02))),
            u_v=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_v", 0.10))),
            u_p=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_p", 0.02))),
            u_f=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_f", 0.15))),
            u_vib=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_vib", 0.08))),
            u_t_core=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_t_core", 0.08))),
            u_t_cool=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_t_cool", 0.05))),
            u_w=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_w", 0.00005))),
            u_xi=float(self.rng_dynamics.normal(0.0, sigmas.get("sigma_xi", 0.002))),
        )
