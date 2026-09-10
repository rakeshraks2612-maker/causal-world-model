"""Central System Parameters and Configuration for THC-SCM System.

Loads physics constants, stochastic noise levels, sensor calibration, and safety thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class PhysicsConstants:
    delta_t: float = 1.0
    p_idle: float = 0.35
    beta_l: float = 3.8
    p_elec_exp: float = 1.25
    eta_elec: float = 0.92
    alpha_amb: float = 0.015
    q_hotspot_ar: float = 0.92
    q_hotspot_mean: float = 0.20
    tau_valve_base: float = 2.5
    valve_wear_lag_scale: float = 1.5
    valve_flow_exp: float = 1.35
    p_base: float = 1.8
    pump_stage_scale: float = 0.35
    pump_stage_offset: float = 0.60
    h_nominal: float = 2.5
    f_max: float = 55.0
    k_pressure: float = 1.6
    delta_leak_pressure: float = 0.7
    wear_flow_loss: float = 0.35
    flush_flow_surge: float = 12.0
    gamma_vib_base: float = 1.0
    gamma_vib_pump: float = 0.8
    gamma_vib_press: float = 0.9
    gamma_vib_wear: float = 2.2
    c_core: float = 0.085
    c_loop: float = 0.45
    k_transfer_base: float = 0.12
    k_transfer_wear_loss: float = 0.45
    f_scale: float = 14.0
    k_ambient_loss: float = 0.005
    k_radiator: float = 0.065
    flush_temp_drop: float = 4.5
    lambda_wear: float = 0.0001
    t_wear_threshold: float = 75.0
    flush_wear_relief: float = 0.03
    lambda_leak: float = 0.005
    p_leak_threshold: float = 4.2


@dataclass
class StochasticConstants:
    theta_amb: float = 0.02
    mu_amb: float = 27.0
    sigma_amb: float = 0.15
    sigma_p_elec: float = 0.02
    sigma_q: float = 0.04
    sigma_q_int: float = 0.02
    sigma_v: float = 0.10
    sigma_p: float = 0.02
    sigma_f: float = 0.15
    sigma_vib: float = 0.08
    sigma_t_core: float = 0.08
    sigma_t_cool: float = 0.05
    sigma_w: float = 0.00005
    sigma_xi: float = 0.002

    def to_sigmas_dict(self) -> Dict[str, float]:
        return {
            "sigma_p_elec": self.sigma_p_elec,
            "sigma_q": self.sigma_q,
            "sigma_q_int": self.sigma_q_int,
            "sigma_v": self.sigma_v,
            "sigma_p": self.sigma_p,
            "sigma_f": self.sigma_f,
            "sigma_vib": self.sigma_vib,
            "sigma_t_core": self.sigma_t_core,
            "sigma_t_cool": self.sigma_t_cool,
            "sigma_w": self.sigma_w,
            "sigma_xi": self.sigma_xi,
        }


@dataclass
class SensorNoiseConstants:
    noise_t_core: float = 0.40
    noise_t_cool: float = 0.30
    noise_p_sys: float = 0.05
    noise_f_cool: float = 0.20
    noise_l_cpu: float = 0.50
    noise_v_pos: float = 0.20
    noise_vib_pump: float = 0.15
    noise_p_elec: float = 0.02

    def to_dict(self) -> Dict[str, float]:
        return {
            "T_core": self.noise_t_core,
            "T_cool": self.noise_t_cool,
            "P_sys": self.noise_p_sys,
            "F_cool": self.noise_f_cool,
            "L_cpu": self.noise_l_cpu,
            "V_pos": self.noise_v_pos,
            "Vib_pump": self.noise_vib_pump,
            "P_elec": self.noise_p_elec,
        }


@dataclass
class SafetyThresholds:
    t_core_runaway_thresh: float = 105.0
    t_core_runaway_duration: int = 3
    p_sys_overpressure_thresh: float = 5.5
    f_cool_cavitation_thresh: float = 2.0
    v_pos_cavitation_thresh: float = 50.0
    a_pump_cavitation_thresh: int = 2


@dataclass
class SystemConstants:
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    stochastic: StochasticConstants = field(default_factory=StochasticConstants)
    sensors: SensorNoiseConstants = field(default_factory=SensorNoiseConstants)
    safety: SafetyThresholds = field(default_factory=SafetyThresholds)
    episode_length: int = 120

    @classmethod
    def load_default(cls) -> SystemConstants:
        """Load from default YAML config if available, else instantiate default dataclass."""
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "thc_default.yaml"
        if config_path.exists():
            return cls.from_yaml(config_path)
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> SystemConstants:
        """Parse configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        phys_data = data.get("physics", {})
        stoch_data = data.get("stochastic", {})
        sensor_data = data.get("sensors", {})
        safety_data = data.get("safety", {})
        proto_data = data.get("protocol", {})

        return cls(
            physics=PhysicsConstants(**phys_data),
            stochastic=StochasticConstants(**stoch_data),
            sensors=SensorNoiseConstants(**sensor_data),
            safety=SafetyThresholds(**safety_data),
            episode_length=proto_data.get("episode_length", 120),
        )
