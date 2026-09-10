"""OOD Regimes and Confounding Experimental Scenario Generators.

Implements:
- Task 2.5: 5 Explicit Out-Of-Distribution (OOD) Stress Regimes:
    1. OOD-1: Extreme Ambient Heatwave (T_amb in [38.0, 46.0]°C) [Environmental Extrapolation]
    2. OOD-2: Extreme Mechanical & Interface Wear (W_wear in [0.80, 0.95]) [Latent Degradation]
    3. OOD-3: Combined Multi-Factorial Stress (L > 85% + T_amb > 38°C + W_wear > 0.75) [Compositional Joint OOD]
    4. OOD-4: Latent Silicon Hotspot Flux (Q_internal in [1.20, 2.50] kW) [Severe Latent-Cause Extrapolation]
    5. OOD-5: Rapid Hydraulic Seal Degradation (xi_leak in [25.0, 48.0] mL/hr) [Severe Latent-Cause Extrapolation]

- Task 2.6: Confounding Experimental Protocol (2x3 Factorial Matched Grid):
    Accurately maps the Structural Causal Model (SCM):
    
                    [ Ambient Temp (T_amb) ] (Hidden Confounder)
                          /          \
                         ▼            ▼
                 [ L_target ]    [ Q_radiator ]
                      │                │
                      ▼                ▼
                   [ L_cpu ]       [ T_cool ]
                      │                │
                      ▼                │
                  [ P_elec ]           │
                      │                │
                      ▼                ▼
                   [ Q_in ] ──────► [ T_core ] ◄───── [ Latent Hotspot (Q_int) ]
                                       │
                                       ▼
                                [ Failure Risk ]
                                
    Separately, the hydraulic and acoustic paths operate via:
    V_pos -> F_cool -> P_sys -> Vib_pump, and F_cool -> Vib_pump
    
    The 2x3 Factorial Grid evaluates:
    Seasons: {Summer Peak (T_amb ~ 38°C), Winter Baseline (T_amb ~ 20°C)}
    Interventions: {Observational Natural, do(L_cpu = 20%), do(L_cpu = 80%)}
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.splits import get_partition_seed, generate_episode_id
from prism.dataset.validators import validate_oracle_episode_integrity, validate_learner_isolation
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.policies import NominalController, HighLoadController, BasePolicy


class OODRegime(str, Enum):
    """The 5 explicit benchmark Out-Of-Distribution regimes."""
    EXTREME_AMBIENT = "ood_1_extreme_ambient"
    EXTREME_WEAR = "ood_2_extreme_wear"
    COMBINED_STRESS = "ood_3_combined_stress"
    LATENT_HOTSPOT = "ood_4_latent_hotspot"
    LEAK_DEGRADATION = "ood_5_leak_degradation"


def sample_ood_initial_state(
    regime: OODRegime,
    rng: np.random.Generator,
) -> Tuple[StateVector, BasePolicy]:
    """Sample initial physical states specifically pushed outside the training support."""
    if regime == OODRegime.EXTREME_AMBIENT:
        # Severe ambient facility heatwave (38.0 - 46.0°C vs training max 30°C)
        t_amb_val = float(rng.uniform(38.0, 46.0))
        init_state = StateVector(
            T_core=float(rng.normal(76.0 + (t_amb_val - 38.0) * 0.8, 1.5)),
            T_cool=float(rng.normal(46.0 + (t_amb_val - 38.0) * 0.7, 1.2)),
            P_sys=float(rng.normal(3.20, 0.08)),
            F_cool=float(rng.normal(30.0, 1.5)),
            L_cpu=float(rng.normal(65.0, 3.0)),
            V_pos=float(rng.normal(65.0, 2.0)),
            Vib_pump=float(rng.normal(4.6, 0.2)),
            P_elec=float(rng.normal(2.2, 0.1)),
            T_amb=t_amb_val,
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.20)),
            Q_internal=float(rng.normal(0.18, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=68.0)

    elif regime == OODRegime.EXTREME_WEAR:
        # Severe actuator stiction and thermal interface degradation (W_wear 0.80 - 0.95 vs train max 0.25)
        wear_val = float(rng.uniform(0.80, 0.95))
        init_state = StateVector(
            T_core=float(rng.normal(78.0, 2.0)),
            T_cool=float(rng.normal(38.0, 1.5)),
            P_sys=float(rng.normal(2.95, 0.10)),
            F_cool=float(rng.normal(20.0, 1.5)),  # Flow heavily attenuated by wear
            L_cpu=float(rng.normal(50.0, 3.0)),
            V_pos=float(rng.normal(50.0, 2.0)),
            Vib_pump=float(rng.normal(6.5, 0.3)),  # High vibration from mechanical wear
            P_elec=float(rng.normal(1.85, 0.05)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=wear_val,
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=68.0)

    elif regime == OODRegime.COMBINED_STRESS:
        # High Workload (85-95%) + High Ambient (38-42°C) + High Wear (0.75-0.90)
        t_amb_val = float(rng.uniform(38.0, 42.0))
        wear_val = float(rng.uniform(0.75, 0.90))
        init_state = StateVector(
            T_core=float(rng.normal(84.0, 2.0)),
            T_cool=float(rng.normal(52.0, 1.5)),
            P_sys=float(rng.normal(3.35, 0.10)),
            F_cool=float(rng.normal(22.0, 1.5)),
            L_cpu=float(rng.uniform(85.0, 95.0)),
            V_pos=float(rng.normal(85.0, 3.0)),
            Vib_pump=float(rng.normal(7.8, 0.4)),
            P_elec=float(rng.normal(2.9, 0.1)),
            T_amb=t_amb_val,
            W_wear=wear_val,
            Q_internal=float(rng.normal(0.25, 0.04)),
            xi_leak=0.0,
        )
        policy = HighLoadController(fixed_valve=75.0, pump_stage=3)

    elif regime == OODRegime.LATENT_HOTSPOT:
        # Severe unobserved localized silicon hotspot flux (1.20 - 2.50 kW vs train max 0.35 kW)
        q_int_val = float(rng.uniform(1.20, 2.50))
        init_state = StateVector(
            T_core=float(rng.normal(82.0, 2.0)),
            T_cool=float(rng.normal(40.0, 1.5)),
            P_sys=float(rng.normal(3.15, 0.08)),
            F_cool=float(rng.normal(32.0, 1.5)),
            L_cpu=float(rng.normal(50.0, 3.0)),
            V_pos=float(rng.normal(60.0, 2.0)),
            Vib_pump=float(rng.normal(4.3, 0.2)),
            P_elec=float(rng.normal(1.85, 0.05)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.05, 0.01), 0.0, 0.15)),
            Q_internal=q_int_val,
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=68.0)

    elif regime == OODRegime.LEAK_DEGRADATION:
        # Rapid seal micro-leakage (25.0 - 48.0 mL/hr vs train ~0)
        leak_val = float(rng.uniform(25.0, 48.0))
        init_state = StateVector(
            T_core=float(rng.normal(74.0, 2.0)),
            T_cool=float(rng.normal(36.0, 1.5)),
            P_sys=float(rng.normal(2.40, 0.12)),  # Depressurized loop
            F_cool=float(rng.normal(22.0, 2.0)),  # Flow loss
            L_cpu=float(rng.normal(50.0, 3.0)),
            V_pos=float(rng.normal(65.0, 2.0)),
            Vib_pump=float(rng.normal(4.2, 0.2)),
            P_elec=float(rng.normal(1.85, 0.05)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.15)),
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=leak_val,
        )
        policy = NominalController(target_temp=68.0)

    else:
        raise ValueError(f"Unknown OOD regime: {regime}")

    return init_state, policy


def generate_ood_episode(
    regime: OODRegime,
    index: int,
    length: int = 120,
) -> OracleEpisode:
    """Generate an OracleEpisode for a specific OOD regime."""
    split = SplitType.OOD
    seed = get_partition_seed(split, index)
    ep_id = f"ep_{regime.value}_{index:05d}"

    init_rng = np.random.default_rng(seed)
    init_state, policy = sample_ood_initial_state(regime, init_rng)

    # For sustained heatwave regimes (OOD-1, OOD-3), maintain high ambient across the episode
    inv_reg = None
    if regime in (OODRegime.EXTREME_AMBIENT, OODRegime.COMBINED_STRESS):
        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention("T_amb", float(init_state.T_amb), start_step=0))

    sim = THCSimulator(seed=seed)
    raw_ep = sim.run_episode(
        policy=policy,
        length=length,
        initial_state=init_state,
        interventions=inv_reg,
        episode_id=ep_id,
    )

    mask = ~np.isnan(raw_ep.observations)

    oracle_ep = OracleEpisode(
        episode_id=ep_id,
        seed=seed,
        split=split,
        timestamps=raw_ep.timestamps,
        observations=raw_ep.observations,
        observation_mask=mask,
        actions=raw_ep.actions,
        ground_truth_states=raw_ep.ground_truth_states,
        exogenous_noise=raw_ep.exogenous_noise,
        failure_latched=raw_ep.failure_latched,
        failure_mode=raw_ep.failure_mode,
        failure_timestamp=raw_ep.failure_timestamp,
        oracle_metadata={
            "regime": regime.value,
            "ood_type": regime.value,
            "policy": policy.__class__.__name__,
            "seed": seed,
            "length": length,
            "initial_t_amb": float(init_state.T_amb),
            "initial_w_wear": float(init_state.W_wear),
            "initial_q_internal": float(init_state.Q_internal),
            "initial_xi_leak": float(init_state.xi_leak),
        },
    )

    validate_oracle_episode_integrity(oracle_ep)
    return oracle_ep


def generate_confounding_episode(
    regime: str,  # "summer_peak" (high T_amb) or "winter_baseline" (low T_amb)
    index: int,
    intervention: Optional[Intervention] = None,
    length: int = 120,
) -> OracleEpisode:
    """Task 2.6: Generate an episode specifically for the confounding evaluation protocol.
    
    In observational summer_peak, T_amb is high (36-40°C), driving both natural workload L_cpu ~ 75-85%
    and high coolant temperature T_cool, producing elevated core temp and high pump vibration.
    
    By applying an intervention do(L_cpu = 20%) or do(T_amb = 25°C), we isolate whether a learned model
    correctly understands that ambient heat is the confounding root cause.
    """
    split = SplitType.TEST  # Confounding evaluation partition
    seed = 350_000 + index
    inv_tag = f"_do_{intervention.target}" if intervention else "_obs"
    ep_id = f"ep_confound_{regime}{inv_tag}_{index:05d}"

    rng = np.random.default_rng(seed)

    if regime == "summer_peak":
        t_amb = float(rng.uniform(36.0, 41.0))
        init_state = StateVector(
            T_core=float(rng.normal(75.0, 1.5)),
            T_cool=float(rng.normal(44.0, 1.2)),
            P_sys=float(rng.normal(3.20, 0.08)),
            F_cool=float(rng.normal(30.0, 1.5)),
            L_cpu=float(rng.normal(35.0 + 1.2 * t_amb, 2.0)),  # Natural coupling L ~ 80%
            V_pos=float(rng.normal(65.0, 2.0)),
            Vib_pump=float(rng.normal(5.0, 0.2)),
            P_elec=float(rng.normal(2.5, 0.1)),
            T_amb=t_amb,
            W_wear=0.08,
            Q_internal=0.18,
            xi_leak=0.0,
        )
    else:  # winter_baseline
        t_amb = float(rng.uniform(18.0, 22.0))
        init_state = StateVector(
            T_core=float(rng.normal(64.0, 1.5)),
            T_cool=float(rng.normal(28.0, 1.2)),
            P_sys=float(rng.normal(3.10, 0.08)),
            F_cool=float(rng.normal(32.0, 1.5)),
            L_cpu=float(rng.normal(35.0 + 1.2 * t_amb, 2.0)),  # Natural coupling L ~ 58%
            V_pos=float(rng.normal(50.0, 2.0)),
            Vib_pump=float(rng.normal(4.0, 0.2)),
            P_elec=float(rng.normal(1.7, 0.08)),
            T_amb=t_amb,
            W_wear=0.05,
            Q_internal=0.15,
            xi_leak=0.0,
        )

    inv_registry = None
    if intervention is not None:
        inv_registry = InterventionRegistry()
        inv_registry.add(intervention)

    sim = THCSimulator(seed=seed)
    raw_ep = sim.run_episode(
        policy=NominalController(target_temp=68.0),
        length=length,
        initial_state=init_state,
        interventions=inv_registry,
        episode_id=ep_id,
    )

    mask = ~np.isnan(raw_ep.observations)

    oracle_ep = OracleEpisode(
        episode_id=ep_id,
        seed=seed,
        split=split,
        timestamps=raw_ep.timestamps,
        observations=raw_ep.observations,
        observation_mask=mask,
        actions=raw_ep.actions,
        ground_truth_states=raw_ep.ground_truth_states,
        exogenous_noise=raw_ep.exogenous_noise,
        failure_latched=raw_ep.failure_latched,
        failure_mode=raw_ep.failure_mode,
        failure_timestamp=raw_ep.failure_timestamp,
        oracle_metadata={
            "confounding_regime": regime,
            "is_intervened": intervention is not None,
            "intervention_target": intervention.target if intervention else None,
            "intervention_value": intervention.value if intervention else None,
            "initial_t_amb": float(init_state.T_amb),
        },
    )

    validate_oracle_episode_integrity(oracle_ep)
    return oracle_ep
