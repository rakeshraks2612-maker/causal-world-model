"""Intervention-Aware Training Dataset Generator (Task 3.4D-B).

Generates rich training distributions containing previously missing physics:
1. Nominal closed-loop feedback regimes (preserving nominal accuracy)
2. Sustained high-load operations (L_cpu in [70, 90]%, P_elec in [2.8, 3.6] kW)
3. Thermal runaway failure trajectories (T_core >= 105°C for >= 3s)
4. Recovery trajectories (thermal alarm -> emergency valve open/throttle down -> safe cooling)
5. Open-loop state and action step interventions (do(L_cpu), do(V_pos), do(Vib_pump))

Maintains strict two-tier data contracts:
- Learner format: data/intervention_aware/learner/{split}/ep_XXXXX.npz (Zero ground truth leakage)
- Oracle format: data/intervention_aware/oracle/{split}/ep_XXXXX.npz (Complete ground truth for evaluation)
"""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.validators import validate_oracle_episode_integrity, validate_learner_isolation
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.policies import (
    BasePolicy,
    NominalController,
    HighLoadController,
    AggressiveCoolingController,
    FailureInducingController,
    RecoveryController,
)


class InterventionAwareRegime(str, Enum):
    """Regimes in the intervention-aware training distribution."""
    NOMINAL_PID = "nominal_pid"
    SUSTAINED_HIGH_LOAD = "sustained_high_load"
    THERMAL_RUNAWAY = "thermal_runaway"
    RECOVERY_SUPERVISION = "recovery_supervision"
    STEP_INTERVENTIONS = "step_interventions"


class StepInterventionController(BasePolicy):
    """Executes nominal control until t*, then steps action or holds uncompensated load."""

    def __init__(
        self,
        t_star: int = 30,
        pre_policy: Optional[BasePolicy] = None,
        post_valve: float = 30.0,
        post_throttle: float = 100.0,
        post_pump: int = 2,
    ) -> None:
        self.t_star = t_star
        self.pre_policy = pre_policy or NominalController()
        self.post_valve = post_valve
        self.post_throttle = post_throttle
        self.post_pump = post_pump

    def reset(self) -> None:
        self.pre_policy.reset()

    def select_action(
        self,
        step: int,
        obs: Any,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        if step < self.t_star:
            return self.pre_policy.select_action(step, obs, prev_action)
        return ActionVector(
            A_valve=float(self.post_valve),
            A_throttle=float(self.post_throttle),
            A_pump=self.post_pump,
            A_flush=0,
        )


def sample_intervention_aware_initial_state(
    regime: InterventionAwareRegime,
    rng: np.random.Generator,
) -> Tuple[StateVector, BasePolicy, Optional[InterventionRegistry]]:
    """Sample physical initial states, controllers, and intervention registries."""
    inv_reg: Optional[InterventionRegistry] = None

    if regime == InterventionAwareRegime.NOMINAL_PID:
        init_state = StateVector(
            T_core=float(rng.normal(68.5, 1.5)),
            T_cool=float(rng.normal(34.0, 1.0)),
            P_sys=float(rng.normal(3.15, 0.08)),
            F_cool=float(rng.normal(32.0, 1.2)),
            L_cpu=float(rng.normal(50.0, 4.0)),
            V_pos=float(rng.normal(55.0, 3.0)),
            Vib_pump=float(rng.normal(4.2, 0.2)),
            P_elec=float(rng.normal(1.85, 0.08)),
            T_amb=float(rng.normal(25.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=float(rng.normal(68.0, 1.0)))

    elif regime == InterventionAwareRegime.SUSTAINED_HIGH_LOAD:
        # L_cpu in [70, 90]% sustained via graph intervention with fixed moderate cooling valve
        load_val = float(rng.uniform(70.0, 90.0))
        valve_val = float(rng.uniform(35.0, 55.0))
        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention(target="L_cpu", value=load_val, start_step=0))
        init_state = StateVector(
            T_core=float(rng.normal(74.0, 2.0)),
            T_cool=float(rng.normal(38.0, 1.5)),
            P_sys=float(rng.normal(3.20, 0.10)),
            F_cool=float(rng.normal(26.0, 2.0)),
            L_cpu=load_val,
            V_pos=valve_val,
            Vib_pump=float(rng.normal(4.5, 0.3)),
            P_elec=float(0.04 * load_val + rng.normal(0.0, 0.05)),
            T_amb=float(rng.normal(26.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.20)),
            Q_internal=float(rng.normal(0.20, 0.04)),
            xi_leak=0.0,
        )
        policy = HighLoadController(fixed_valve=valve_val, pump_stage=2)

    elif regime == InterventionAwareRegime.THERMAL_RUNAWAY:
        # High load [75, 95]% combined with starved valve (8-22%) forcing T_core >= 105°C
        load_val = float(rng.uniform(75.0, 95.0))
        starved_valve = float(rng.uniform(6.0, 22.0))
        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention(target="L_cpu", value=load_val, start_step=0))
        init_state = StateVector(
            T_core=float(rng.normal(82.0, 2.5)),
            T_cool=float(rng.normal(42.0, 2.0)),
            P_sys=float(rng.normal(2.95, 0.10)),
            F_cool=float(rng.normal(12.0, 1.5)),
            L_cpu=load_val,
            V_pos=starved_valve,
            Vib_pump=float(rng.normal(4.8, 0.3)),
            P_elec=float(0.04 * load_val + rng.normal(0.0, 0.05)),
            T_amb=float(rng.normal(27.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.10, 0.03), 0.0, 0.25)),
            Q_internal=float(rng.normal(0.25, 0.05)),
            xi_leak=0.0,
        )
        policy = FailureInducingController()

    elif regime == InterventionAwareRegime.RECOVERY_SUPERVISION:
        # Begins in high thermal stress with load in [75, 90]%, RecoveryController responds at T_core >= 80/92°C
        load_val = float(rng.uniform(75.0, 90.0))
        inv_reg = InterventionRegistry()
        # High load until t=40, then controller throttles or natural demand takes over
        inv_reg.add(Intervention(target="L_cpu", value=load_val, start_step=0, duration=40))
        init_state = StateVector(
            T_core=float(rng.normal(84.0, 2.0)),
            T_cool=float(rng.normal(40.0, 1.5)),
            P_sys=float(rng.normal(3.10, 0.10)),
            F_cool=float(rng.normal(20.0, 2.0)),
            L_cpu=load_val,
            V_pos=float(rng.uniform(30.0, 45.0)),
            Vib_pump=float(rng.normal(4.6, 0.3)),
            P_elec=float(rng.normal(3.2, 0.1)),
            T_amb=float(rng.normal(26.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.20)),
            Q_internal=float(rng.normal(0.22, 0.04)),
            xi_leak=0.0,
        )
        policy = RecoveryController()

    elif regime == InterventionAwareRegime.STEP_INTERVENTIONS:
        # Interventions applied mid-flight
        t_star = int(rng.choice([20, 30, 40, 50]))
        inv_type = rng.choice(["do_L_cpu", "do_V_pos", "step_action_valve", "step_throttle"])

        init_state = StateVector(
            T_core=float(rng.normal(69.0, 1.5)),
            T_cool=float(rng.normal(34.5, 1.0)),
            P_sys=float(rng.normal(3.15, 0.08)),
            F_cool=float(rng.normal(30.0, 1.5)),
            L_cpu=float(rng.normal(55.0, 3.0)),
            V_pos=float(rng.normal(52.0, 2.0)),
            Vib_pump=float(rng.normal(4.3, 0.2)),
            P_elec=float(rng.normal(2.0, 0.08)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.15)),
            Q_internal=float(rng.normal(0.16, 0.03)),
            xi_leak=0.0,
        )

        if inv_type == "do_L_cpu":
            target_val = float(rng.choice([75.0, 80.0, 85.0, 90.0]))
            inv_reg = InterventionRegistry()
            inv_reg.add(Intervention(target="L_cpu", value=target_val, start_step=t_star))
            policy = NominalController()
        elif inv_type == "do_V_pos":
            target_val = float(rng.choice([15.0, 25.0, 75.0, 90.0]))
            inv_reg = InterventionRegistry()
            inv_reg.add(Intervention(target="V_pos", value=target_val, start_step=t_star))
            policy = NominalController()
        elif inv_type == "step_action_valve":
            post_v = float(rng.choice([10.0, 20.0, 80.0, 95.0]))
            policy = StepInterventionController(t_star=t_star, post_valve=post_v)
        else:
            post_thr = float(rng.choice([20.0, 40.0]))
            policy = StepInterventionController(t_star=t_star, post_throttle=post_thr)

    else:
        raise ValueError(f"Unknown regime: {regime}")

    return init_state, policy, inv_reg


def generate_intervention_aware_episode(
    split: SplitType,
    index: int,
    regime: InterventionAwareRegime,
    length: int = 120,
    seed_offset: int = 500000,
) -> OracleEpisode:
    """Generate a single intervention-aware episode with dedicated training seed space."""
    # Seed offset ensures zero overlap with pilot benchmark seeds
    seed = seed_offset + index * 13 + hash(regime.value) % 10000
    ep_id = f"ep_{split.value}_int_{index:05d}_{regime.value}"

    init_rng = np.random.default_rng(seed)
    init_state, policy, inv_reg = sample_intervention_aware_initial_state(regime, init_rng)

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
            "policy": policy.__class__.__name__,
            "seed": seed,
            "length": length,
            "has_interventions": inv_reg is not None and len(inv_reg.interventions) > 0,
            "initial_t_amb": float(init_state.T_amb),
            "initial_w_wear": float(init_state.W_wear),
            "initial_q_internal": float(init_state.Q_internal),
        },
    )

    validate_oracle_episode_integrity(oracle_ep)
    return oracle_ep


def generate_intervention_aware_dataset(
    output_dir: str | Path = "data/intervention_aware",
    train_count: int = 100,
    val_count: int = 20,
    length: int = 120,
) -> Dict[str, Any]:
    """Generate complete intervention-aware Train and Validation splits."""
    base_path = Path(output_dir)
    oracle_train_dir = base_path / "oracle" / "train"
    learner_train_dir = base_path / "learner" / "train"
    oracle_val_dir = base_path / "oracle" / "validation"
    learner_val_dir = base_path / "learner" / "validation"

    for d in [oracle_train_dir, learner_train_dir, oracle_val_dir, learner_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Regime mix proportions
    regimes_mix = [
        (InterventionAwareRegime.NOMINAL_PID, 0.30),
        (InterventionAwareRegime.SUSTAINED_HIGH_LOAD, 0.25),
        (InterventionAwareRegime.THERMAL_RUNAWAY, 0.20),
        (InterventionAwareRegime.RECOVERY_SUPERVISION, 0.15),
        (InterventionAwareRegime.STEP_INTERVENTIONS, 0.10),
    ]

    # Generate Train Split
    train_oracle_eps: List[OracleEpisode] = []
    train_learner_eps: List[LearnerEpisode] = []
    idx = 0
    for regime, prop in regimes_mix:
        n_regime = int(round(train_count * prop))
        for _ in range(n_regime):
            orc_ep = generate_intervention_aware_episode(SplitType.TRAIN, idx, regime, length=length, seed_offset=500000)
            learn_ep = orc_ep.to_learner_episode()
            validate_learner_isolation(learn_ep)

            orc_ep.save_npz(oracle_train_dir / f"{orc_ep.episode_id}.npz")
            learn_ep.save_npz(learner_train_dir / f"{learn_ep.episode_id}.npz")

            train_oracle_eps.append(orc_ep)
            train_learner_eps.append(learn_ep)
            idx += 1

    # Generate Validation Split
    val_oracle_eps: List[OracleEpisode] = []
    val_learner_eps: List[LearnerEpisode] = []
    idx = 0
    for regime, prop in regimes_mix:
        n_regime = int(round(val_count * prop))
        for _ in range(n_regime):
            orc_ep = generate_intervention_aware_episode(SplitType.VAL, idx, regime, length=length, seed_offset=700000)
            learn_ep = orc_ep.to_learner_episode()
            validate_learner_isolation(learn_ep)

            orc_ep.save_npz(oracle_val_dir / f"{orc_ep.episode_id}.npz")
            learn_ep.save_npz(learner_val_dir / f"{learn_ep.episode_id}.npz")

            val_oracle_eps.append(orc_ep)
            val_learner_eps.append(learn_ep)
            idx += 1

    return {
        "train_count": len(train_oracle_eps),
        "val_count": len(val_oracle_eps),
        "total_train_steps": len(train_oracle_eps) * (length + 1),
        "total_val_steps": len(val_oracle_eps) * (length + 1),
    }
