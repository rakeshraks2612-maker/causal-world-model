"""Action-Excitation Training Dataset Generator (Task 5.7C).

Generates rich, balanced training distributions covering all causal mechanisms:
1. Nominal PID (Inaction baseline, Scenario 1)
2. Valve bottleneck steps (Low valve -> Open valve, Scenario 2)
3. Throttle bottleneck steps (High load -> Throttle compute, Scenario 3)
4. Pump bottleneck steps (Low pump -> Pump stage 3/4, Scenario 4)
5. Orthogonal multi-channel excitation (Short holds 5-15 steps, zero confounding)
6. Structured pump staircase sweeps (1 <-> 2 <-> 3 <-> 4)
7. Thermal runaway failure trajectories
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
from prism.simulator.observations import ObservationVector
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.policies import (
    BasePolicy,
    NominalController,
    HighLoadController,
    AggressiveCoolingController,
    FailureInducingController,
    RecoveryController,
)


class ExcitationRegime(str, Enum):
    """Regimes in the action-excitation training distribution."""
    NOMINAL_PID = "nominal_pid"
    VALVE_BOTTLENECK_STEP = "valve_bottleneck_step"
    THROTTLE_BOTTLENECK_STEP = "throttle_bottleneck_step"
    PUMP_BOTTLENECK_STEP = "pump_bottleneck_step"
    ORTHOGONAL_EXCITATION = "orthogonal_excitation"
    STRUCTURED_PUMP_SWEEP = "structured_pump_sweep"
    THERMAL_RUNAWAY = "thermal_runaway"


class StepInterventionController(BasePolicy):
    """Executes baseline control until t*, then steps action on valve, pump, or throttle."""

    def __init__(
        self,
        t_star: int = 30,
        pre_valve: float = 50.0,
        pre_throttle: float = 100.0,
        pre_pump: int = 2,
        post_valve: float = 85.0,
        post_throttle: float = 100.0,
        post_pump: int = 3,
    ) -> None:
        self.t_star = t_star
        self.pre_valve = pre_valve
        self.pre_throttle = pre_throttle
        self.pre_pump = pre_pump
        self.post_valve = post_valve
        self.post_throttle = post_throttle
        self.post_pump = post_pump

    def reset(self) -> None:
        pass

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        if step < self.t_star:
            return ActionVector(
                A_valve=float(self.pre_valve),
                A_throttle=float(self.pre_throttle),
                A_pump=int(self.pre_pump),
                A_flush=0,
            )
        return ActionVector(
            A_valve=float(self.post_valve),
            A_throttle=float(self.post_throttle),
            A_pump=int(self.post_pump),
            A_flush=0,
        )


class OrthogonalExcitationController(BasePolicy):
    """Independently samples actions and short hold durations (5-15 steps) across all channels."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.pump_stages = [1, 2, 3, 4]
        self.valve_levels = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        self.throttle_levels = [30.0, 50.0, 70.0, 85.0, 100.0]

        self.current_pump = int(self.rng.choice(self.pump_stages))
        self.current_valve = float(self.rng.choice(self.valve_levels))
        self.current_throttle = float(self.rng.choice(self.throttle_levels))

        self.pump_timer = int(self.rng.integers(5, 16))
        self.valve_timer = int(self.rng.integers(5, 16))
        self.throttle_timer = int(self.rng.integers(5, 16))

    def reset(self) -> None:
        self.current_pump = int(self.rng.choice(self.pump_stages))
        self.current_valve = float(self.rng.choice(self.valve_levels))
        self.current_throttle = float(self.rng.choice(self.throttle_levels))
        self.pump_timer = int(self.rng.integers(5, 16))
        self.valve_timer = int(self.rng.integers(5, 16))
        self.throttle_timer = int(self.rng.integers(5, 16))

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        self.pump_timer -= 1
        if self.pump_timer <= 0:
            remaining_pumps = [p for p in self.pump_stages if p != self.current_pump]
            self.current_pump = int(self.rng.choice(remaining_pumps))
            self.pump_timer = int(self.rng.integers(5, 16))

        self.valve_timer -= 1
        if self.valve_timer <= 0:
            remaining_valves = [v for v in self.valve_levels if v != self.current_valve]
            self.current_valve = float(self.rng.choice(remaining_valves))
            self.valve_timer = int(self.rng.integers(5, 16))

        self.throttle_timer -= 1
        if self.throttle_timer <= 0:
            remaining_throttles = [th for th in self.throttle_levels if th != self.current_throttle]
            self.current_throttle = float(self.rng.choice(remaining_throttles))
            self.throttle_timer = int(self.rng.integers(5, 16))

        flush = 1 if (self.rng.random() < 0.03) else 0

        return ActionVector(
            A_valve=self.current_valve,
            A_throttle=self.current_throttle,
            A_pump=self.current_pump,
            A_flush=flush,
        )


class StructuredPumpSweepController(BasePolicy):
    """Executes systematic forward and backward pump staircase transitions (1->2->3->4->3->2->1)."""

    def __init__(self, fixed_valve: float = 50.0, fixed_throttle: float = 100.0, hold_steps: int = 10) -> None:
        self.fixed_valve = fixed_valve
        self.fixed_throttle = fixed_throttle
        self.hold_steps = hold_steps
        self.staircase = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 2, 1]

    def reset(self) -> None:
        pass

    def select_action(
        self,
        step: int,
        obs: ObservationVector,
        prev_action: Optional[ActionVector] = None,
    ) -> ActionVector:
        idx = (step // self.hold_steps) % len(self.staircase)
        pump = self.staircase[idx]
        return ActionVector(
            A_valve=float(self.fixed_valve),
            A_throttle=float(self.fixed_throttle),
            A_pump=int(pump),
            A_flush=0,
        )


def sample_excitation_initial_state(
    regime: ExcitationRegime,
    rng: np.random.Generator,
    seed: int,
) -> Tuple[StateVector, BasePolicy, Optional[InterventionRegistry]]:
    """Sample physical initial states, controllers, and intervention registries."""
    inv_reg: Optional[InterventionRegistry] = None

    if regime == ExcitationRegime.NOMINAL_PID:
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

    elif regime == ExcitationRegime.VALVE_BOTTLENECK_STEP:
        # Starved valve initial state -> valve opened at t*
        t_star = int(rng.choice([25, 30, 35]))
        starved_v = float(rng.choice([8.0, 15.0, 22.0]))
        post_v = float(rng.choice([15.0, 30.0, 60.0, 85.0, 95.0]))
        pump_stg = int(rng.choice([2, 3]))

        init_state = StateVector(
            T_core=float(rng.normal(92.0, 2.0)),
            T_cool=float(rng.normal(44.0, 1.5)),
            P_sys=float(rng.normal(2.5, 0.1)),
            F_cool=float(rng.normal(12.0, 1.5)),
            L_cpu=float(rng.normal(68.0, 3.0)),
            V_pos=starved_v,
            Vib_pump=float(rng.normal(4.4, 0.2)),
            P_elec=float(rng.normal(2.6, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.18, 0.03)),
            xi_leak=0.0,
        )
        policy = StepInterventionController(
            t_star=t_star,
            pre_valve=starved_v, pre_throttle=100.0, pre_pump=pump_stg,
            post_valve=post_v, post_throttle=100.0, post_pump=pump_stg,
        )

    elif regime == ExcitationRegime.THROTTLE_BOTTLENECK_STEP:
        # High compute load initial state -> throttle down at t*
        t_star = int(rng.choice([25, 30, 35]))
        load_val = float(rng.uniform(85.0, 95.0))
        post_th = float(rng.choice([20.0, 30.0, 50.0, 80.0, 100.0]))
        valve_v = float(rng.choice([60.0, 75.0, 85.0]))
        pump_stg = int(rng.choice([2, 3]))

        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention(target="L_cpu", value=load_val, start_step=0, duration=t_star))

        init_state = StateVector(
            T_core=float(rng.normal(91.0, 2.0)),
            T_cool=float(rng.normal(43.0, 1.5)),
            P_sys=float(rng.normal(3.4, 0.1)),
            F_cool=float(rng.normal(38.0, 2.0)),
            L_cpu=load_val,
            V_pos=valve_v,
            Vib_pump=float(rng.normal(4.5, 0.2)),
            P_elec=float(rng.normal(3.5, 0.1)),
            T_amb=float(rng.normal(26.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.20, 0.03)),
            xi_leak=0.0,
        )
        policy = StepInterventionController(
            t_star=t_star,
            pre_valve=valve_v, pre_throttle=100.0, pre_pump=pump_stg,
            post_valve=valve_v, post_throttle=post_th, post_pump=pump_stg,
        )

    elif regime == ExcitationRegime.PUMP_BOTTLENECK_STEP:
        # Open valve + low pump (stage 1) -> step pump to 1, 2, 3, or 4 at t*
        t_star = int(rng.choice([25, 30, 35]))
        valve_v = float(rng.choice([80.0, 90.0, 95.0]))
        post_pump = int(rng.choice([1, 2, 3, 4]))

        init_state = StateVector(
            T_core=float(rng.normal(87.0, 1.5)),
            T_cool=float(rng.normal(39.0, 1.0)),
            P_sys=float(rng.normal(2.5, 0.1)),
            F_cool=float(rng.normal(37.0, 1.5)),
            L_cpu=float(rng.normal(68.0, 3.0)),
            V_pos=valve_v,
            Vib_pump=float(rng.normal(5.8, 0.3)),
            P_elec=float(rng.normal(2.5, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.18, 0.03)),
            xi_leak=0.0,
        )
        policy = StepInterventionController(
            t_star=t_star,
            pre_valve=valve_v, pre_throttle=100.0, pre_pump=1,
            post_valve=valve_v, post_throttle=100.0, post_pump=post_pump,
        )

    elif regime == ExcitationRegime.ORTHOGONAL_EXCITATION:
        init_state = StateVector(
            T_core=float(rng.normal(72.0, 2.0)),
            T_cool=float(rng.normal(35.0, 1.5)),
            P_sys=float(rng.normal(3.2, 0.1)),
            F_cool=float(rng.normal(28.0, 2.0)),
            L_cpu=float(rng.normal(60.0, 5.0)),
            V_pos=float(rng.normal(50.0, 5.0)),
            Vib_pump=float(rng.normal(4.3, 0.2)),
            P_elec=float(rng.normal(2.2, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.16, 0.03)),
            xi_leak=0.0,
        )
        policy = OrthogonalExcitationController(seed=seed)

    elif regime == ExcitationRegime.STRUCTURED_PUMP_SWEEP:
        fixed_v = float(rng.choice([40.0, 60.0, 85.0, 95.0]))
        fixed_th = float(rng.choice([70.0, 100.0]))
        hold = int(rng.choice([20, 25, 30]))
        init_state = StateVector(
            T_core=float(rng.normal(75.0, 2.0)),
            T_cool=float(rng.normal(36.0, 1.5)),
            P_sys=float(rng.normal(3.2, 0.1)),
            F_cool=float(rng.normal(35.0, 2.0)),
            L_cpu=float(rng.normal(65.0, 5.0)),
            V_pos=fixed_v,
            Vib_pump=float(rng.normal(6.0, 0.5)),
            P_elec=float(rng.normal(2.4, 0.1)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.18)),
            Q_internal=float(rng.normal(0.16, 0.03)),
            xi_leak=0.0,
        )
        policy = StructuredPumpSweepController(fixed_valve=fixed_v, fixed_throttle=fixed_th, hold_steps=hold)

    elif regime == ExcitationRegime.THERMAL_RUNAWAY:
        load_val = float(rng.uniform(80.0, 95.0))
        starved_valve = float(rng.uniform(6.0, 18.0))
        pump_stg = int(rng.choice([1, 2, 3, 4]))
        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention(target="L_cpu", value=load_val, start_step=0))
        init_state = StateVector(
            T_core=float(rng.normal(82.0, 2.5)),
            T_cool=float(rng.normal(42.0, 2.0)),
            P_sys=float(rng.normal(2.95, 0.10)),
            F_cool=float(rng.normal(10.0, 1.5)),
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

    else:
        raise ValueError(f"Unknown regime: {regime}")

    return init_state, policy, inv_reg


def generate_excitation_episode(
    split: SplitType,
    index: int,
    regime: ExcitationRegime,
    length: int = 120,
    seed_offset: int = 800000,
) -> OracleEpisode:
    """Generate a single action-excitation episode with isolated seed space."""
    seed = seed_offset + index * 17 + hash(regime.value) % 10000
    ep_id = f"ep_{split.value}_exc_{index:05d}_{regime.value}"

    init_rng = np.random.default_rng(seed)
    init_state, policy, inv_reg = sample_excitation_initial_state(regime, init_rng, seed)

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


def generate_excitation_dataset(
    output_dir: str | Path = "data/excitation_dataset",
    train_count: int = 120,
    val_count: int = 25,
    length: int = 120,
) -> Dict[str, Any]:
    """Generate complete action-excitation Train and Validation splits."""
    base_path = Path(output_dir)
    oracle_train_dir = base_path / "oracle" / "train"
    learner_train_dir = base_path / "learner" / "train"
    oracle_val_dir = base_path / "oracle" / "validation"
    learner_val_dir = base_path / "learner" / "validation"

    for d in [oracle_train_dir, learner_train_dir, oracle_val_dir, learner_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Perfectly balanced regime mix covering all 4 decision scenarios and orthogonal dynamics
    regimes_mix = [
        (ExcitationRegime.NOMINAL_PID, 0.15),
        (ExcitationRegime.VALVE_BOTTLENECK_STEP, 0.15),
        (ExcitationRegime.THROTTLE_BOTTLENECK_STEP, 0.15),
        (ExcitationRegime.PUMP_BOTTLENECK_STEP, 0.15),
        (ExcitationRegime.ORTHOGONAL_EXCITATION, 0.25),
        (ExcitationRegime.STRUCTURED_PUMP_SWEEP, 0.10),
        (ExcitationRegime.THERMAL_RUNAWAY, 0.05),
    ]

    # Generate Train Split
    train_oracle_eps: List[OracleEpisode] = []
    train_learner_eps: List[LearnerEpisode] = []
    idx = 0
    for regime, prop in regimes_mix:
        n_regime = int(round(train_count * prop))
        for _ in range(n_regime):
            orc_ep = generate_excitation_episode(SplitType.TRAIN, idx, regime, length=length, seed_offset=800000)
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
            orc_ep = generate_excitation_episode(SplitType.VAL, idx, regime, length=length, seed_offset=900000)
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
