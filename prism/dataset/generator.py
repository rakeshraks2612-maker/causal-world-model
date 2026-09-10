"""Training and Evaluation Dataset Generation Pipeline.

Implements the Task 2.4 training distribution generator:
- 70% Nominal PI feedback control
- 15% Moderate load variation (L_cpu 30-75%)
- 10% Moderate ambient variation (T_amb 22-30°C)
- 5% Moderate wear variation (W_wear 0.05-0.25)

Maintains strict two-tier export:
- Learner format: data/learner/{split}/ep_XXXXX.npz (Zero ground truth leakage)
- Oracle format: data/oracle/{split}/ep_XXXXX.npz (Complete ground truth for evaluation)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.splits import get_partition_seed, generate_episode_id
from prism.dataset.validators import validate_oracle_episode_integrity, validate_learner_isolation
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.policies import NominalController, HighLoadController, BasePolicy


def sample_training_initial_state(
    regime: str,
    rng: np.random.Generator,
) -> Tuple[StateVector, BasePolicy]:
    """Sample an initial physical state and policy corresponding to the training regime."""
    if regime == "nominal":
        # Standard nominal baseline with small natural variation
        init_state = StateVector(
            T_core=float(rng.normal(68.5, 1.5)),
            T_cool=float(rng.normal(34.0, 1.0)),
            P_sys=float(rng.normal(3.15, 0.08)),
            F_cool=float(rng.normal(32.0, 1.2)),
            L_cpu=float(rng.normal(50.0, 3.0)),
            V_pos=float(rng.normal(55.0, 2.0)),
            Vib_pump=float(rng.normal(4.2, 0.2)),
            P_elec=float(rng.normal(1.85, 0.05)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=float(np.clip(rng.normal(0.05, 0.01), 0.0, 0.15)),
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=float(rng.normal(68.0, 1.0)))

    elif regime == "moderate_load":
        # Moderate workload modulation
        target_valve = float(rng.uniform(40.0, 65.0))
        target_load = float(rng.uniform(35.0, 75.0))
        init_state = StateVector(
            T_core=float(rng.normal(70.0, 2.0)),
            T_cool=float(rng.normal(36.0, 1.5)),
            P_sys=float(rng.normal(3.20, 0.10)),
            F_cool=float(rng.normal(30.0, 2.0)),
            L_cpu=target_load,
            V_pos=target_valve,
            Vib_pump=float(rng.normal(4.4, 0.2)),
            P_elec=float(rng.normal(2.1, 0.1)),
            T_amb=float(rng.normal(26.0, 1.5)),
            W_wear=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.20)),
            Q_internal=float(rng.normal(0.18, 0.04)),
            xi_leak=0.0,
        )
        policy = HighLoadController(fixed_valve=target_valve)

    elif regime == "moderate_ambient":
        # Moderate ambient temperature variation (22 - 30°C)
        t_amb_val = float(rng.uniform(22.0, 30.0))
        init_state = StateVector(
            T_core=float(rng.normal(69.0 + (t_amb_val - 25.0) * 0.4, 1.5)),
            T_cool=float(rng.normal(35.0 + (t_amb_val - 25.0) * 0.5, 1.2)),
            P_sys=float(rng.normal(3.15, 0.08)),
            F_cool=float(rng.normal(32.0, 1.5)),
            L_cpu=float(rng.normal(50.0 + (t_amb_val - 25.0) * 0.8, 2.5)),
            V_pos=float(rng.normal(55.0, 2.0)),
            Vib_pump=float(rng.normal(4.3, 0.2)),
            P_elec=float(rng.normal(1.9, 0.08)),
            T_amb=t_amb_val,
            W_wear=float(np.clip(rng.normal(0.06, 0.02), 0.0, 0.20)),
            Q_internal=float(rng.normal(0.15, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=68.0)

    elif regime == "moderate_wear":
        # Mild wear variation (W_wear 0.05 - 0.25)
        wear_val = float(rng.uniform(0.08, 0.25))
        init_state = StateVector(
            T_core=float(rng.normal(71.0, 1.8)),
            T_cool=float(rng.normal(35.5, 1.2)),
            P_sys=float(rng.normal(3.10, 0.10)),
            F_cool=float(rng.normal(28.0, 1.8)),
            L_cpu=float(rng.normal(50.0, 2.5)),
            V_pos=float(rng.normal(52.0, 2.0)),
            Vib_pump=float(rng.normal(4.8, 0.3)),
            P_elec=float(rng.normal(1.85, 0.05)),
            T_amb=float(rng.normal(25.0, 1.0)),
            W_wear=wear_val,
            Q_internal=float(rng.normal(0.16, 0.03)),
            xi_leak=0.0,
        )
        policy = NominalController(target_temp=68.0)

    else:
        raise ValueError(f"Unknown training regime: {regime}")

    return init_state, policy


def generate_single_episode(
    split: SplitType,
    index: int,
    regime: str,
    length: int = 120,
) -> OracleEpisode:
    """Generate a single OracleEpisode using deterministic seed allocation."""
    seed = get_partition_seed(split, index)
    ep_id = generate_episode_id(split, index)

    # Isolated generator for sampling initial state parameters
    init_rng = np.random.default_rng(seed)
    init_state, policy = sample_training_initial_state(regime, init_rng)

    sim = THCSimulator(seed=seed)
    raw_ep = sim.run_episode(
        policy=policy,
        length=length,
        initial_state=init_state,
        episode_id=ep_id,
    )

    # Construct observation mask (all True for standard training split)
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
            "regime": regime,
            "policy": policy.__class__.__name__,
            "seed": seed,
            "length": length,
            "initial_t_amb": float(init_state.T_amb),
            "initial_w_wear": float(init_state.W_wear),
            "initial_q_internal": float(init_state.Q_internal),
        },
    )

    validate_oracle_episode_integrity(oracle_ep)
    return oracle_ep


def generate_split_dataset(
    split: SplitType,
    count: int,
    regime_distribution: Dict[str, float],
    output_dir: str | Path,
    length: int = 120,
) -> Tuple[List[OracleEpisode], List[LearnerEpisode]]:
    """Generate a complete partition split (e.g. Train, Val, Test) and export both Oracle and Learner archives."""
    base_path = Path(output_dir)
    oracle_dir = base_path / "oracle" / split.value
    learner_dir = base_path / "learner" / split.value
    oracle_dir.mkdir(parents=True, exist_ok=True)
    learner_dir.mkdir(parents=True, exist_ok=True)

    regimes = list(regime_distribution.keys())
    probs = np.array([regime_distribution[r] for r in regimes], dtype=np.float64)
    probs /= np.sum(probs)

    # Deterministic assignment of regimes across indices
    rng_partition = np.random.default_rng(get_partition_seed(split, 0))
    regime_choices = rng_partition.choice(regimes, size=count, p=probs)

    oracle_episodes: List[OracleEpisode] = []
    learner_episodes: List[LearnerEpisode] = []

    for i in range(count):
        regime = str(regime_choices[i])
        oracle_ep = generate_single_episode(split=split, index=i, regime=regime, length=length)
        learner_ep = oracle_ep.to_learner_episode()

        validate_learner_isolation(learner_ep)

        # Save to disk
        oracle_ep.save_npz(oracle_dir / f"{oracle_ep.episode_id}.npz")
        learner_ep.save_npz(learner_dir / f"{learner_ep.episode_id}.npz")

        oracle_episodes.append(oracle_ep)
        learner_episodes.append(learner_ep)

    return oracle_episodes, learner_episodes
