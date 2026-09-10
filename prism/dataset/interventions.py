"""PRISM Causal Intervention Dataset and Evaluation Subsystem.

Implements Task 2.7:
- Class A (Atomic State Interventions): do(V_pos), do(L_cpu), do(Vib_pump)
- Class B (Action-Mediated Interventions): A_valve, A_throttle, A_pump
- Multi-horizon causal effect targets: h in {1, 5, 10, 20, 40}
- Strict Learner vs Oracle record separation with twin-world baseline pairing
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode, FORBIDDEN_LEARNER_KEYS
from prism.dataset.validators import DatasetValidationError
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector, ALL_STATE_VARIABLES, OBSERVABLE_VARIABLES
from prism.simulator.actions import ActionVector
from prism.simulator.interventions import InterventionRegistry, Intervention


class InterventionClass(str, Enum):
    """Classification of interventions."""
    CLASS_A_ATOMIC_STATE = "class_a_atomic_state"
    CLASS_B_ACTION_CONTROL = "class_b_action_control"
    CLASS_C_MULTIVARIABLE = "class_c_multivariable"


@dataclass
class InterventionSpec:
    """Specification of an intervention."""

    target: str
    value: float
    intervention_time: int
    duration: Optional[int] = None  # None = permanent until end of episode
    category: InterventionClass = InterventionClass.CLASS_A_ATOMIC_STATE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate target domain and intervention timing."""
        if self.target not in ALL_STATE_VARIABLES and self.target not in ["A_valve", "A_throttle", "A_pump", "A_flush"]:
            raise ValueError(f"Unknown intervention target: {self.target}")
        if self.intervention_time < 0 or self.intervention_time >= 120:
            raise ValueError(f"Intervention time {self.intervention_time} out of bounds [0, 119]")

        # Physical domain checks
        if self.target == "V_pos" and not (0.0 <= self.value <= 100.0):
            raise ValueError(f"Intervention value for V_pos must be in [0, 100]%, got {self.value}")
        if self.target == "L_cpu" and not (0.0 <= self.value <= 100.0):
            raise ValueError(f"Intervention value for L_cpu must be in [0, 100]%, got {self.value}")
        if self.target == "Vib_pump" and not (0.0 <= self.value <= 30.0):
            raise ValueError(f"Intervention value for Vib_pump must be in [0, 30] mm/s, got {self.value}")


@dataclass
class FailureMetrics:
    """Rigorous failure outcome evaluation contract.
    
    Distinguishes:
    - Failure probability P(Fail | do) vs P(Fail | base)
    - Absolute failure probability delta: P(Fail | do) - P(Fail | base)
    - Relative failure risk: P(Fail | do) / max(1e-4, P(Fail | base))
    - Failure mode transition: (e.g. 'none -> thermal_runaway')
    - Time-to-failure delta: t_fail_int - t_fail_base
    """

    baseline_failed: bool
    intervened_failed: bool
    failure_probability_baseline: float
    failure_probability_intervention: float
    absolute_failure_probability_delta: float  # P(Fail|do) - P(Fail|base) in [-1.0, 1.0]
    relative_failure_risk: float               # P(Fail|do) / max(1e-4, P(Fail|base))
    failure_mode_baseline: Optional[str] = None
    failure_mode_intervention: Optional[str] = None
    failure_mode_transition: str = "none -> none"
    baseline_failure_time: Optional[int] = None
    intervention_failure_time: Optional[int] = None
    time_to_failure_delta: Optional[int] = None  # t_fail_int - t_fail_base

    # Backwards compatibility alias
    @property
    def absolute_failure_delta(self) -> float:
        return self.absolute_failure_probability_delta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonEffect:
    """True causal effect delta at a specific forecast horizon h: Delta Y(t* + h) = Y_int - Y_base."""

    horizon: int
    delta_t_core: float
    delta_t_cool: float
    delta_p_sys: float
    delta_f_cool: float
    delta_l_cpu: float
    delta_v_pos: float
    delta_vib_pump: float
    delta_p_elec: float
    intervened_failed: bool
    baseline_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearnerInterventionRecord:
    """Learner-facing intervention record.
    
    Contains strictly observable pre-intervention history, requested intervention specification,
    and planned future actions. Contains ZERO ground truth oracle trajectories or latent state!
    """

    intervention_id: str
    parent_episode_id: str
    intervention_time: int
    target: str
    value: float
    category: InterventionClass
    pre_intervention_observations: np.ndarray  # Shape: [t* + 1, 8]
    pre_intervention_actions: np.ndarray       # Shape: [t* + 1, 4]
    future_actions: np.ndarray                 # Shape: [T - (t* + 1), 4]
    timestamps: np.ndarray                     # Shape: [T]
    support_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate isolation."""
        t_pre = self.intervention_time + 1
        if self.pre_intervention_observations.shape != (t_pre, 8):
            raise ValueError(f"Pre-intervention observations shape must be ({t_pre}, 8), got {self.pre_intervention_observations.shape}")
        if self.pre_intervention_actions.shape != (t_pre, 4):
            raise ValueError(f"Pre-intervention actions shape must be ({t_pre}, 4), got {self.pre_intervention_actions.shape}")

        for k in self.support_metadata:
            if k in FORBIDDEN_LEARNER_KEYS:
                raise DatasetValidationError(f"Forbidden latent key '{k}' in LearnerInterventionRecord")

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            intervention_id=np.array(self.intervention_id),
            parent_episode_id=np.array(self.parent_episode_id),
            intervention_time=np.array(self.intervention_time),
            target=np.array(self.target),
            value=np.array(self.value),
            category=np.array(self.category.value if isinstance(self.category, InterventionClass) else str(self.category)),
            pre_intervention_observations=self.pre_intervention_observations,
            pre_intervention_actions=self.pre_intervention_actions,
            future_actions=self.future_actions,
            timestamps=self.timestamps,
            support_metadata=np.array(self.support_metadata, dtype=object),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> LearnerInterventionRecord:
        data = np.load(file_path, allow_pickle=True)
        meta_arr = data["support_metadata"]
        meta = meta_arr.item() if meta_arr.ndim == 0 else dict(meta_arr)
        return cls(
            intervention_id=str(data["intervention_id"]),
            parent_episode_id=str(data["parent_episode_id"]),
            intervention_time=int(data["intervention_time"]),
            target=str(data["target"]),
            value=float(data["value"]),
            category=InterventionClass(str(data["category"])),
            pre_intervention_observations=data["pre_intervention_observations"],
            pre_intervention_actions=data["pre_intervention_actions"],
            future_actions=data["future_actions"],
            timestamps=data["timestamps"],
            support_metadata=meta,
        )


@dataclass
class OracleInterventionRecord:
    """Complete evaluation record with matched baseline, full oracle trajectory, and causal deltas."""

    intervention_id: str
    parent_episode_id: str
    intervention_time: int
    target: str
    value: float
    category: InterventionClass
    baseline_oracle_ep: OracleEpisode
    intervened_oracle_ep: OracleEpisode
    horizon_effects: Dict[int, HorizonEffect]
    aggregate_deltas: Dict[str, float]
    failure_metrics: FailureMetrics

    def to_learner_record(self) -> LearnerInterventionRecord:
        """Strip oracle data to produce LearnerInterventionRecord."""
        t_star = self.intervention_time
        obs_pre = self.baseline_oracle_ep.observations[:t_star + 1]
        act_pre = self.baseline_oracle_ep.actions[:t_star + 1]
        act_future = self.intervened_oracle_ep.actions[t_star + 1:]

        # Historical observed support for this variable in the baseline episode
        obs_idx = OBSERVABLE_VARIABLES.index(self.target) if self.target in OBSERVABLE_VARIABLES else -1
        if obs_idx >= 0:
            obs_vals = self.baseline_oracle_ep.observations[:t_star + 1, obs_idx]
            support_min = float(np.nanmin(obs_vals))
            support_max = float(np.nanmax(obs_vals))
            support_dist = float(max(0.0, support_min - self.value, self.value - support_max))
        else:
            support_min = 0.0
            support_max = 100.0
            support_dist = 0.0

        support_meta = {
            "historical_support_min": support_min,
            "historical_support_max": support_max,
            "distance_from_observed_support": support_dist,
            "is_supported": support_dist <= 5.0,
        }

        return LearnerInterventionRecord(
            intervention_id=self.intervention_id,
            parent_episode_id=self.parent_episode_id,
            intervention_time=self.intervention_time,
            target=self.target,
            value=self.value,
            category=self.category,
            pre_intervention_observations=np.copy(obs_pre),
            pre_intervention_actions=np.copy(act_pre),
            future_actions=np.copy(act_future),
            timestamps=np.copy(self.baseline_oracle_ep.timestamps),
            support_metadata=support_meta,
        )

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        horizons_dict = {str(h): eff.to_dict() for h, eff in self.horizon_effects.items()}
        np.savez_compressed(
            path,
            intervention_id=np.array(self.intervention_id),
            parent_episode_id=np.array(self.parent_episode_id),
            intervention_time=np.array(self.intervention_time),
            target=np.array(self.target),
            value=np.array(self.value),
            category=np.array(self.category.value),
            baseline_observations=self.baseline_oracle_ep.observations,
            baseline_ground_truth_states=self.baseline_oracle_ep.ground_truth_states,
            intervened_observations=self.intervened_oracle_ep.observations,
            intervened_ground_truth_states=self.intervened_oracle_ep.ground_truth_states,
            horizon_effects=np.array(horizons_dict, dtype=object),
            aggregate_deltas=np.array(self.aggregate_deltas, dtype=object),
            failure_metrics=np.array(self.failure_metrics.to_dict(), dtype=object),
        )


def execute_intervention_experiment(
    parent_oracle_ep: OracleEpisode,
    spec: InterventionSpec,
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> OracleInterventionRecord:
    """Execute a matched intervention on a historical baseline episode using twin-simulator replay.
    
    Supports:
    - Class A: Atomic State Interventions do(X=x) via graph surgery
    - Class B: Action-Mediated Interventions A_target = x from t* onward through actuator dynamics
    """
    spec.validate()
    t_star = spec.intervention_time
    init_state = StateVector.from_array(parent_oracle_ep.ground_truth_states[0])

    if spec.category == InterventionClass.CLASS_A_ATOMIC_STATE:
        # Atomic graph surgery on state variable
        inv_reg = InterventionRegistry()
        inv_reg.add(Intervention(
            target=spec.target,
            value=spec.value,
            start_step=t_star,
            duration=spec.duration,
        ))
        actions_seq = parent_oracle_ep.actions[:-1]
    elif spec.category == InterventionClass.CLASS_B_ACTION_CONTROL:
        # Action-mediated intervention: modify action sequence from t* onward, letting actuator dynamics run
        inv_reg = None
        actions_seq = np.copy(parent_oracle_ep.actions[:-1])
        act_idx = {"A_valve": 0, "A_throttle": 1, "A_pump": 2, "A_flush": 3}[spec.target]
        dur = spec.duration if spec.duration is not None else (len(actions_seq) - t_star)
        end_step = min(len(actions_seq), t_star + dur)
        actions_seq[t_star:end_step, act_idx] = spec.value
    else:
        raise NotImplementedError(f"Intervention category {spec.category} not implemented")

    sim = THCSimulator(seed=parent_oracle_ep.seed)
    intervened_raw = sim.replay_episode(
        recorded_noise=parent_oracle_ep.exogenous_noise,
        action_sequence=actions_seq,
        initial_state=init_state,
        interventions=inv_reg,
        episode_id=f"{parent_oracle_ep.episode_id}_{spec.target}_{f'{spec.value:g}'.replace('.', '_')}",
    )

    intervened_oracle_ep = OracleEpisode(
        episode_id=intervened_raw.episode_id,
        seed=parent_oracle_ep.seed,
        split=SplitType.INTERVENTION,
        timestamps=intervened_raw.timestamps,
        observations=intervened_raw.observations,
        observation_mask=~np.isnan(intervened_raw.observations),
        actions=intervened_raw.actions,
        ground_truth_states=intervened_raw.ground_truth_states,
        exogenous_noise=intervened_raw.exogenous_noise,
        failure_latched=intervened_raw.failure_latched,
        failure_mode=intervened_raw.failure_mode,
        failure_timestamp=intervened_raw.failure_timestamp,
        oracle_metadata={
            "parent_episode_id": parent_oracle_ep.episode_id,
            "intervention_target": spec.target,
            "intervention_value": spec.value,
            "intervention_time": t_star,
            "category": spec.category.value,
        },
    )

    # Compute multi-horizon causal target deltas
    horizon_effects: Dict[int, HorizonEffect] = {}
    base_states = parent_oracle_ep.ground_truth_states
    int_states = intervened_oracle_ep.ground_truth_states

    for h in horizons:
        step_idx = t_star + h
        if step_idx < len(base_states):
            delta = int_states[step_idx] - base_states[step_idx]
            horizon_effects[h] = HorizonEffect(
                horizon=h,
                delta_t_core=float(delta[0]),
                delta_t_cool=float(delta[1]),
                delta_p_sys=float(delta[2]),
                delta_f_cool=float(delta[3]),
                delta_l_cpu=float(delta[4]),
                delta_v_pos=float(delta[5]),
                delta_vib_pump=float(delta[6]),
                delta_p_elec=float(delta[7]),
                intervened_failed=intervened_oracle_ep.failure_latched,
                baseline_failed=parent_oracle_ep.failure_latched,
            )

    # Compute failure metrics contract
    base_failed = parent_oracle_ep.failure_latched
    int_failed = intervened_oracle_ep.failure_latched
    base_prob = 1.0 if base_failed else 0.0
    int_prob = 1.0 if int_failed else 0.0
    abs_delta = int_prob - base_prob
    rel_risk = int_prob / max(1e-4, base_prob) if base_failed else (float("inf") if int_failed else 1.0)
    time_delta = None
    if base_failed and int_failed and parent_oracle_ep.failure_timestamp and intervened_oracle_ep.failure_timestamp:
        time_delta = intervened_oracle_ep.failure_timestamp - parent_oracle_ep.failure_timestamp

    mode_base = parent_oracle_ep.failure_mode
    mode_int = intervened_oracle_ep.failure_mode
    mode_transition = f"{mode_base or 'none'} -> {mode_int or 'none'}"

    fail_metrics = FailureMetrics(
        baseline_failed=base_failed,
        intervened_failed=int_failed,
        failure_probability_baseline=base_prob,
        failure_probability_intervention=int_prob,
        absolute_failure_probability_delta=abs_delta,
        relative_failure_risk=rel_risk,
        failure_mode_baseline=mode_base,
        failure_mode_intervention=mode_int,
        failure_mode_transition=mode_transition,
        baseline_failure_time=parent_oracle_ep.failure_timestamp,
        intervention_failure_time=intervened_oracle_ep.failure_timestamp,
        time_to_failure_delta=time_delta,
    )

    # Compute aggregate trajectory deltas over post-intervention window
    post_base = base_states[t_star:]
    post_int = int_states[t_star:]
    agg_deltas = {
        "mean_delta_t_core": float(np.mean(post_int[:, 0] - post_base[:, 0])),
        "max_delta_t_core": float(np.max(post_int[:, 0] - post_base[:, 0])),
        "min_delta_t_core": float(np.min(post_int[:, 0] - post_base[:, 0])),
        "mean_delta_f_cool": float(np.mean(post_int[:, 3] - post_base[:, 3])),
        "mean_delta_p_sys": float(np.mean(post_int[:, 2] - post_base[:, 2])),
        "mean_delta_vib_pump": float(np.mean(post_int[:, 6] - post_base[:, 6])),
    }

    val_str = f"{spec.value:g}".replace(".", "_")
    inv_id = f"inv_{parent_oracle_ep.episode_id}_t{t_star}_{spec.target}_{val_str}"

    return OracleInterventionRecord(
        intervention_id=inv_id,
        parent_episode_id=parent_oracle_ep.episode_id,
        intervention_time=t_star,
        target=spec.target,
        value=spec.value,
        category=spec.category,
        baseline_oracle_ep=parent_oracle_ep,
        intervened_oracle_ep=intervened_oracle_ep,
        horizon_effects=horizon_effects,
        aggregate_deltas=agg_deltas,
        failure_metrics=fail_metrics,
    )
