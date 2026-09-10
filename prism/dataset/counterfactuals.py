"""PRISM Counterfactual Dataset and Latent Abduction Subsystem.

Implements Task 2.8 (Pearl Level 3 Counterfactual Reasoning):
- Evaluates: P(Y_{a'} | O_{0:t*}, A_{0:t*})
- Twin-World Frozen Replay: identical X0, identical U_{0:T}, identical pre-intervention history,
  single-action substitution at t* (A^{CF}_{t*} != A^{fact}_{t*}), identical future actions for t > t*.
- Three Layers of Truth:
  1. Learner Layer: Strictly historical observations O_{0:t*}, actions A_{0:t*}, counterfactual action A'_{t*}, future actions A_{t*+1:T}.
  2. Oracle Abduction Layer: True latent state Z_{t*} = [T_amb, W_wear, Q_internal, xi_leak] and noise history U_{0:t*}.
  3. Oracle Counterfactual Layer: Ground-truth counterfactual trajectories, multi-horizon effect deltas, failure transitions.
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
from prism.simulator.state import StateVector, ALL_STATE_VARIABLES, OBSERVABLE_VARIABLES, LATENT_VARIABLES
from prism.simulator.actions import ActionVector, ACTION_VARIABLES


@dataclass
class CounterfactualSpec:
    """Specification for a single-action counterfactual substitution at timestep t*."""

    target_action: str
    counterfactual_value: float
    counterfactual_time: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate target action channel, domain bounds, and timing."""
        if self.target_action not in ACTION_VARIABLES:
            raise ValueError(f"Unknown counterfactual action target: '{self.target_action}'. Must be one of {ACTION_VARIABLES}")
        if self.counterfactual_time < 0 or self.counterfactual_time >= 120:
            raise ValueError(f"Counterfactual time {self.counterfactual_time} out of bounds [0, 119]")

        # Physical domain checks for control actions
        if self.target_action == "A_valve" and not (0.0 <= self.counterfactual_value <= 100.0):
            raise ValueError(f"Counterfactual value for A_valve must be in [0, 100]%, got {self.counterfactual_value}")
        if self.target_action == "A_throttle" and not (0.0 <= self.counterfactual_value <= 100.0):
            raise ValueError(f"Counterfactual value for A_throttle must be in [0, 100]%, got {self.counterfactual_value}")
        if self.target_action == "A_pump" and not (0.0 <= self.counterfactual_value <= 4.0):
            raise ValueError(f"Counterfactual value for A_pump must be in stage [0, 4], got {self.counterfactual_value}")
        if self.target_action == "A_flush" and self.counterfactual_value not in [0.0, 1.0]:
            raise ValueError(f"Counterfactual value for A_flush must be 0 or 1, got {self.counterfactual_value}")


@dataclass
class CounterfactualFailureMetrics:
    """Rigorous failure outcome evaluation contract for counterfactuals."""

    factual_failed: bool
    counterfactual_failed: bool
    failure_probability_factual: float
    failure_probability_counterfactual: float
    absolute_failure_probability_delta: float  # P(Fail|CF) - P(Fail|fact) in [-1.0, 1.0]
    relative_failure_risk: float               # P(Fail|CF) / max(1e-4, P(Fail|fact))
    failure_mode_factual: Optional[str] = None
    failure_mode_counterfactual: Optional[str] = None
    failure_mode_transition: str = "none -> none"
    factual_failure_time: Optional[int] = None
    counterfactual_failure_time: Optional[int] = None
    time_to_failure_delta: Optional[int] = None  # t_fail_cf - t_fail_fact

    @property
    def absolute_failure_delta(self) -> float:
        return self.absolute_failure_probability_delta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualHorizonEffect:
    """True counterfactual causal effect delta at a specific forecast horizon h: Delta Y(t* + h) = Y_cf - Y_fact."""

    horizon: int
    delta_t_core: float
    delta_t_cool: float
    delta_p_sys: float
    delta_f_cool: float
    delta_l_cpu: float
    delta_v_pos: float
    delta_vib_pump: float
    delta_p_elec: float
    counterfactual_failed: bool
    factual_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical substitution matrix for Task 2.8 Counterfactual Dataset
COUNTERFACTUAL_ACTION_MATRIX: Dict[str, List[float]] = {
    "A_valve": [50.0, 70.0, 85.0, 100.0],
    "A_throttle": [20.0, 50.0, 80.0, 100.0],
    "A_pump": [1.0, 2.0, 3.0, 4.0],
    "A_flush": [0.0, 1.0],
}


@dataclass
class OracleLatentTruth:
    """Ground-truth latent and exogenous information retained strictly for benchmark evaluation.
    
    NOTE: This is NOT an inferred posterior distribution! It represents the simulator's true
    unobserved state Z_{t*} = [T_amb, W_wear, Q_internal, xi_leak] and noise innovations U_{0:t*},
    against which a future learned abduction model (P(Z, U | O, A)) will be quantitatively evaluated.
    """

    true_latent_state_t_star: np.ndarray  # Shape: [4] -> [T_amb, W_wear, Q_internal, xi_leak]
    true_noise_history: np.ndarray        # Shape: [t* + 1, 12]
    true_state_t_star: np.ndarray         # Shape: [12] -> full ground-truth state at t*

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_latent_state_t_star": self.true_latent_state_t_star,
            "true_noise_history": self.true_noise_history,
            "true_state_t_star": self.true_state_t_star,
        }


# Backwards compatibility alias
OracleAbductionState = OracleLatentTruth


@dataclass
class LearnerCounterfactualRecord:
    """Learner-facing counterfactual evaluation record.
    
    Contains strictly observable pre-intervention history (O_{0:t*}, A_{0:t*}),
    the substituted counterfactual action (A'_{t*}), planned future actions (A_{t*+1:T}),
    and historical support metadata.
    Contains ZERO ground truth oracle trajectories, latent states, or exogenous noises!
    """

    counterfactual_id: str
    parent_episode_id: str
    counterfactual_time: int
    target_action: str
    counterfactual_value: float
    historical_observations: np.ndarray       # Shape: [t* + 1, 8]
    historical_observation_mask: np.ndarray  # Shape: [t* + 1, 8]
    historical_actions: np.ndarray            # Shape: [t* + 1, 4]
    factual_action: np.ndarray                # Shape: [4]
    counterfactual_action: np.ndarray         # Shape: [4]
    future_actions: np.ndarray                # Shape: [T - (t* + 1), 4]
    timestamps: np.ndarray                    # Shape: [T]
    support_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate isolation and dimensional integrity."""
        t_pre = self.counterfactual_time + 1
        if self.historical_observations.shape != (t_pre, 8):
            raise ValueError(f"Historical observations shape must be ({t_pre}, 8), got {self.historical_observations.shape}")
        if self.historical_observation_mask.shape != (t_pre, 8):
            raise ValueError(f"Historical observation mask shape must be ({t_pre}, 8), got {self.historical_observation_mask.shape}")
        if self.historical_actions.shape != (t_pre, 4):
            raise ValueError(f"Historical actions shape must be ({t_pre}, 4), got {self.historical_actions.shape}")
        if self.factual_action.shape != (4,):
            raise ValueError(f"Factual action shape must be (4,), got {self.factual_action.shape}")
        if self.counterfactual_action.shape != (4,):
            raise ValueError(f"Counterfactual action shape must be (4,), got {self.counterfactual_action.shape}")

        for k in self.support_metadata:
            if k in FORBIDDEN_LEARNER_KEYS:
                raise DatasetValidationError(f"Forbidden latent key '{k}' in LearnerCounterfactualRecord support_metadata")

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            counterfactual_id=np.array(self.counterfactual_id),
            parent_episode_id=np.array(self.parent_episode_id),
            counterfactual_time=np.array(self.counterfactual_time),
            target_action=np.array(self.target_action),
            counterfactual_value=np.array(self.counterfactual_value),
            historical_observations=self.historical_observations,
            historical_observation_mask=self.historical_observation_mask,
            historical_actions=self.historical_actions,
            factual_action=self.factual_action,
            counterfactual_action=self.counterfactual_action,
            future_actions=self.future_actions,
            timestamps=self.timestamps,
            support_metadata=np.array(self.support_metadata, dtype=object),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> LearnerCounterfactualRecord:
        data = np.load(file_path, allow_pickle=True)
        meta_arr = data["support_metadata"]
        meta = meta_arr.item() if meta_arr.ndim == 0 else dict(meta_arr)
        return cls(
            counterfactual_id=str(data["counterfactual_id"]),
            parent_episode_id=str(data["parent_episode_id"]),
            counterfactual_time=int(data["counterfactual_time"]),
            target_action=str(data["target_action"]),
            counterfactual_value=float(data["counterfactual_value"]),
            historical_observations=data["historical_observations"],
            historical_observation_mask=data["historical_observation_mask"],
            historical_actions=data["historical_actions"],
            factual_action=data["factual_action"],
            counterfactual_action=data["counterfactual_action"],
            future_actions=data["future_actions"],
            timestamps=data["timestamps"],
            support_metadata=meta,
        )


@dataclass
class OracleCounterfactualRecord:
    """Complete oracle counterfactual evaluation record retained solely for benchmark scoring."""

    counterfactual_id: str
    parent_episode_id: str
    counterfactual_time: int
    spec: CounterfactualSpec
    factual_oracle_ep: OracleEpisode
    counterfactual_oracle_ep: OracleEpisode
    abduction_ground_truth: OracleLatentTruth
    horizon_effects: Dict[int, CounterfactualHorizonEffect]
    aggregate_effects: Dict[str, float]
    failure_metrics: CounterfactualFailureMetrics

    def to_learner_record(self) -> LearnerCounterfactualRecord:
        """Strip oracle truth to produce isolated LearnerCounterfactualRecord."""
        t_star = self.counterfactual_time
        obs_pre = self.factual_oracle_ep.observations[:t_star + 1]
        mask_pre = self.factual_oracle_ep.observation_mask[:t_star + 1]
        act_pre = self.factual_oracle_ep.actions[:t_star + 1]
        act_future = self.counterfactual_oracle_ep.actions[t_star + 1:]
        act_fact = self.factual_oracle_ep.actions[t_star]
        act_cf = self.counterfactual_oracle_ep.actions[t_star]

        act_idx = ACTION_VARIABLES.index(self.spec.target_action)
        hist_act_vals = self.factual_oracle_ep.actions[:t_star + 1, act_idx]
        support_min = float(np.min(hist_act_vals))
        support_max = float(np.max(hist_act_vals))
        support_dist = float(max(0.0, support_min - self.spec.counterfactual_value, self.spec.counterfactual_value - support_max))

        support_meta = {
            "historical_action_support_min": support_min,
            "historical_action_support_max": support_max,
            "distance_from_observed_support": support_dist,
            "is_supported": support_dist == 0.0,
            "action_channel": self.spec.target_action,
        }

        return LearnerCounterfactualRecord(
            counterfactual_id=self.counterfactual_id,
            parent_episode_id=self.parent_episode_id,
            counterfactual_time=self.counterfactual_time,
            target_action=self.spec.target_action,
            counterfactual_value=self.spec.counterfactual_value,
            historical_observations=np.copy(obs_pre),
            historical_observation_mask=np.copy(mask_pre),
            historical_actions=np.copy(act_pre),
            factual_action=np.copy(act_fact),
            counterfactual_action=np.copy(act_cf),
            future_actions=np.copy(act_future),
            timestamps=np.copy(self.factual_oracle_ep.timestamps),
            support_metadata=support_meta,
        )

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        horizons_dict = {str(h): eff.to_dict() for h, eff in self.horizon_effects.items()}
        np.savez_compressed(
            path,
            counterfactual_id=np.array(self.counterfactual_id),
            parent_episode_id=np.array(self.parent_episode_id),
            counterfactual_time=np.array(self.counterfactual_time),
            target_action=np.array(self.spec.target_action),
            counterfactual_value=np.array(self.spec.counterfactual_value),
            factual_observations=self.factual_oracle_ep.observations,
            factual_ground_truth_states=self.factual_oracle_ep.ground_truth_states,
            factual_exogenous_noise=self.factual_oracle_ep.exogenous_noise,
            counterfactual_observations=self.counterfactual_oracle_ep.observations,
            counterfactual_ground_truth_states=self.counterfactual_oracle_ep.ground_truth_states,
            counterfactual_exogenous_noise=self.counterfactual_oracle_ep.exogenous_noise,
            abduction_true_latent=self.abduction_ground_truth.true_latent_state_t_star,
            abduction_true_state=self.abduction_ground_truth.true_state_t_star,
            horizon_effects=np.array(horizons_dict, dtype=object),
            aggregate_effects=np.array(self.aggregate_effects, dtype=object),
            failure_metrics=np.array(self.failure_metrics.to_dict(), dtype=object),
        )


def execute_counterfactual_experiment(
    factual_oracle_ep: OracleEpisode,
    spec: CounterfactualSpec,
    horizons: Tuple[int, ...] = (1, 5, 10, 20, 40),
) -> OracleCounterfactualRecord:
    """Execute a single-action counterfactual substitution on a historical episode using twin-world replay.
    
    Frozen world guarantees:
    - Same initial state X0
    - Same exogenous noise tensor U_{0:T} (bit-for-bit)
    - Same pre-intervention history A_{0:t*-1}
    - Distinct action at t*: A^{CF}[t*, act_idx] = spec.counterfactual_value != A^{fact}[t*, act_idx]
    - Same future actions A_{t*+1:T}
    """
    spec.validate()
    t_star = spec.counterfactual_time
    act_idx = ACTION_VARIABLES.index(spec.target_action)

    factual_val = float(factual_oracle_ep.actions[t_star, act_idx])
    if np.isclose(factual_val, spec.counterfactual_value, atol=1e-3):
        raise ValueError(
            f"Trivial counterfactual: factual action {spec.target_action}={factual_val:.2f} "
            f"is identical to requested counterfactual value {spec.counterfactual_value:.2f}"
        )

    # Construct counterfactual action sequence: modify strictly step t* on target channel
    cf_actions = np.copy(factual_oracle_ep.actions[:-1])
    cf_actions[t_star, act_idx] = spec.counterfactual_value

    # Replay counterfactual world with identical X0 and identical exogenous noise tensor
    init_state = StateVector.from_array(factual_oracle_ep.ground_truth_states[0])
    sim = THCSimulator(seed=factual_oracle_ep.seed)
    cf_raw = sim.replay_episode(
        recorded_noise=factual_oracle_ep.exogenous_noise,
        action_sequence=cf_actions,
        initial_state=init_state,
        episode_id=f"{factual_oracle_ep.episode_id}_cf_t{t_star}_{spec.target_action}_{int(spec.counterfactual_value)}",
    )

    cf_oracle_ep = OracleEpisode(
        episode_id=cf_raw.episode_id,
        seed=factual_oracle_ep.seed,
        split=SplitType.COUNTERFACTUAL,
        timestamps=cf_raw.timestamps,
        observations=cf_raw.observations,
        observation_mask=~np.isnan(cf_raw.observations),
        actions=cf_raw.actions,
        ground_truth_states=cf_raw.ground_truth_states,
        exogenous_noise=cf_raw.exogenous_noise,
        failure_latched=cf_raw.failure_latched,
        failure_mode=cf_raw.failure_mode,
        failure_timestamp=cf_raw.failure_timestamp,
        oracle_metadata={
            "parent_episode_id": factual_oracle_ep.episode_id,
            "target_action": spec.target_action,
            "factual_action_value": factual_val,
            "counterfactual_action_value": spec.counterfactual_value,
            "counterfactual_time": t_star,
        },
    )

    # Construct Oracle Abduction State at t*
    # Latents: [T_amb, W_wear, Q_internal, xi_leak] at indices 8, 9, 10, 11
    true_latent_t_star = np.copy(factual_oracle_ep.ground_truth_states[t_star, 8:12])
    true_noise_hist = np.copy(factual_oracle_ep.exogenous_noise[:t_star + 1])
    true_state_t_star = np.copy(factual_oracle_ep.ground_truth_states[t_star])

    abduction_gt = OracleLatentTruth(
        true_latent_state_t_star=true_latent_t_star,
        true_noise_history=true_noise_hist,
        true_state_t_star=true_state_t_star,
    )

    # Compute multi-horizon counterfactual effects: Delta Y(t* + h) = Y_cf - Y_fact
    horizon_effects: Dict[int, CounterfactualHorizonEffect] = {}
    fact_states = factual_oracle_ep.ground_truth_states
    cf_states = cf_oracle_ep.ground_truth_states

    for h in horizons:
        step_idx = t_star + h
        if step_idx < len(fact_states):
            delta = cf_states[step_idx] - fact_states[step_idx]
            horizon_effects[h] = CounterfactualHorizonEffect(
                horizon=h,
                delta_t_core=float(delta[0]),
                delta_t_cool=float(delta[1]),
                delta_p_sys=float(delta[2]),
                delta_f_cool=float(delta[3]),
                delta_l_cpu=float(delta[4]),
                delta_v_pos=float(delta[5]),
                delta_vib_pump=float(delta[6]),
                delta_p_elec=float(delta[7]),
                counterfactual_failed=cf_oracle_ep.failure_latched,
                factual_failed=factual_oracle_ep.failure_latched,
            )

    # Compute counterfactual failure metrics
    fact_failed = factual_oracle_ep.failure_latched
    cf_failed = cf_oracle_ep.failure_latched
    fact_prob = 1.0 if fact_failed else 0.0
    cf_prob = 1.0 if cf_failed else 0.0
    abs_delta = cf_prob - fact_prob
    rel_risk = cf_prob / max(1e-4, fact_prob) if fact_failed else (float("inf") if cf_failed else 1.0)
    time_delta = None
    if fact_failed and cf_failed and factual_oracle_ep.failure_timestamp and cf_oracle_ep.failure_timestamp:
        time_delta = cf_oracle_ep.failure_timestamp - factual_oracle_ep.failure_timestamp

    mode_fact = factual_oracle_ep.failure_mode
    mode_cf = cf_oracle_ep.failure_mode
    mode_trans = f"{mode_fact or 'none'} -> {mode_cf or 'none'}"

    fail_metrics = CounterfactualFailureMetrics(
        factual_failed=fact_failed,
        counterfactual_failed=cf_failed,
        failure_probability_factual=fact_prob,
        failure_probability_counterfactual=cf_prob,
        absolute_failure_probability_delta=abs_delta,
        relative_failure_risk=rel_risk,
        failure_mode_factual=mode_fact,
        failure_mode_counterfactual=mode_cf,
        failure_mode_transition=mode_trans,
        factual_failure_time=factual_oracle_ep.failure_timestamp,
        counterfactual_failure_time=cf_oracle_ep.failure_timestamp,
        time_to_failure_delta=time_delta,
    )

    # Compute aggregate trajectory effects over post-counterfactual window [t* : T]
    post_fact = fact_states[t_star:]
    post_cf = cf_states[t_star:]
    agg_effects = {
        "peak_t_core_factual": float(np.max(post_fact[:, 0])),
        "peak_t_core_counterfactual": float(np.max(post_cf[:, 0])),
        "mean_t_core_factual": float(np.mean(post_fact[:, 0])),
        "mean_t_core_counterfactual": float(np.mean(post_cf[:, 0])),
        "mean_delta_t_core": float(np.mean(post_cf[:, 0] - post_fact[:, 0])),
        "mean_delta_t_cool": float(np.mean(post_cf[:, 1] - post_fact[:, 1])),
        "mean_delta_p_sys": float(np.mean(post_cf[:, 2] - post_fact[:, 2])),
        "mean_delta_f_cool": float(np.mean(post_cf[:, 3] - post_fact[:, 3])),
        "max_p_sys_counterfactual": float(np.max(post_cf[:, 2])),
        "min_f_cool_counterfactual": float(np.min(post_cf[:, 3])),
    }

    cf_id = f"cf_{factual_oracle_ep.episode_id}_t{t_star}_{spec.target_action}_{int(spec.counterfactual_value)}"

    return OracleCounterfactualRecord(
        counterfactual_id=cf_id,
        parent_episode_id=factual_oracle_ep.episode_id,
        counterfactual_time=t_star,
        spec=spec,
        factual_oracle_ep=factual_oracle_ep,
        counterfactual_oracle_ep=cf_oracle_ep,
        abduction_ground_truth=abduction_gt,
        horizon_effects=horizon_effects,
        aggregate_effects=agg_effects,
        failure_metrics=fail_metrics,
    )
