"""PRISM Decision Quality Benchmark Schema and Data Structures (Task 5.1).

Defines the isolated 2-layer benchmark contracts:
1. LearnerDecisionScenario:
   - Historical observations O_{0:t*}
   - Historical actions A_{0:t*}
   - Planned future baseline actions
   - Candidate intervention specs
   - Target decision horizon
   - NO ORACLE HIDDEN STATES OR NOISE HISTORIES!
2. OracleDecisionScenario:
   - Ground truth initial state X0 and latent state Z_t*
   - Ground truth physical trajectories for every candidate
   - Ground truth optimal candidate ID
   - Ground truth safety verdicts and utility scores
   - Expected decision class (RECOMMEND, CAUTION, ABSTAIN)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from prism.dataset.schema import FORBIDDEN_LEARNER_KEYS
from prism.dataset.validators import DatasetValidationError
from prism.intervention.spec import InterventionSpec, InterventionType
from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES, ALL_STATE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES


class DecisionClass(str, Enum):
    """Decision classification state."""
    RECOMMEND = "RECOMMEND"  # Safe, high confidence, within support
    CAUTION = "CAUTION"      # Safe, elevated uncertainty or marginal constraints
    ABSTAIN = "ABSTAIN"      # Unsafe, OOD, or no candidate satisfies constraints


@dataclass
class CandidateActionSpec:
    """Action candidate evaluated in the decision scenario."""

    candidate_id: str
    target: str
    value: float
    intervention_type: str = "action_control"
    secondary_target: Optional[str] = None
    secondary_value: Optional[float] = None

    def to_specs(self, t_star: int) -> List[InterventionSpec]:
        """Convert to InterventionSpec instances."""
        itype = InterventionType.ACTION_CONTROL if self.intervention_type == "action_control" else InterventionType.STATE_CLAMP
        specs = [InterventionSpec(target=self.target, value=self.value, intervention_time=t_star, intervention_type=itype)]
        if self.secondary_target is not None and self.secondary_value is not None:
            specs.append(InterventionSpec(target=self.secondary_target, value=self.secondary_value, intervention_time=t_star, intervention_type=itype))
        return specs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleCandidateOutcome:
    """Ground truth simulated outcome for a candidate."""

    candidate_id: str
    spec: CandidateActionSpec
    is_safe: bool
    safety_violations: List[str]
    peak_t_core: float
    max_pressure: float
    min_flow: float
    mean_cpu_load: float
    mean_power: float
    operational_cost: float
    true_utility: float
    ground_truth_states: np.ndarray       # Shape: [H, 12]
    ground_truth_observations: np.ndarray # Shape: [H, 8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "spec": self.spec.to_dict(),
            "is_safe": self.is_safe,
            "safety_violations": self.safety_violations,
            "peak_t_core": float(self.peak_t_core),
            "max_pressure": float(self.max_pressure),
            "min_flow": float(self.min_flow),
            "mean_cpu_load": float(self.mean_cpu_load),
            "mean_power": float(self.mean_power),
            "operational_cost": float(self.operational_cost),
            "true_utility": float(self.true_utility),
        }


@dataclass
class LearnerDecisionScenario:
    """Learner-facing decision scenario record.
    
    Contains strictly historical observations and candidate actions.
    FIREWALL: Contains ZERO ground-truth unobserved latent variables!
    """

    scenario_id: str
    scenario_name: str
    description: str
    intervention_time: int
    historical_observations: np.ndarray       # Shape: [t* + 1, 8]
    historical_observation_mask: np.ndarray  # Shape: [t* + 1, 8]
    historical_actions: np.ndarray            # Shape: [t* + 1, 4]
    future_baseline_actions: np.ndarray       # Shape: [H_fut, 4]
    candidate_actions: List[CandidateActionSpec]
    support_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate isolation and dimensional integrity."""
        t_pre = self.intervention_time + 1
        if self.historical_observations.shape != (t_pre, 8):
            raise ValueError(f"Historical observations shape must be ({t_pre}, 8), got {self.historical_observations.shape}")
        if self.historical_observation_mask.shape != (t_pre, 8):
            raise ValueError(f"Historical observation mask shape must be ({t_pre}, 8), got {self.historical_observation_mask.shape}")
        if self.historical_actions.shape != (t_pre, 4):
            raise ValueError(f"Historical actions shape must be ({t_pre}, 4), got {self.historical_actions.shape}")

        for k in self.support_metadata:
            if k in FORBIDDEN_LEARNER_KEYS:
                raise DatasetValidationError(f"Forbidden latent key '{k}' in LearnerDecisionScenario support_metadata")

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cands_dict = [c.to_dict() for c in self.candidate_actions]
        np.savez_compressed(
            path,
            scenario_id=np.array(self.scenario_id),
            scenario_name=np.array(self.scenario_name),
            description=np.array(self.description),
            intervention_time=np.array(self.intervention_time),
            historical_observations=self.historical_observations,
            historical_observation_mask=self.historical_observation_mask,
            historical_actions=self.historical_actions,
            future_baseline_actions=self.future_baseline_actions,
            candidate_actions=np.array(cands_dict, dtype=object),
            support_metadata=np.array(self.support_metadata, dtype=object),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> LearnerDecisionScenario:
        data = np.load(file_path, allow_pickle=True)
        meta_arr = data["support_metadata"]
        meta = meta_arr.item() if meta_arr.ndim == 0 else dict(meta_arr)
        cands_arr = data["candidate_actions"]
        cands_raw = cands_arr.tolist() if hasattr(cands_arr, "tolist") else list(cands_arr)
        cands = [CandidateActionSpec(**c) for c in cands_raw]
        return cls(
            scenario_id=str(data["scenario_id"]),
            scenario_name=str(data["scenario_name"]),
            description=str(data["description"]),
            intervention_time=int(data["intervention_time"]),
            historical_observations=data["historical_observations"],
            historical_observation_mask=data["historical_observation_mask"],
            historical_actions=data["historical_actions"],
            future_baseline_actions=data["future_baseline_actions"],
            candidate_actions=cands,
            support_metadata=meta,
        )


@dataclass
class OracleDecisionScenario:
    """Oracle ground-truth scenario evaluation record."""

    scenario_id: str
    scenario_name: str
    description: str
    intervention_time: int
    true_latent_state_t_star: np.ndarray       # Shape: [4]
    true_state_history: np.ndarray             # Shape: [t* + 1, 12]
    exogenous_noise_history: np.ndarray        # Shape: [t* + 1, 12]
    candidate_outcomes: Dict[str, OracleCandidateOutcome]
    oracle_optimal_candidate_id: str
    expected_decision_class: DecisionClass
    oracle_rationale: str

    def to_learner_scenario(self, learner_ep: Any) -> LearnerDecisionScenario:
        """Construct learner scenario with strict latent isolation."""
        cands = [out.spec for out in self.candidate_outcomes.values()]
        return LearnerDecisionScenario(
            scenario_id=self.scenario_id,
            scenario_name=self.scenario_name,
            description=self.description,
            intervention_time=self.intervention_time,
            historical_observations=np.copy(learner_ep.observations[: self.intervention_time + 1]),
            historical_observation_mask=np.copy(learner_ep.observation_mask[: self.intervention_time + 1]),
            historical_actions=np.copy(learner_ep.actions[: self.intervention_time + 1]),
            future_baseline_actions=np.copy(learner_ep.actions[self.intervention_time + 1 :]),
            candidate_actions=cands,
            support_metadata={
                "scenario_name": self.scenario_name,
                "candidate_count": len(cands),
            },
        )

    def save_npz(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        outcomes_dict = {cid: out.to_dict() for cid, out in self.candidate_outcomes.items()}
        np.savez_compressed(
            path,
            scenario_id=np.array(self.scenario_id),
            scenario_name=np.array(self.scenario_name),
            description=np.array(self.description),
            intervention_time=np.array(self.intervention_time),
            true_latent_state_t_star=self.true_latent_state_t_star,
            true_state_history=self.true_state_history,
            exogenous_noise_history=self.exogenous_noise_history,
            candidate_outcomes=np.array(outcomes_dict, dtype=object),
            oracle_optimal_candidate_id=np.array(self.oracle_optimal_candidate_id),
            expected_decision_class=np.array(self.expected_decision_class.value),
            oracle_rationale=np.array(self.oracle_rationale),
        )

    @classmethod
    def load_npz(cls, file_path: str | Path) -> OracleDecisionScenario:
        data = np.load(file_path, allow_pickle=True)
        outcomes_raw = data["candidate_outcomes"].item() if data["candidate_outcomes"].ndim == 0 else dict(data["candidate_outcomes"])
        outcomes = {}
        for cid, out_dict in outcomes_raw.items():
            spec_dict = out_dict["spec"]
            spec = CandidateActionSpec(**spec_dict)
            outcomes[cid] = OracleCandidateOutcome(
                candidate_id=out_dict["candidate_id"],
                spec=spec,
                is_safe=bool(out_dict["is_safe"]),
                safety_violations=list(out_dict["safety_violations"]),
                peak_t_core=float(out_dict["peak_t_core"]),
                max_pressure=float(out_dict["max_pressure"]),
                min_flow=float(out_dict["min_flow"]),
                mean_cpu_load=float(out_dict["mean_cpu_load"]),
                mean_power=float(out_dict["mean_power"]),
                operational_cost=float(out_dict["operational_cost"]),
                true_utility=float(out_dict["true_utility"]),
                ground_truth_states=np.empty((0, 12)),
                ground_truth_observations=np.empty((0, 8)),
            )
        return cls(
            scenario_id=str(data["scenario_id"]),
            scenario_name=str(data["scenario_name"]),
            description=str(data["description"]),
            intervention_time=int(data["intervention_time"]),
            true_latent_state_t_star=data["true_latent_state_t_star"],
            true_state_history=data["true_state_history"],
            exogenous_noise_history=data["exogenous_noise_history"],
            candidate_outcomes=outcomes,
            oracle_optimal_candidate_id=str(data["oracle_optimal_candidate_id"]),
            expected_decision_class=DecisionClass(str(data["expected_decision_class"])),
            oracle_rationale=str(data["oracle_rationale"]),
        )

