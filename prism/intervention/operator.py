"""Intervention Operator for Learned Causal World Models.

Executes strict Pearl do-calculus graph surgery vs action control:
- Class A: Atomic State Clamps do(X = x)
    - Directly severs parent arrows (PA(X) -> None)
    - Action inputs A_t are UNMODIFIED (parent setpoint is cut)
    - Decoded state channel X is strictly clamped to x for all active steps
- Class B: Action Controls A = a
    - Modifies controller setpoint A_t
    - State variables evolve through natural dynamical transitions (actuator lag)
- Non-Descendant Invariance:
    - Non-descendant clamps (e.g. do(Vib_pump)) leave all other state channels invariant
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Set
import numpy as np
import torch
from torch import Tensor

from prism.simulator.state import OBSERVABLE_VARIABLES
from prism.simulator.actions import ACTION_VARIABLES
from prism.intervention.spec import InterventionSpec, InterventionType


# Mapping of observable variables to channel indices
OBSERVABLE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(OBSERVABLE_VARIABLES)}
ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ACTION_VARIABLES)}

# Variables that are strictly non-causal on thermal/fluid system dynamics (leaf sensors)
NON_CAUSAL_LEAF_VARIABLES: Set[str] = {
    "Vib_pump",
}


class InterventionOperator:
    """Applies graph-surgical interventions and action overrides during world model rollout."""

    def __init__(self, specs: Optional[Union[InterventionSpec, List[InterventionSpec]]] = None) -> None:
        if specs is None:
            self.specs: List[InterventionSpec] = []
        elif isinstance(specs, InterventionSpec):
            self.specs = [specs]
        else:
            self.specs = list(specs)

    def add_spec(self, spec: InterventionSpec) -> None:
        """Add an intervention specification."""
        self.specs.append(spec)

    def clear(self) -> None:
        """Clear all registered specifications."""
        self.specs.clear()

    @property
    def has_action_controls(self) -> bool:
        """True if any registered intervention is an action control."""
        return any(s.is_action_control for s in self.specs)

    @property
    def has_state_clamps(self) -> bool:
        """True if any registered intervention is a state clamp."""
        return any(s.is_state_clamp for s in self.specs)

    def get_modified_actions(
        self,
        future_actions: Tensor | np.ndarray,
        t_star: int,
    ) -> Tensor:
        """Apply action overrides to future action sequence for Class B action interventions.
        
        Strict Graph Surgery Invariant:
        State clamps do(X = x) do NOT modify the action sequence! The incoming causal arrow
        PA(X) -> X is cut, leaving the upstream control actions invariant while X is clamped.
        
        Args:
            future_actions: Action tensor of shape [B, H, 4] or [H, 4] starting at timestep t_star
            t_star: Global simulation timestep corresponding to horizon step 0
            
        Returns:
            Modified action tensor with identical shape
        """
        is_numpy = isinstance(future_actions, np.ndarray)
        if is_numpy:
            acts = torch.tensor(future_actions, dtype=torch.float32)
        else:
            acts = future_actions.clone()

        has_batch = (acts.ndim == 3)
        if not has_batch:
            acts = acts.unsqueeze(0)  # [1, H, 4]

        batch_size, horizon, act_dim = acts.shape

        for spec in self.specs:
            # ONLY modify actions for ACTION_CONTROL interventions!
            # State clamps do(X=x) explicitly cut PA(X) -> X and do NOT modify actions.
            if not spec.is_action_control:
                continue

            if spec.target in ACTION_TO_IDX:
                idx = ACTION_TO_IDX[spec.target]
                for step in range(horizon):
                    global_step = t_star + step
                    if spec.is_active(global_step):
                        acts[:, step, idx] = spec.value

        if not has_batch:
            acts = acts.squeeze(0)

        return acts

    def apply_observation_clamps(
        self,
        observations: Tensor | np.ndarray,
        t_star: int,
        normalizer: Optional[Any] = None,
        is_normalized: bool = False,
    ) -> Tensor:
        """Apply state clamps do(X_j = x) directly to decoded observations.
        
        Args:
            observations: Decoded observation tensor [B, H, 8] or [H, 8] starting at t_star
            t_star: Global simulation timestep corresponding to horizon step 0
            normalizer: Optional ObservationNormalizer to handle normalized values
            is_normalized: If True, clamp values are normalized before insertion
            
        Returns:
            Clamped observation tensor with identical shape
        """
        is_numpy = isinstance(observations, np.ndarray)
        if is_numpy:
            obs = torch.tensor(observations, dtype=torch.float32)
        else:
            obs = observations.clone()

        has_batch = (obs.ndim == 3)
        if not has_batch:
            obs = obs.unsqueeze(0)  # [1, H, 8]

        batch_size, horizon, obs_dim = obs.shape

        for spec in self.specs:
            if not spec.is_state_clamp:
                continue

            if spec.target not in OBSERVABLE_TO_IDX:
                continue

            idx = OBSERVABLE_TO_IDX[spec.target]
            target_val = spec.value

            if is_normalized and normalizer is not None:
                # Convert clamp value to normalized space
                mean_val = float(normalizer.stats.means[idx])
                std_val = float(normalizer.stats.stds[idx])
                clamp_val = (target_val - mean_val) / std_val
            else:
                clamp_val = target_val

            for step in range(horizon):
                global_step = t_star + step
                if spec.is_active(global_step):
                    obs[:, step, idx] = clamp_val

        if not has_batch:
            obs = obs.squeeze(0)

        return obs
