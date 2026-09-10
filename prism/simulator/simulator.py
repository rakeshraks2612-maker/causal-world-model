"""Top-Level THCSimulator Class for THC-SCM System.

Integrates:
- State Container
- RNG Manager & Exogenous Stochastic Innovations
- Structural Equations & Dynamics
- Intervention Registry & Graph Surgery
- Failure Evaluator & Safety Latching
- Observation Sensor Model & Telemetry Masking
- Deterministic Replay & Episode Generation
"""

from __future__ import annotations
import uuid
from typing import Optional, Set, Tuple, List, Dict, Any
import numpy as np

from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.observations import ObservationVector
from prism.simulator.noise import NoiseVector, RNGManager
from prism.simulator.parameters import SystemConstants
from prism.simulator.dynamics import StructuralDynamics
from prism.simulator.interventions import InterventionRegistry, Intervention
from prism.simulator.failures import FailureEvaluator, FailureState
from prism.simulator.policies import BasePolicy
from prism.simulator.episode import Episode


class THCSimulator:
    """The Oracle Ground-Truth Simulator for the ThermoHydro-Compute System."""

    def __init__(
        self,
        seed: int = 42,
        constants: Optional[SystemConstants] = None,
    ) -> None:
        self.seed = int(seed)
        self.constants = constants or SystemConstants.load_default()
        self.rng_mgr = RNGManager(self.seed)
        self.dynamics = StructuralDynamics(self.constants)
        self.failure_eval = FailureEvaluator(self.constants.safety)

        self.current_step = 0
        self.current_state: StateVector = StateVector.default_nominal()
        self.interventions: InterventionRegistry = InterventionRegistry()

    def reset(
        self,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
        missing_channels: Optional[Set[str]] = None,
    ) -> Tuple[ObservationVector, StateVector]:
        """Reset simulator to initial state and reseed if specified."""
        if seed is not None:
            self.seed = int(seed)
            self.rng_mgr = RNGManager(self.seed)

        self.current_step = 0
        self.current_state = initial_state if initial_state is not None else StateVector.default_nominal()
        self.failure_eval.reset()
        self.interventions.clear()

        # Initial observation
        init_obs = ObservationVector.from_state(
            self.current_state,
            sensor_noise=self.constants.sensors.to_dict(),
            missing_channels=missing_channels,
            rng=self.rng_mgr.rng_sensors,
        )
        return init_obs, self.current_state

    def step(
        self,
        action: ActionVector,
        interventions: Optional[InterventionRegistry] = None,
        missing_channels: Optional[Set[str]] = None,
    ) -> Tuple[ObservationVector, FailureState, StateVector, NoiseVector]:
        """Execute one step forward in time.
        
        Args:
            action: Validated ActionVector
            interventions: Optional active interventions for this step
            missing_channels: Optional telemetry dropout channels
            
        Returns:
            (ObservationVector, FailureState, NextStateVector, NoiseVector)
        """
        active_inv = interventions or self.interventions
        stoch = self.constants.stochastic

        # 1. Sample ambient step (Ornstein-Uhlenbeck)
        next_t_amb, u_amb = self.rng_mgr.sample_ambient_step(
            current_t_amb=self.current_state.T_amb,
            theta_amb=stoch.theta_amb,
            mu_amb=stoch.mu_amb,
            sigma_amb=stoch.sigma_amb,
            delta_t=self.constants.physics.delta_t,
        )

        # 2. Sample physical process innovations
        process_noise = self.rng_mgr.sample_process_innovations(stoch.to_sigmas_dict())
        realized_noise = NoiseVector(
            u_amb=u_amb,
            u_p_elec=process_noise.u_p_elec,
            u_q=process_noise.u_q,
            u_q_int=process_noise.u_q_int,
            u_v=process_noise.u_v,
            u_p=process_noise.u_p,
            u_f=process_noise.u_f,
            u_vib=process_noise.u_vib,
            u_t_core=process_noise.u_t_core,
            u_t_cool=process_noise.u_t_cool,
            u_w=process_noise.u_w,
            u_xi=process_noise.u_xi,
        )

        # 3. Propagate SCM structural equations
        next_state = self.dynamics.step(
            step_idx=self.current_step,
            state=self.current_state,
            action=action,
            noise=realized_noise,
            interventions=active_inv,
            next_t_amb_override=next_t_amb,
        )

        # 4. Evaluate physical failure boundaries (latched)
        fail_status = self.failure_eval.evaluate_step(
            step=self.current_step + 1,
            state=next_state,
            action=action,
        )

        # 5. Generate partial noisy observation
        obs = ObservationVector.from_state(
            state=next_state,
            sensor_noise=self.constants.sensors.to_dict(),
            missing_channels=missing_channels,
            rng=self.rng_mgr.rng_sensors,
        )

        self.current_step += 1
        self.current_state = next_state

        return obs, fail_status, next_state, realized_noise

    def run_episode(
        self,
        policy: BasePolicy,
        length: int = 120,
        initial_state: Optional[StateVector] = None,
        interventions: Optional[InterventionRegistry] = None,
        missing_channels: Optional[Set[str]] = None,
        episode_id: Optional[str] = None,
    ) -> Episode:
        """Execute a full standardized episode trajectory under the specified policy."""
        ep_id = episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        obs, state = self.reset(initial_state=initial_state, missing_channels=missing_channels)

        timestamps = [0.0]
        obs_list = [obs.to_array()]
        actions_list: List[np.ndarray] = []
        states_list = [state.to_array()]
        noise_list = [NoiseVector.zeros().to_array()]

        last_action: Optional[ActionVector] = None
        final_failure_state: FailureState = self.failure_eval.state

        for step in range(length):
            action = policy.select_action(step=step, obs=obs, prev_action=last_action)
            actions_list.append(action.to_array())

            obs, fail_status, state, noise = self.step(
                action=action,
                interventions=interventions,
                missing_channels=missing_channels,
            )

            timestamps.append(float(step + 1) * self.constants.physics.delta_t)
            obs_list.append(obs.to_array())
            states_list.append(state.to_array())
            noise_list.append(noise.to_array())

            last_action = action
            final_failure_state = fail_status

        # If length steps executed, actions_list has length `length`. Pad last action for shape alignment:
        actions_list.append(actions_list[-1])

        return Episode(
            episode_id=ep_id,
            seed=self.seed,
            timestamps=np.array(timestamps, dtype=np.float64),
            observations=np.array(obs_list, dtype=np.float64),
            actions=np.array(actions_list, dtype=np.float64),
            ground_truth_states=np.array(states_list, dtype=np.float64),
            exogenous_noise=np.array(noise_list, dtype=np.float64),
            failure_latched=final_failure_state.failed,
            failure_mode=final_failure_state.failure_mode,
            failure_timestamp=final_failure_state.failure_timestamp,
            metadata={
                "policy": policy.__class__.__name__,
                "length": length,
                "missing_channels": list(missing_channels) if missing_channels else [],
            },
        )

    def replay_episode(
        self,
        recorded_noise: np.ndarray,
        action_sequence: np.ndarray,
        initial_state: StateVector,
        interventions: Optional[InterventionRegistry] = None,
        episode_id: Optional[str] = None,
    ) -> Episode:
        """Deterministic twin-simulator replay: Execute given noise and actions exactly."""
        ep_id = episode_id or f"replay_{uuid.uuid4().hex[:8]}"
        self.failure_eval.reset()
        
        num_steps = len(action_sequence)
        timestamps = [0.0]
        states = [initial_state.to_array()]
        noises = [NoiseVector.zeros().to_array()]
        observations = [
            ObservationVector.from_state(
                initial_state,
                sensor_noise=self.constants.sensors.to_dict(),
                rng=None,  # Exact zero-noise observation baseline for deterministic replay checks
            ).to_array()
        ]

        curr_state = initial_state
        final_fail = self.failure_eval.state

        for step in range(num_steps):
            act = ActionVector.from_array(action_sequence[step])
            noise_vec = NoiseVector.from_array(recorded_noise[step + 1] if (step + 1) < len(recorded_noise) else recorded_noise[step])

            # Propagate dynamics with exact frozen noise
            next_state = self.dynamics.step(
                step_idx=step,
                state=curr_state,
                action=act,
                noise=noise_vec,
                interventions=interventions,
                next_t_amb_override=curr_state.T_amb + 0.02 * (27.0 - curr_state.T_amb) + noise_vec.u_amb,
            )

            fail_status = self.failure_eval.evaluate_step(step=step + 1, state=next_state, action=act)
            obs = ObservationVector.from_state(next_state, sensor_noise=self.constants.sensors.to_dict(), rng=None)

            timestamps.append(float(step + 1) * self.constants.physics.delta_t)
            states.append(next_state.to_array())
            noises.append(noise_vec.to_array())
            observations.append(obs.to_array())

            curr_state = next_state
            final_fail = fail_status

        # Align actions array length
        actions_arr = np.vstack([action_sequence, action_sequence[-1:]])

        return Episode(
            episode_id=ep_id,
            seed=self.seed,
            timestamps=np.array(timestamps, dtype=np.float64),
            observations=np.array(observations, dtype=np.float64),
            actions=actions_arr,
            ground_truth_states=np.array(states, dtype=np.float64),
            exogenous_noise=np.array(noises, dtype=np.float64),
            failure_latched=final_fail.failed,
            failure_mode=final_fail.failure_mode,
            failure_timestamp=final_fail.failure_timestamp,
            metadata={"replay": True},
        )
