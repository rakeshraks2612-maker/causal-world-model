"""Structural Dynamics and Transition Equations for THC-SCM System.

Implements the ground truth structural causal equations f_SCM(X_t, A_t, U_t, interventions)
with rigorous dependency ordering and Pearl's Level 2 graph-surgery overrides.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.noise import NoiseVector
from prism.simulator.parameters import SystemConstants
from prism.simulator.interventions import InterventionRegistry


class StructuralDynamics:
    """Computes single-step state transition X_{t+1} = f_SCM(X_t, A_t, U_t)."""

    def __init__(self, constants: Optional[SystemConstants] = None) -> None:
        self.constants = constants or SystemConstants.load_default()

    def step(
        self,
        step_idx: int,
        state: StateVector,
        action: ActionVector,
        noise: NoiseVector,
        interventions: Optional[InterventionRegistry] = None,
        next_t_amb_override: Optional[float] = None,
    ) -> StateVector:
        """Execute one discrete-time step forward through the structural equations.
        
        Args:
            step_idx: Current simulation step index t
            state: Current StateVector X_t
            action: Control ActionVector A_t
            noise: Realized Exogenous NoiseVector U_t
            interventions: Optional InterventionRegistry with active do(X=x) overrides
            next_t_amb_override: Precomputed ambient temperature from Ornstein-Uhlenbeck step
            
        Returns:
            Next StateVector X_{t+1}
        """
        p = self.constants.physics
        inv = interventions

        # -------------------------------------------------------------
        # 1. Ambient Temperature T_amb (Confounder)
        # -------------------------------------------------------------
        if inv and inv.has_intervention("T_amb", step_idx):
            next_t_amb = float(inv.get_intervened_value("T_amb", step_idx))
        elif next_t_amb_override is not None:
            next_t_amb = float(next_t_amb_override)
        else:
            # Fallback drift if not provided by RNG manager
            drift = p.delta_t * 0.02 * (27.0 - state.T_amb)
            next_t_amb = float(state.T_amb + drift + noise.u_amb)

        # -------------------------------------------------------------
        # 2. Latent States: Hotspot Flux, Wear, Micro-Leak
        # -------------------------------------------------------------
        # Hotspot flux autoregression
        if inv and inv.has_intervention("Q_internal", step_idx):
            next_q_internal = float(inv.get_intervened_value("Q_internal", step_idx))
        else:
            next_q_internal = float(
                p.q_hotspot_ar * state.Q_internal
                + (1.0 - p.q_hotspot_ar) * p.q_hotspot_mean
                + noise.u_q_int
            )

        # Mechanical and thermal wear accumulation
        if inv and inv.has_intervention("W_wear", step_idx):
            next_w_wear = float(inv.get_intervened_value("W_wear", step_idx))
        else:
            delta_wear = p.lambda_wear * max(0.0, state.T_core - p.t_wear_threshold) * p.delta_t
            flush_relief = p.flush_wear_relief * float(action.A_flush)
            next_w_wear = float(
                np.clip(state.W_wear + delta_wear - flush_relief + noise.u_w, 0.0, 1.0)
            )

        # Hydraulic seal micro-leak rate
        if inv and inv.has_intervention("xi_leak", step_idx):
            next_xi_leak = float(inv.get_intervened_value("xi_leak", step_idx))
        else:
            delta_leak = p.lambda_leak * max(0.0, state.P_sys - p.p_leak_threshold) * p.delta_t
            next_xi_leak = float(
                np.clip(state.xi_leak + delta_leak + noise.u_xi, 0.0, 50.0)
            )

        # -------------------------------------------------------------
        # 3. Compute Workload L_cpu
        # -------------------------------------------------------------
        if inv and inv.has_intervention("L_cpu", step_idx):
            next_l_cpu = float(inv.get_intervened_value("L_cpu", step_idx))
        else:
            # Ambient-confounded natural demand capped by A_throttle
            natural_demand = 35.0 + 1.2 * next_t_amb
            target_load = min(float(action.A_throttle), natural_demand)
            # Smooth tracking with inertia
            next_l_cpu = float(
                np.clip(state.L_cpu + 0.35 * (target_load - state.L_cpu) + noise.u_q, 0.0, 100.0)
            )

        # -------------------------------------------------------------
        # 4. Electrical Power Draw P_elec
        # -------------------------------------------------------------
        if inv and inv.has_intervention("P_elec", step_idx):
            next_p_elec = float(inv.get_intervened_value("P_elec", step_idx))
        else:
            load_factor = (next_l_cpu / 100.0) ** p.p_elec_exp
            next_p_elec = float(
                np.clip(p.p_idle + p.beta_l * load_factor + noise.u_p_elec, 0.1, 6.0)
            )

        # -------------------------------------------------------------
        # 5. Thermal Heat Input Q_in
        # -------------------------------------------------------------
        q_nominal = p.eta_elec * next_p_elec + p.alpha_amb * max(0.0, next_t_amb - 25.0)
        q_in = q_nominal + next_q_internal + noise.u_q

        # -------------------------------------------------------------
        # 6. Coolant Valve Position Actuator V_pos
        # -------------------------------------------------------------
        if inv and inv.has_intervention("V_pos", step_idx):
            # Graph surgery: Severs A_valve and W_wear dependencies!
            next_v_pos = float(np.clip(inv.get_intervened_value("V_pos", step_idx), 0.0, 100.0))
        else:
            # Natural mechanism with wear-induced lag
            tau_eff = p.tau_valve_base * (1.0 + p.valve_wear_lag_scale * next_w_wear)
            delta_v = (p.delta_t / max(0.1, tau_eff)) * (float(action.A_valve) - state.V_pos)
            next_v_pos = float(np.clip(state.V_pos + delta_v + noise.u_v, 0.0, 100.0))

        # -------------------------------------------------------------
        # 7. Pump Head, Flow Rate F_cool, and Hydraulic Pressure P_sys
        # -------------------------------------------------------------
        h_pump = p.p_base * (p.pump_stage_offset + p.pump_stage_scale * float(action.A_pump))

        if inv and inv.has_intervention("F_cool", step_idx):
            next_f_cool = float(np.clip(inv.get_intervened_value("F_cool", step_idx), 0.0, 65.0))
        else:
            head_ratio = np.sqrt(max(0.1, h_pump / p.h_nominal))
            aperture_factor = (next_v_pos / 100.0) ** p.valve_flow_exp
            f_ideal = p.f_max * head_ratio * aperture_factor
            wear_attenuation = 1.0 - p.wear_flow_loss * next_w_wear
            leak_attenuation = max(0.0, 1.0 - (next_xi_leak / 100.0))
            flush_surge = p.flush_flow_surge * float(action.A_flush)

            f_candidate = f_ideal * wear_attenuation * leak_attenuation + flush_surge + noise.u_f
            next_f_cool = float(np.clip(f_candidate, 0.0, 65.0))

        if inv and inv.has_intervention("P_sys", step_idx):
            next_p_sys = float(np.clip(inv.get_intervened_value("P_sys", step_idx), 0.2, 7.0))
        else:
            flow_backpressure = p.k_pressure * ((next_f_cool / p.f_max) ** 2)
            leak_drop = p.delta_leak_pressure * (next_xi_leak / 50.0)
            p_candidate = h_pump + flow_backpressure - leak_drop + noise.u_p
            next_p_sys = float(np.clip(p_candidate, 0.2, 7.0))

        # -------------------------------------------------------------
        # 8. Chassis Acoustic Vibration Vib_pump
        # -------------------------------------------------------------
        if inv and inv.has_intervention("Vib_pump", step_idx):
            next_vib = float(np.clip(inv.get_intervened_value("Vib_pump", step_idx), 0.0, 30.0))
        else:
            vib_candidate = (
                p.gamma_vib_base
                + p.gamma_vib_pump * (float(action.A_pump) ** 1.4)
                + p.gamma_vib_press * (next_p_sys ** 1.6)
                + p.gamma_vib_wear * next_w_wear
                + noise.u_vib
            )
            next_vib = float(np.clip(vib_candidate, 0.0, 30.0))

        # -------------------------------------------------------------
        # 9. Convective Heat Removal Q_out
        # -------------------------------------------------------------
        k_eff = p.k_transfer_base * (1.0 - p.k_transfer_wear_loss * next_w_wear)
        phi_flow = 1.0 - np.exp(-next_f_cool / p.f_scale)
        q_out = k_eff * phi_flow * (state.T_core - state.T_cool)

        # -------------------------------------------------------------
        # 10. Core Temperature T_core
        # -------------------------------------------------------------
        if inv and inv.has_intervention("T_core", step_idx):
            next_t_core = float(np.clip(inv.get_intervened_value("T_core", step_idx), 15.0, 135.0))
        else:
            passive_loss = p.k_ambient_loss * (state.T_core - next_t_amb)
            d_t_core_dt = (q_in - q_out - passive_loss) / p.c_core
            t_core_cand = state.T_core + p.delta_t * d_t_core_dt + noise.u_t_core
            next_t_core = float(np.clip(t_core_cand, 15.0, 135.0))

        # -------------------------------------------------------------
        # 11. Coolant Loop Temperature T_cool
        # -------------------------------------------------------------
        if inv and inv.has_intervention("T_cool", step_idx):
            next_t_cool = float(np.clip(inv.get_intervened_value("T_cool", step_idx), 10.0, 105.0))
        else:
            q_radiator = p.k_radiator * (state.T_cool - next_t_amb)
            flush_cooling = p.flush_temp_drop * float(action.A_flush)
            d_t_cool_dt = (q_out - q_radiator) / p.c_loop
            t_cool_cand = state.T_cool + p.delta_t * d_t_cool_dt - flush_cooling + noise.u_t_cool
            next_t_cool = float(np.clip(t_cool_cand, 10.0, 105.0))

        # Construct new StateVector
        return StateVector(
            T_core=next_t_core,
            T_cool=next_t_cool,
            P_sys=next_p_sys,
            F_cool=next_f_cool,
            L_cpu=next_l_cpu,
            V_pos=next_v_pos,
            Vib_pump=next_vib,
            P_elec=next_p_elec,
            T_amb=next_t_amb,
            W_wear=next_w_wear,
            Q_internal=next_q_internal,
            xi_leak=next_xi_leak,
        )
