"""Generate 5-Class Diagnostic Representation Dataset for Task 5.9.

Generates 250 total evaluation episodes across 5 distinct thermal/transient classes:
- Class A: Nominal Operation (T_core < 85°C)
- Class B: High-Safe Operation (85°C <= T_core < 94°C)
- Class C: Safety-Boundary Operation (94°C <= T_core <= 105°C)
- Class D: Catastrophic Runaway (T_core > 105°C)
- Class E: Acute Shocks (Delta_T >= +15°C over 1-3 steps)
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from prism.dataset.schema import LearnerEpisode, SplitType
from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.actions import ActionVector
from prism.simulator.observations import ObservationVector
from prism.simulator.policies import NominalController, HighLoadController, BasePolicy


class ConstantActionPolicy(BasePolicy):
    """Executes fixed actuator commands."""
    def __init__(self, valve: float, throttle: float, pump: int, flush: int = 0) -> None:
        super().__init__()
        self.valve = valve
        self.throttle = throttle
        self.pump = pump
        self.flush = flush

    def select_action(self, step: int, obs: ObservationVector, prev_action: Optional[ActionVector] = None) -> ActionVector:
        return ActionVector(
            A_valve=float(self.valve),
            A_throttle=float(self.throttle),
            A_pump=int(self.pump),
            A_flush=int(self.flush),
        )


class ThermalShockPolicy(BasePolicy):
    """Executes nominal control then induces an acute thermal shock at t_shock."""
    def __init__(self, t_shock: int = 30, pre_valve: float = 65.0, shock_valve: float = 2.0, shock_throttle: float = 100.0) -> None:
        super().__init__()
        self.t_shock = t_shock
        self.pre_valve = pre_valve
        self.shock_valve = shock_valve
        self.shock_throttle = shock_throttle

    def select_action(self, step: int, obs: ObservationVector, prev_action: Optional[ActionVector] = None) -> ActionVector:
        if step < self.t_shock:
            return ActionVector(A_valve=self.pre_valve, A_throttle=45.0, A_pump=2, A_flush=0)
        else:
            return ActionVector(A_valve=self.shock_valve, A_throttle=self.shock_throttle, A_pump=1, A_flush=0)


def generate_diagnostic_dataset(out_dir: str | Path = "data/diagnostic_representation_dataset", num_per_class: int = 50) -> Dict[str, Any]:
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    sim = THCSimulator(seed=42)
    
    classes = ["class_a_nominal", "class_b_high_safe", "class_c_boundary", "class_d_runaway", "class_e_acute_shocks"]
    class_stats = {c: [] for c in classes}

    for cls_name in classes:
        cls_dir = out_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_per_class):
            ep_id = f"ep_{cls_name}_{i:03d}"
            
            if cls_name == "class_a_nominal":
                # Strict Nominal (< 85°C, target ~ 68-75°C)
                init_s = StateVector(
                    T_core=float(rng.uniform(68.0, 74.0)),
                    T_cool=float(rng.uniform(32.0, 36.0)),
                    P_sys=float(rng.uniform(3.0, 3.3)),
                    F_cool=float(rng.uniform(32.0, 38.0)),
                    L_cpu=float(rng.uniform(40.0, 55.0)),
                    V_pos=float(rng.uniform(60.0, 75.0)),
                    Vib_pump=4.0, P_elec=1.8, T_amb=25.0, W_wear=0.05, Q_internal=0.15, xi_leak=0.0
                )
                policy = NominalController(target_temp=float(rng.uniform(68.0, 72.0)), kp=2.5, ki=0.05)
                
            elif cls_name == "class_b_high_safe":
                # High-Safe (85°C <= T_core < 94°C)
                init_s = StateVector(
                    T_core=float(rng.uniform(86.0, 91.0)),
                    T_cool=float(rng.uniform(40.0, 44.0)),
                    P_sys=float(rng.uniform(3.2, 3.5)),
                    F_cool=float(rng.uniform(24.0, 30.0)),
                    L_cpu=float(rng.uniform(65.0, 75.0)),
                    V_pos=float(rng.uniform(45.0, 55.0)),
                    Vib_pump=4.5, P_elec=2.2, T_amb=28.0, W_wear=0.08, Q_internal=0.25, xi_leak=0.0
                )
                policy = NominalController(target_temp=float(rng.uniform(87.0, 91.0)), kp=2.0, ki=0.05)
                
            elif cls_name == "class_c_boundary":
                # Boundary Margin (94°C <= T_core <= 104°C)
                init_s = StateVector(
                    T_core=float(rng.uniform(95.0, 99.0)),
                    T_cool=float(rng.uniform(45.0, 50.0)),
                    P_sys=float(rng.uniform(3.3, 3.6)),
                    F_cool=float(rng.uniform(18.0, 24.0)),
                    L_cpu=float(rng.uniform(80.0, 90.0)),
                    V_pos=float(rng.uniform(35.0, 45.0)),
                    Vib_pump=4.8, P_elec=2.5, T_amb=32.0, W_wear=0.12, Q_internal=0.35, xi_leak=0.0
                )
                policy = NominalController(target_temp=float(rng.uniform(96.0, 101.0)), kp=1.5, ki=0.03)
                
            elif cls_name == "class_d_runaway":
                # Severe runaway (T_core > 105°C up to 135°C)
                init_s = StateVector(
                    T_core=float(rng.uniform(108.0, 120.0)),
                    T_cool=float(rng.uniform(55.0, 65.0)),
                    P_sys=float(rng.uniform(4.0, 5.0)),
                    F_cool=float(rng.uniform(6.0, 12.0)),
                    L_cpu=float(rng.uniform(92.0, 100.0)),
                    V_pos=float(rng.uniform(10.0, 20.0)),
                    Vib_pump=6.5, P_elec=2.8, T_amb=40.0, W_wear=0.25, Q_internal=0.60, xi_leak=0.01
                )
                policy = ConstantActionPolicy(valve=rng.uniform(5.0, 15.0), throttle=100.0, pump=1)
                
            elif cls_name == "class_e_acute_shocks":
                # Starts nominal (T ~ 70-76°C), acute shock at t=30 resulting in rapid Delta_T jump
                t_shock = int(rng.integers(25, 35))
                init_s = StateVector(
                    T_core=float(rng.uniform(70.0, 75.0)),
                    T_cool=float(rng.uniform(32.0, 36.0)),
                    P_sys=float(rng.uniform(3.1, 3.3)),
                    F_cool=float(rng.uniform(32.0, 36.0)),
                    L_cpu=45.0, V_pos=65.0, Vib_pump=4.0, P_elec=1.8, T_amb=25.0, W_wear=0.05, Q_internal=0.15, xi_leak=0.0
                )
                policy = ThermalShockPolicy(t_shock=t_shock, pre_valve=65.0, shock_valve=rng.uniform(1.0, 5.0), shock_throttle=100.0)
            
            # Simulate 60 timesteps
            init_obs, _ = sim.reset(initial_state=init_s)
            obs_list = [init_obs.to_array()]
            act_list = []
            curr_obs = init_obs
            prev_act = None
            
            for t in range(59):
                act = policy.select_action(t, curr_obs, prev_act)
                act_list.append(act.to_array())
                obs_v, _, _, _ = sim.step(act)
                obs_list.append(obs_v.to_array())
                curr_obs = obs_v
                prev_act = act
                
            act_list.append(policy.select_action(59, curr_obs, prev_act).to_array())
                
            obs_arr = np.array(obs_list) # (60, 8)
            act_arr = np.array(act_list) # (60, 4)
            mask_arr = np.ones_like(obs_arr, dtype=bool)
            
            peak_T = float(obs_arr[:, 0].max())
            mean_T = float(obs_arr[:, 0].mean())
            max_delta_T = float(np.max(obs_arr[1:, 0] - obs_arr[:-1, 0]))
            
            class_stats[cls_name].append({
                "ep_id": ep_id,
                "peak_T": peak_T,
                "mean_T": mean_T,
                "max_delta_T": max_delta_T,
            })
            
            lep = LearnerEpisode(
                episode_id=ep_id,
                split=SplitType.TEST,
                timestamps=np.arange(60),
                observations=obs_arr,
                observation_mask=mask_arr,
                actions=act_arr,
                metadata={"class": cls_name, "peak_T": peak_T, "max_delta_T": max_delta_T}
            )
            lep.save_npz(cls_dir / f"{ep_id}.npz")

    print("=========================================================================")
    print("      DIAGNOSTIC REPRESENTATION DATASET SUMMARY (TASK 5.9)               ")
    print("=========================================================================")
    for cls_name in classes:
        peaks = [e["peak_T"] for e in class_stats[cls_name]]
        means = [e["mean_T"] for e in class_stats[cls_name]]
        dTs = [e["max_delta_T"] for e in class_stats[cls_name]]
        print(f"Class: {cls_name:<22} | Episodes: {len(peaks)} | Peak T: [{min(peaks):5.1f}, {max(peaks):5.1f}]°C (mean {np.mean(peaks):5.1f}) | Max Delta_T: {max(dTs):5.1f}°C/step")
    print(f"All episodes saved to {out_path}")
    return class_stats


if __name__ == "__main__":
    generate_diagnostic_dataset()
