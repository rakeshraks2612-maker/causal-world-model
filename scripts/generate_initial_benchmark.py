"""Initial Benchmark Dataset Generator for THC-SCM System.

Generates 70 benchmark episodes across 7 controlled experimental regimes:
1. 10 Nominal Episodes (PI feedback control)
2. 10 High-Load Episodes (Compute stress testing)
3. 10 High-Ambient Episodes (Confounded external heatwave)
4. 10 High-Wear Episodes (Actuator lag & degraded interface)
5. 10 Failure Episodes (Thermal runaway & overpressure)
6. 10 Intervention Episodes (Forced do(V_pos=85%) overrides)
7. 10 Missing-Observation Episodes (Sensor telemetry dropouts)
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np

from prism.simulator.simulator import THCSimulator
from prism.simulator.state import StateVector
from prism.simulator.interventions import InterventionRegistry
from prism.simulator.policies import (
    NominalController,
    HighLoadController,
    AggressiveCoolingController,
    FailureInducingController,
    RecoveryController,
)


def generate_benchmark_suite(output_dir: str | Path = "data/initial_benchmark") -> None:
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating 70 benchmark episodes to {base_path}...")

    # Regime 1: 10 Nominal Episodes
    print("-> Generating 10 Nominal Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=1000 + i)
        ep = sim.run_episode(NominalController(), length=120, episode_id=f"nominal_{i:02d}")
        ep.save_npz(base_path / "nominal" / f"ep_{i:02d}.npz")

    # Regime 2: 10 High-Load Episodes
    print("-> Generating 10 High-Load Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=2000 + i)
        ep = sim.run_episode(HighLoadController(fixed_valve=40.0), length=120, episode_id=f"high_load_{i:02d}")
        ep.save_npz(base_path / "high_load" / f"ep_{i:02d}.npz")

    # Regime 3: 10 High-Ambient Episodes (Confounded heatwave: T_amb = 38°C)
    print("-> Generating 10 High-Ambient Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=3000 + i)
        init_state = StateVector(
            T_core=74.0,
            T_cool=42.0,
            P_sys=3.2,
            F_cool=30.0,
            L_cpu=65.0,
            V_pos=60.0,
            Vib_pump=4.5,
            P_elec=2.2,
            T_amb=38.0,  # Elevated ambient heatwave
            W_wear=0.1,
            Q_internal=0.3,
            xi_leak=0.0,
        )
        ep = sim.run_episode(NominalController(), length=120, initial_state=init_state, episode_id=f"high_amb_{i:02d}")
        ep.save_npz(base_path / "high_ambient" / f"ep_{i:02d}.npz")

    # Regime 4: 10 High-Wear Episodes (Actuator lag & degraded cooling: W_wear = 0.85)
    print("-> Generating 10 High-Wear Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=4000 + i)
        init_state = StateVector(
            T_core=75.0,
            T_cool=38.0,
            P_sys=3.0,
            F_cool=22.0,
            L_cpu=50.0,
            V_pos=50.0,
            Vib_pump=5.8,
            P_elec=1.85,
            T_amb=25.0,
            W_wear=0.85,  # Severe mechanical & interface wear
            Q_internal=0.2,
            xi_leak=5.0,
        )
        ep = sim.run_episode(NominalController(), length=120, initial_state=init_state, episode_id=f"high_wear_{i:02d}")
        ep.save_npz(base_path / "high_wear" / f"ep_{i:02d}.npz")

    # Regime 5: 10 Failure Episodes (Thermal runaway & overpressure)
    print("-> Generating 10 Failure Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=5000 + i)
        ep = sim.run_episode(FailureInducingController(), length=120, episode_id=f"failure_{i:02d}")
        ep.save_npz(base_path / "failure" / f"ep_{i:02d}.npz")

    # Regime 6: 10 Intervention Episodes (Forced do(V_pos=85%) graph surgery at t=40)
    print("-> Generating 10 Intervention Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=6000 + i)
        inv = InterventionRegistry.create_single("V_pos", 85.0, step=40, duration=40)
        ep = sim.run_episode(NominalController(), length=120, interventions=inv, episode_id=f"intervention_{i:02d}")
        ep.save_npz(base_path / "intervention" / f"ep_{i:02d}.npz")

    # Regime 7: 10 Missing-Observation Episodes (Dropped pressure and flow telemetry)
    print("-> Generating 10 Missing-Observation Episodes...")
    for i in range(10):
        sim = THCSimulator(seed=7000 + i)
        ep = sim.run_episode(
            NominalController(),
            length=120,
            missing_channels={"P_sys", "F_cool"},
            episode_id=f"missing_obs_{i:02d}",
        )
        ep.save_npz(base_path / "missing_obs" / f"ep_{i:02d}.npz")

    print("Successfully generated all 70 benchmark episodes.")


if __name__ == "__main__":
    generate_benchmark_suite()
