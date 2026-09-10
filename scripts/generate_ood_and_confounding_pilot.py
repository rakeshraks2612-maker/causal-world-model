"""OOD and Confounding Pilot Dataset Generator and Distribution Comparison.

Generates:
- 25 OOD Pilot Episodes (5 per OOD regime)
- 10 Confounding Pilot Episodes (paired observational vs interventional)
and prints a distribution comparison table against the Training distribution.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode
from prism.dataset.scenarios import OODRegime, generate_ood_episode, generate_confounding_episode
from prism.dataset.validators import validate_learner_isolation
from prism.simulator.interventions import Intervention
from prism.simulator.state import OBSERVABLE_VARIABLES, LATENT_VARIABLES


def run_ood_and_confounding_pilot(output_dir: str = "data/pilot") -> None:
    base_path = Path(output_dir)
    print(f"Generating OOD & Confounding pilot datasets to {base_path}...")

    # 1. Generate OOD Episodes (5 per regime)
    ood_oracle_list: list[OracleEpisode] = []
    ood_learner_list: list[LearnerEpisode] = []

    for regime in OODRegime:
        regime_oracle_dir = base_path / "oracle" / "ood" / regime.value
        regime_learner_dir = base_path / "learner" / "ood" / regime.value
        regime_oracle_dir.mkdir(parents=True, exist_ok=True)
        regime_learner_dir.mkdir(parents=True, exist_ok=True)

        print(f"-> Generating 5 episodes for {regime.value}...")
        for i in range(5):
            oracle_ep = generate_ood_episode(regime, index=i, length=120)
            learner_ep = oracle_ep.to_learner_episode()
            validate_learner_isolation(learner_ep)

            oracle_ep.save_npz(regime_oracle_dir / f"{oracle_ep.episode_id}.npz")
            learner_ep.save_npz(regime_learner_dir / f"{learner_ep.episode_id}.npz")

            ood_oracle_list.append(oracle_ep)
            ood_learner_list.append(learner_ep)

    # 2. Generate Confounding Episodes (10 paired)
    conf_oracle_dir = base_path / "oracle" / "confounding"
    conf_learner_dir = base_path / "learner" / "confounding"
    conf_oracle_dir.mkdir(parents=True, exist_ok=True)
    conf_learner_dir.mkdir(parents=True, exist_ok=True)

    print("-> Generating 10 Confounding Episodes (Summer vs Winter, Obs vs Intervened)...")
    for i in range(5):
        # Summer peak observational vs do(L_cpu=20%)
        ep_obs = generate_confounding_episode("summer_peak", index=i*2, intervention=None, length=120)
        ep_inv = generate_confounding_episode(
            "summer_peak", index=i*2+1, intervention=Intervention("L_cpu", 20.0, start_step=0), length=120
        )
        for ep in [ep_obs, ep_inv]:
            lep = ep.to_learner_episode()
            validate_learner_isolation(lep)
            ep.save_npz(conf_oracle_dir / f"{ep.episode_id}.npz")
            lep.save_npz(conf_learner_dir / f"{lep.episode_id}.npz")

    # 3. Print Distribution Comparison
    print("\n" + "=" * 90)
    print(" " * 25 + "TRAINING vs OOD DISTRIBUTION COMPARISON")
    print("=" * 90)

    train_obs_files = list((base_path / "learner" / "train").glob("*.npz"))
    if train_obs_files:
        train_eps = [LearnerEpisode.load_npz(f) for f in train_obs_files]
        all_train_obs = np.vstack([ep.observations for ep in train_eps])

        print(f"\n[TRAIN DISTRIBUTION] N = {len(train_eps)} episodes ({len(all_train_obs)} steps):")
        for idx, var_name in enumerate(OBSERVABLE_VARIABLES):
            col = all_train_obs[:, idx]
            print(f"  {var_name:12s} | Mean: {np.nanmean(col):6.2f} | Std: {np.nanstd(col):5.2f} | Range: [{np.nanmin(col):6.2f}, {np.nanmax(col):6.2f}]")

    for regime in OODRegime:
        reg_files = list((base_path / "learner" / "ood" / regime.value).glob("*.npz"))
        if reg_files:
            reg_eps = [LearnerEpisode.load_npz(f) for f in reg_files]
            all_reg_obs = np.vstack([ep.observations for ep in reg_eps])
            print(f"\n[{regime.value.upper()}] N = {len(reg_eps)} episodes ({len(all_reg_obs)} steps):")
            for idx, var_name in enumerate(OBSERVABLE_VARIABLES):
                col = all_reg_obs[:, idx]
                print(f"  {var_name:12s} | Mean: {np.nanmean(col):6.2f} | Std: {np.nanstd(col):5.2f} | Range: [{np.nanmin(col):6.2f}, {np.nanmax(col):6.2f}]")

    print("\n" + "=" * 90)
    print("OOD & Confounding pilot dataset successfully generated and verified.")
    print("=" * 90)


if __name__ == "__main__":
    run_ood_and_confounding_pilot()
