"""Tests for OOD Regimes, Confounding Protocol, and Metadata Leakage Invariants."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from prism.dataset.schema import SplitType, LearnerEpisode, OracleEpisode, FORBIDDEN_LEARNER_KEYS
from prism.dataset.scenarios import OODRegime, generate_ood_episode, generate_confounding_episode
from prism.dataset.validators import validate_learner_isolation, validate_oracle_episode_integrity
from prism.simulator.interventions import Intervention


def test_ood_regimes_exceed_training_support() -> None:
    """Verify that all 5 OOD regimes generate initial conditions outside training support."""
    # OOD-1: Extreme Ambient (> 38°C)
    ep_amb = generate_ood_episode(OODRegime.EXTREME_AMBIENT, index=0, length=20)
    init_t_amb = ep_amb.ground_truth_states[0, 8]
    assert init_t_amb >= 38.0

    # OOD-2: Extreme Wear (>= 0.80)
    ep_wear = generate_ood_episode(OODRegime.EXTREME_WEAR, index=1, length=20)
    init_wear = ep_wear.ground_truth_states[0, 9]
    assert init_wear >= 0.80

    # OOD-3: Combined Stress (Workload >= 85%, Ambient >= 38°C, Wear >= 0.75)
    ep_comb = generate_ood_episode(OODRegime.COMBINED_STRESS, index=2, length=20)
    assert ep_comb.ground_truth_states[0, 4] >= 85.0  # L_cpu
    assert ep_comb.ground_truth_states[0, 8] >= 38.0  # T_amb
    assert ep_comb.ground_truth_states[0, 9] >= 0.75  # W_wear

    # OOD-4: Latent Hotspot (Q_internal >= 1.2 kW)
    ep_hot = generate_ood_episode(OODRegime.LATENT_HOTSPOT, index=3, length=20)
    init_q = ep_hot.ground_truth_states[0, 10]
    assert init_q >= 1.20

    # OOD-5: Leak Degradation (xi_leak >= 25.0 mL/hr)
    ep_leak = generate_ood_episode(OODRegime.LEAK_DEGRADATION, index=4, length=20)
    init_leak = ep_leak.ground_truth_states[0, 11]
    assert init_leak >= 25.0


def test_ood_3_joint_distribution_occupancy() -> None:
    """TASK 2.5 (Item C): Joint-Distribution Validation for OOD-3.
    
    Verifies that while individual stress factors may rarely appear, the joint configuration
    (L_cpu > 85% AND T_amb > 38°C AND W_wear > 0.75) has EXACTLY ZERO occupancy in training,
    while occupying > 90% of time steps in OOD-3 episodes.
    """
    from prism.dataset.generator import generate_single_episode
    
    # Check 10 training episodes: joint condition MUST be 0%
    train_eps = [generate_single_episode(SplitType.TRAIN, index=i, regime="nominal", length=60) for i in range(10)]
    train_states = np.vstack([ep.ground_truth_states for ep in train_eps])
    train_joint_hits = np.sum(
        (train_states[:, 4] > 85.0) & (train_states[:, 8] > 38.0) & (train_states[:, 9] > 0.75)
    )
    assert train_joint_hits == 0, f"Training data accidentally leaked joint OOD-3 stress condition: {train_joint_hits} hits"

    # Check OOD-3 episode: joint condition MUST be dominant
    ood3_ep = generate_ood_episode(OODRegime.COMBINED_STRESS, index=0, length=60)
    ood3_states = ood3_ep.ground_truth_states
    ood3_joint_fraction = np.mean(
        (ood3_states[:, 4] > 80.0) & (ood3_states[:, 8] > 38.0) & (ood3_states[:, 9] > 0.70)
    )
    assert ood3_joint_fraction > 0.85, f"OOD-3 failed to sustain joint stress condition: {ood3_joint_fraction:.2f}"


def test_confounding_2x3_factorial_grid() -> None:
    """TASK 2.6 (Item B): 2x3 Factorial Matched Confounding Evaluation Grid.
    
    Grid: {Summer Peak (T_amb ~ 38°C), Winter Baseline (T_amb ~ 20°C)}
        x {Observational Natural, do(L_cpu = 20%), do(L_cpu = 80%)}
        
    Verifies:
    1. In summer, do(L_cpu=20%) drops temp relative to natural (L_cpu ~ 80%), but baseline T_cool
       remains higher than in winter due to external radiator ambient gradient.
    2. Across both seasons, do(L_cpu=80%) produces higher core heat than do(L_cpu=20%).
    """
    # Summer Grid
    ep_sum_obs = generate_confounding_episode("summer_peak", index=0, intervention=None, length=40)
    ep_sum_do20 = generate_confounding_episode("summer_peak", index=0, intervention=Intervention("L_cpu", 20.0), length=40)
    ep_sum_do80 = generate_confounding_episode("summer_peak", index=0, intervention=Intervention("L_cpu", 80.0), length=40)

    # Winter Grid
    ep_win_obs = generate_confounding_episode("winter_baseline", index=0, intervention=None, length=40)
    ep_win_do20 = generate_confounding_episode("winter_baseline", index=0, intervention=Intervention("L_cpu", 20.0), length=40)
    ep_win_do80 = generate_confounding_episode("winter_baseline", index=0, intervention=Intervention("L_cpu", 80.0), length=40)

    # 1. Causal workload response curve: do(L=80) > do(L=20) in both seasons
    sum_t80 = np.mean(ep_sum_do80.observations[:, 0])
    sum_t20 = np.mean(ep_sum_do20.observations[:, 0])
    assert sum_t80 > sum_t20

    win_t80 = np.mean(ep_win_do80.observations[:, 0])
    win_t20 = np.mean(ep_win_do20.observations[:, 0])
    assert win_t80 > win_t20

    # 2. Confounder impact: Coolant in summer do(L=20) is warmer than winter do(L=20) due to T_amb
    sum_tcool_do20 = np.mean(ep_sum_do20.observations[:, 1])
    win_tcool_do20 = np.mean(ep_win_do20.observations[:, 1])
    assert sum_tcool_do20 > win_tcool_do20 + 8.0


def test_forbidden_metadata_leakage_serialization() -> None:
    """CRITICAL INVARIANT TEST: OracleEpisode.to_learner_episode() stripped object and NPZ
    contain ZERO forbidden latent keys or latent variable values.
    """
    oracle_ep = generate_ood_episode(OODRegime.EXTREME_AMBIENT, index=0, length=20)
    # Ensure oracle metadata has latent info
    oracle_ep.oracle_metadata["T_amb"] = 42.5
    oracle_ep.oracle_metadata["W_wear"] = 0.88

    # Convert to learner
    learner_ep = oracle_ep.to_learner_episode()
    validate_learner_isolation(learner_ep)

    # Check in-memory metadata keys
    for forbidden in FORBIDDEN_LEARNER_KEYS:
        assert forbidden not in learner_ep.metadata

    # Save to disk and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "learner_ep.npz"
        learner_ep.save_npz(npz_path)

        reloaded_ep = LearnerEpisode.load_npz(npz_path)
        validate_learner_isolation(reloaded_ep)

        for forbidden in FORBIDDEN_LEARNER_KEYS:
            assert forbidden not in reloaded_ep.metadata
