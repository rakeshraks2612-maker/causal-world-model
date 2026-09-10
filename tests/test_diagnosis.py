"""Tests for Diagnostic Tools: Domain Metrics, Reconstruction vs Prediction, Latent Probes, and Action Sensitivity."""

from __future__ import annotations
import pytest
import torch
import numpy as np

from prism.dataset.schema import LearnerEpisode, OracleEpisode, SplitType
from prism.world_model.config import WorldModelConfig
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.training.dataset import PrismWindowedDataset
from prism.training.batching import create_dataloader
from prism.diagnosis.domain_metrics import compute_comprehensive_metrics, DOMAINS
from prism.diagnosis.reconstruction_vs_prediction import evaluate_reconstruction_vs_prediction
from prism.diagnosis.latent_probes import fit_and_evaluate_latent_probes
from prism.diagnosis.action_sensitivity import evaluate_action_sensitivity


@pytest.fixture
def synthetic_episodes():
    torch.manual_seed(42)
    np.random.seed(42)
    T = 50
    timestamps = np.arange(T, dtype=np.int64)

    train_eps = []
    test_eps = []

    for i in range(4):
        obs = np.random.uniform(20.0, 80.0, size=(T, 8))
        mask = np.ones((T, 8), dtype=bool)
        act = np.random.uniform(10.0, 90.0, size=(T, 4))
        gt = np.concatenate([obs, np.random.uniform(5.0, 30.0, size=(T, 4))], axis=1) # 12-dim
        noise = np.random.randn(T, 12)

        ep = OracleEpisode(
            episode_id=f"ep_train_{i}",
            seed=42 + i,
            split=SplitType.TRAIN,
            timestamps=timestamps,
            observations=obs,
            observation_mask=mask,
            actions=act,
            ground_truth_states=gt,
            exogenous_noise=noise,
        )
        train_eps.append(ep)

    for i in range(2):
        obs = np.random.uniform(20.0, 80.0, size=(T, 8))
        mask = np.ones((T, 8), dtype=bool)
        act = np.random.uniform(10.0, 90.0, size=(T, 4))
        gt = np.concatenate([obs, np.random.uniform(5.0, 30.0, size=(T, 4))], axis=1)
        noise = np.random.randn(T, 12)

        ep = OracleEpisode(
            episode_id=f"ep_test_{i}",
            seed=100 + i,
            split=SplitType.TEST,
            timestamps=timestamps,
            observations=obs,
            observation_mask=mask,
            actions=act,
            ground_truth_states=gt,
            exogenous_noise=noise,
        )
        test_eps.append(ep)

    return train_eps, test_eps


def test_comprehensive_metrics_domain_grouping(synthetic_episodes):
    train_eps, test_eps = synthetic_episodes
    learner_train = [ep.to_learner_episode() for ep in train_eps]
    normalizer = ObservationNormalizer.fit(learner_train)

    preds = torch.randn(20, 8) * 10.0 + 50.0
    targets = torch.randn(20, 8) * 10.0 + 50.0
    mask = torch.ones(20, 8)

    report = compute_comprehensive_metrics(preds, targets, mask, normalizer, model_name="TestModel")
    assert report.model_name == "TestModel"
    assert "thermal" in report.domains
    assert "hydraulic" in report.domains
    assert "compute" in report.domains
    assert len(report.per_channel) == 8
    assert report.train_normalized_aggregate_mae >= 0.0


def test_reconstruction_vs_prediction_evaluator(synthetic_episodes):
    train_eps, test_eps = synthetic_episodes
    learner_train = [ep.to_learner_episode() for ep in train_eps]
    learner_test = [ep.to_learner_episode() for ep in test_eps]

    normalizer = ObservationNormalizer.fit(learner_train)
    test_ds = PrismWindowedDataset(learner_test, context_length=10, prediction_length=1, stride=5, normalizer=normalizer)
    test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False)

    model = CausalWorldModel(WorldModelConfig())
    recon_rep, pred_rep = evaluate_reconstruction_vs_prediction(model, test_loader, normalizer, deterministic=True)

    assert recon_rep.raw_overall_mae >= 0.0
    assert pred_rep.raw_overall_mae >= 0.0
    assert "T_core" in recon_rep.per_channel
    assert "T_core" in pred_rep.per_channel


def test_latent_probes_evaluator(synthetic_episodes):
    train_eps, test_eps = synthetic_episodes
    learner_train = [ep.to_learner_episode() for ep in train_eps]
    normalizer = ObservationNormalizer.fit(learner_train)

    model = CausalWorldModel(WorldModelConfig())
    probe_rep = fit_and_evaluate_latent_probes(model, train_eps, test_eps, normalizer)

    assert probe_rep.latent_dim == 32
    assert "T_amb" in probe_rep.probe_metrics
    assert "W_wear" in probe_rep.probe_metrics
    assert "Q_internal" in probe_rep.probe_metrics
    assert "xi_leak" in probe_rep.probe_metrics


def test_action_sensitivity_evaluator(synthetic_episodes):
    train_eps, _ = synthetic_episodes
    learner_train = [ep.to_learner_episode() for ep in train_eps]
    normalizer = ObservationNormalizer.fit(learner_train)

    model = CausalWorldModel(WorldModelConfig())
    act_rep = evaluate_action_sensitivity(model, normalizer)

    assert act_rep.valve_sweep.action_name == "A_valve"
    assert act_rep.throttle_sweep.action_name == "A_throttle"
    assert len(act_rep.valve_sweep.sweep_values) == 3
