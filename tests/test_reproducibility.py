"""Tests for Training Determinism and Reproducibility."""

import pytest
import numpy as np
import torch

from prism.training.seed import set_seed
from prism.world_model.model import CausalWorldModel
from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs


def test_seed_initialization_determinism() -> None:
    """Test that set_seed produces identical model weight initializations."""
    set_seed(42)
    m1 = CausalWorldModel(WorldModelConfig())

    set_seed(42)
    m2 = CausalWorldModel(WorldModelConfig())

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        np.testing.assert_array_equal(p1.detach().numpy(), p2.detach().numpy())


def test_training_step_reproducibility() -> None:
    """Test that identical seeds and data produce bit-identical loss and gradient updates."""
    def run_training_step():
        set_seed(1234)
        model = CausalWorldModel(WorldModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs = torch.randn(2, 10, 8)
        mask = torch.ones(2, 10, 8)
        act = torch.randn(2, 10, 4)
        inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

        optimizer.zero_grad()
        loss = model.compute_loss(inputs).total_loss
        loss.backward()
        optimizer.step()

        return float(loss.item()), [p.detach().clone().numpy() for p in model.parameters()]

    loss1, weights1 = run_training_step()
    loss2, weights2 = run_training_step()

    assert loss1 == loss2
    for w1, w2 in zip(weights1, weights2):
        np.testing.assert_array_equal(w1, w2)
