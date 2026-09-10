"""World Model Observation Decoder Interfaces and Implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.latent_state import ObservationDistribution
from prism.world_model.config import DecoderConfig, ModelArchitectureConfig


class BaseDecoder(ABC, nn.Module):
    """Abstract base class defining the observation reconstruction decoder interface.
    
    Computes predictive likelihood: p_theta(O_t | Z_t).
    """

    def __init__(self, model_cfg: ModelArchitectureConfig, dec_cfg: DecoderConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.dec_cfg = dec_cfg

    @abstractmethod
    def forward(self, z: Tensor) -> ObservationDistribution:
        """Decode latent state vector Z into reconstructed observation distributions.
        
        Args:
            z: Latent state tensor [B, d_z] or [B, T, d_z]
            
        Returns:
            ObservationDistribution parameterized by mu(O_t) [B, 8] and logvar(O_t) [B, 8]
        """
        pass


class MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder mapping Z_t -> p(O_t)."""

    def __init__(self, model_cfg: ModelArchitectureConfig, dec_cfg: DecoderConfig) -> None:
        super().__init__(model_cfg, dec_cfg)

        layers = []
        curr_dim = model_cfg.latent_dim
        for _ in range(dec_cfg.num_layers):
            layers.extend([
                nn.Linear(curr_dim, dec_cfg.hidden_dim),
                nn.LayerNorm(dec_cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dec_cfg.dropout),
            ])
            curr_dim = dec_cfg.hidden_dim

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dec_cfg.hidden_dim, model_cfg.obs_dim)

        if dec_cfg.learn_variance:
            self.logvar_head = nn.Linear(dec_cfg.hidden_dim, model_cfg.obs_dim)
        else:
            self.logvar_head = None

    def forward(self, z: Tensor) -> ObservationDistribution:
        feat = self.net(z)
        mu = self.mu_head(feat)

        if self.logvar_head is not None:
            logvar = self.logvar_head(feat)
        else:
            # Fixed unit / small baseline variance
            logvar = torch.zeros_like(mu)

        return ObservationDistribution(
            mean=mu,
            logvar=logvar,
            min_std=self.model_cfg.min_std,
            max_std=self.model_cfg.max_std,
        )
