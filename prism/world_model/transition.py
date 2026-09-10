"""World Model Latent Transition Dynamics Interfaces and Implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.latent_state import LatentDistribution
from prism.world_model.config import TransitionConfig, ModelArchitectureConfig


class BaseTransition(ABC, nn.Module):
    """Abstract base class defining the latent transition model interface.
    
    Computes prior transition dynamics: p_theta(Z_{t+1} | Z_t, A_t).
    """

    def __init__(self, model_cfg: ModelArchitectureConfig, trans_cfg: TransitionConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.trans_cfg = trans_cfg

    @abstractmethod
    def forward(self, z: Tensor, action: Tensor) -> LatentDistribution:
        """Predict the distribution over the next latent state Z_{t+1}.
        
        Args:
            z: Current latent state tensor [B, d_z] or [B, T, d_z]
            action: Executed action vector [B, 4] or [B, T, 4]
            
        Returns:
            LatentDistribution parameterized by mu(Z_{t+1}) and logvar(Z_{t+1})
        """
        pass


class MLPTransition(BaseTransition):
    """Deep residual transition network mapping (Z_t, A_t) -> p(Z_{t+1})."""

    def __init__(self, model_cfg: ModelArchitectureConfig, trans_cfg: TransitionConfig) -> None:
        super().__init__(model_cfg, trans_cfg)
        in_dim = model_cfg.latent_dim + model_cfg.action_dim

        layers = []
        curr_dim = in_dim
        for _ in range(trans_cfg.num_layers):
            layers.extend([
                nn.Linear(curr_dim, trans_cfg.hidden_dim),
                nn.LayerNorm(trans_cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(trans_cfg.dropout),
            ])
            curr_dim = trans_cfg.hidden_dim

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(trans_cfg.hidden_dim, model_cfg.latent_dim)
        self.logvar_head = nn.Linear(trans_cfg.hidden_dim, model_cfg.latent_dim)

    def forward(self, z: Tensor, action: Tensor) -> LatentDistribution:
        inp = torch.cat([z, action], dim=-1)
        feat = self.net(inp)

        delta_mu = self.mu_head(feat)
        logvar = self.logvar_head(feat)

        # Residual connection: mu_{t+1} = Z_t + Delta(Z_t, A_t)
        if self.trans_cfg.residual:
            mu = z + delta_mu
        else:
            mu = delta_mu

        return LatentDistribution(
            mean=mu,
            logvar=logvar,
            min_std=self.model_cfg.min_std,
            max_std=self.model_cfg.max_std,
        )
