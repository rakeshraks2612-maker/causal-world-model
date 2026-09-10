"""World Model Encoder Interfaces and Baseline Implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from prism.world_model.inputs import ModelInputs
from prism.world_model.latent_state import LatentDistribution
from prism.world_model.config import EncoderConfig, ModelArchitectureConfig


class BaseEncoder(ABC, nn.Module):
    """Abstract base class defining the World Model sequence encoder interface.
    
    Maps history (O_{<=t}, M_{<=t}, A_{<=t}) to variational latent posterior q_phi(Z_t | O, A).
    """

    def __init__(self, model_cfg: ModelArchitectureConfig, enc_cfg: EncoderConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.enc_cfg = enc_cfg

    @abstractmethod
    def forward(
        self,
        inputs: ModelInputs,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[LatentDistribution, Optional[Tensor]]:
        """Encode input history into latent posterior distributions.
        
        Args:
            inputs: Batch of observations [B, T, 8], masks [B, T, 8], and actions [B, T, 4]
            hidden_state: Optional initial recurrent/memory state
            
        Returns:
            (LatentDistribution with mean [B, T, d_z] and logvar [B, T, d_z], final hidden_state)
        """
        pass


class GRUEncoder(BaseEncoder):
    """Recurrent sequence encoder mapping (Observation, Mask, Action) -> LatentDistribution."""

    def __init__(self, model_cfg: ModelArchitectureConfig, enc_cfg: EncoderConfig) -> None:
        super().__init__(model_cfg, enc_cfg)
        # Input features: 8 (obs) + 8 (mask) + 4 (action) = 20 channels
        in_dim = model_cfg.obs_dim + model_cfg.obs_dim + model_cfg.action_dim
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, enc_cfg.hidden_dim),
            nn.LayerNorm(enc_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(enc_cfg.dropout),
        )

        self.gru = nn.GRU(
            input_size=enc_cfg.hidden_dim,
            hidden_size=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            dropout=enc_cfg.dropout if enc_cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=enc_cfg.bidirectional,
        )

        gru_out_dim = enc_cfg.hidden_dim * (2 if enc_cfg.bidirectional else 1)

        self.mu_head = nn.Linear(gru_out_dim, model_cfg.latent_dim)
        self.logvar_head = nn.Linear(gru_out_dim, model_cfg.latent_dim)

    def forward(
        self,
        inputs: ModelInputs,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[LatentDistribution, Optional[Tensor]]:
        # Concatenate observable inputs: [B, T, 20]
        # Observations where mask=0 are safely zero-filled
        clean_obs = inputs.observations * inputs.observation_mask
        x = torch.cat([clean_obs, inputs.observation_mask, inputs.actions], dim=-1)

        feat = self.feature_proj(x)
        gru_out, h_n = self.gru(feat, hidden_state)

        mean = self.mu_head(gru_out)
        logvar = self.logvar_head(gru_out)

        dist = LatentDistribution(
            mean=mean,
            logvar=logvar,
            min_std=self.model_cfg.min_std,
            max_std=self.model_cfg.max_std,
        )
        return dist, h_n
