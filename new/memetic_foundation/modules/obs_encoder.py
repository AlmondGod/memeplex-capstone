"""
obs_encoder.py — Observation encoder.

Maps raw local observation o_i^t to a latent embedding u_i^t.
Two-layer MLP with ReLU activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Encode agent observations into a latent embedding.

    u_i^t = f_θ(o_i^t)

    Architecture: obs_dim → hidden_dim → hidden_dim (ReLU between layers).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (..., obs_dim)

        Returns:
            u: (..., hidden_dim)
        """
        return self.net(obs)
