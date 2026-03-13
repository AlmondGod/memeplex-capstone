"""
comm_module.py — TarMAC-style targeted multi-agent communication.

Each agent generates a message (key, value) from its current state [u; z],
and each receiver attends over all senders' messages to aggregate
incoming information.

Sender:
    k_i = W_k · [u_i ; z_i]
    v_i = W_v · [u_i ; z_i]

Receiver:
    q_j = W_q^comm · [u_j ; z_j]
    a_{j←i} = softmax_i( q_j^T · k_i / √d )
    m̄_j = Σ_i a_{j←i} · v_i
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetedComm(nn.Module):
    """TarMAC-style targeted communication between agents.

    Each agent sends a key-value message derived from [obs_encoding ; memory_summary].
    Each receiver computes attention over all senders (self-masked) and
    produces an aggregated incoming message.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
    ):
        super().__init__()
        input_dim = hidden_dim + mem_dim
        self.comm_dim = comm_dim
        self.scale = math.sqrt(comm_dim)

        # Sender: message key and value
        self.key_head = nn.Linear(input_dim, comm_dim)
        self.value_head = nn.Linear(input_dim, comm_dim)

        # Receiver: communication query
        self.query_head = nn.Linear(input_dim, comm_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.key_head, self.value_head, self.query_head]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (N, hidden_dim) — observation encodings
            z: (N, mem_dim) — memory summaries

        Returns:
            m_bar: (N, comm_dim) — aggregated incoming messages per agent
            comm_attn: (N, N) — communication attention weights
        """
        N = u.shape[0]
        s = torch.cat([u, z], dim=-1)  # (N, hidden_dim + mem_dim)

        # Sender side
        keys = self.key_head(s)      # (N, comm_dim)
        values = self.value_head(s)  # (N, comm_dim)

        # Receiver side
        queries = self.query_head(s)  # (N, comm_dim)

        # Attention: (N, N) — each row is receiver, each col is sender
        scores = torch.mm(queries, keys.t()) / self.scale  # (N, N)

        # Self-masking: agent cannot attend to its own message
        mask = torch.eye(N, device=u.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))

        comm_attn = F.softmax(scores, dim=-1)  # (N, N)
        comm_attn = torch.nan_to_num(comm_attn, nan=0.0)  # handle N=1

        # Aggregate incoming messages
        m_bar = torch.mm(comm_attn, values)  # (N, comm_dim)

        return m_bar, comm_attn
