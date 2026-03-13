"""
memory_read.py — Soft attention read from persistent memory cells.

The agent reads from its own memory cells using the current observation
encoding as a query. Produces a memory summary z_i^t used for action
selection and message generation.

    q_i = W_q^read · u_i
    α_ik = softmax_k( q_i^T · c(m_ik) / √d )
    z_i = Σ_k α_ik · m_ik

Supports two modes:
  - Standard: u (N, d), memory (N, K, d) → per-agent read
  - Batched:  u (B, d) with precomputed keys (N, K, d) → broadcast read
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryReader(nn.Module):
    """Soft attention read over per-agent memory cells.

    Uses a learned query projection from the observation encoding and
    a learned key projection from each memory cell. Produces a weighted
    sum of memory cell contents.
    """

    def __init__(self, hidden_dim: int = 128, mem_dim: int = 128):
        super().__init__()
        self.scale = math.sqrt(mem_dim)

        # Query projection: observation encoding → query space
        self.query_proj = nn.Linear(hidden_dim, mem_dim)

        # Key projection: memory cell → key space (learned linear)
        self.key_proj = nn.Linear(mem_dim, mem_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.query_proj, self.key_proj]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        u: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (B, hidden_dim) — observation encodings
            memory: (B, K, mem_dim) — memory cell contents (same leading dim as u)

        Returns:
            z: (B, mem_dim) — memory-conditioned summary
            attn_weights: (B, K) — attention weights over cells
        """
        q = self.query_proj(u)          # (B, mem_dim)
        keys = self.key_proj(memory)    # (B, K, mem_dim)

        # Attention scores via einsum (cleaner than bmm + squeeze)
        scores = torch.einsum("bd,bkd->bk", q, keys) / self.scale  # (B, K)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: (B, K, 1) * (B, K, mem_dim) → sum → (B, mem_dim)
        z = torch.einsum("bk,bkd->bd", attn_weights, memory)

        return z, attn_weights

    def forward_batched(
        self,
        u_batch: torch.Tensor,
        memory: torch.Tensor,
        n_agents: int,
    ) -> torch.Tensor:
        """Efficient batched read for PPO evaluation.

        Avoids tiling memory from (N, K, d) to (B*N, K, d) by:
        1. Precomputing keys once from memory: (N, K, d)
        2. Computing queries for all batch items: (B*N, d)
        3. Reshaping to (B, N, ...) groups and using broadcast attention

        Args:
            u_batch: (B_total, hidden_dim) — all observation encodings
            memory:  (N, K, mem_dim) — memory state (NOT expanded)
            n_agents: N — number of agents

        Returns:
            z: (B_total, mem_dim) — memory summaries for all batch items
        """
        B_total = u_batch.shape[0]
        N = n_agents
        B = B_total // N  # number of groups

        # Precompute keys once: (N, K, mem_dim)
        keys = self.key_proj(memory)

        # Queries for all: (B_total, mem_dim)
        queries = self.query_proj(u_batch)

        # Reshape to groups: (B, N, mem_dim)
        queries_grouped = queries.view(B, N, -1)

        # Attention: each group uses the same N keys
        # scores: (B, N, K) via einsum — queries_grouped (B,N,d) @ keys (N,K,d)
        scores = torch.einsum("bnd,nkd->bnk", queries_grouped, keys) / self.scale
        attn = F.softmax(scores, dim=-1)  # (B, N, K)

        # Weighted sum: attn (B,N,K) @ memory (N,K,d) → (B,N,d)
        z_grouped = torch.einsum("bnk,nkd->bnd", attn, memory)

        # Flatten back: (B_total, mem_dim)
        return z_grouped.reshape(B_total, -1)
