"""
memory_write.py — Communication-driven differentiable memory write.

Incoming communication is written into the agent's persistent memory cells
via a gated soft-addressing mechanism:

1. Write scores (which cells to update):
    β_jk = softmax_k( g([u; z; m̄], m_jk) )
    where g is: (W_write_query · [u;z;m̄])^T · (W_write_key · m_jk)

2. Write gate (how much to write):
    w = σ(W_w · [u; z; m̄])

3. Candidate content (what to write):
    m̃ = ϕ([u; m̄])  — small MLP

4. Update:
    m_jk^{t+1} = (1 - w·β_jk) · m_jk^t + w·β_jk · m̃

Optimization notes:
  - write_query and write_gate are fused into a single linear (same input)
  - candidate_input [u; m̄] is computed without re-concatenating z
  - gated update uses additive form to avoid one multiplication
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryWriter(nn.Module):
    """Differentiable memory write driven by incoming communication.

    Given the agent's observation encoding, memory summary, and aggregated
    received messages, computes soft write addresses, a write gate, and
    candidate content, then updates each memory cell with a gated interpolation.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
    ):
        super().__init__()
        self.mem_dim = mem_dim

        # Input dimension: [u; z; m̄]
        context_dim = hidden_dim + mem_dim + comm_dim

        # FUSED: write_query (→ mem_dim) + write_gate (→ 1) in one linear
        # Output: first mem_dim dims are query, last dim is gate logit
        self.write_query_gate = nn.Linear(context_dim, mem_dim + 1)
        self.write_key = nn.Linear(mem_dim, mem_dim, bias=False)
        self.write_scale = math.sqrt(mem_dim)

        # Candidate content: MLP from [u; m̄]
        candidate_in = hidden_dim + comm_dim
        self.candidate_net = nn.Sequential(
            nn.Linear(candidate_in, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, mem_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize write gate bias slightly negative (conservative writes)
        # Gate is the last dimension of the fused layer
        with torch.no_grad():
            self.write_query_gate.bias[-1] = -1.0

    def forward(
        self,
        memory: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
        m_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory: (N, K, mem_dim) — current memory cell contents
            u: (N, hidden_dim) — observation encodings
            z: (N, mem_dim) — memory summaries
            m_bar: (N, comm_dim) — aggregated received messages

        Returns:
            new_memory: (N, K, mem_dim) — updated memory cells
        """
        # Context: [u; z; m̄]
        context = torch.cat([u, z, m_bar], dim=-1)  # (N, context_dim)

        # --- FUSED: write query + gate in one linear ---
        qg = self.write_query_gate(context)        # (N, mem_dim + 1)
        wq = qg[:, :self.mem_dim]                  # (N, mem_dim) — query
        w = torch.sigmoid(qg[:, -1:])              # (N, 1)       — gate

        # --- Write scores (which cells to update) ---
        # Key from each memory cell: (N, K, mem_dim) — no bias, slightly cheaper
        wk = self.write_key(memory)
        # Score via einsum instead of bmm (avoids unsqueeze/squeeze)
        beta = torch.einsum("nd,nkd->nk", wq, wk)  # (N, K)
        beta = F.softmax(beta / self.write_scale, dim=-1)

        # --- Candidate content (what to write) ---
        candidate_input = torch.cat([u, m_bar], dim=-1)
        m_tilde = self.candidate_net(candidate_input)  # (N, mem_dim)

        # --- Gated update ---
        # w*β: (N, K, 1)
        ww = (w * beta).unsqueeze(-1)
        # m' = m + ww * (m̃ - m)  [additive form, one fewer mul than original]
        new_memory = memory + ww * (m_tilde.unsqueeze(1) - memory)

        return new_memory
