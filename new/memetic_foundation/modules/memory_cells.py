"""
memory_cells.py — Persistent per-agent memory bank.

Each agent maintains K memory cells of dimension mem_dim.
Memory contents are runtime STATE (register_buffer), not trainable weights.
A learned initial template (nn.Parameter) is copied into state at creation.
Memory persists across episodes by default, detached between rollouts.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """Persistent per-agent memory bank.

    Parameters (learned via gradient descent):
        mem_init: (K, mem_dim) — learned initial template for memory cells

    State (runtime, not optimized directly):
        memory_state: (n_agents, K, mem_dim) — actual evolving memory contents

    The memory_state is registered as a buffer so it moves with .to(device)
    and is included in state_dict, but is NOT a trainable parameter.
    """

    def __init__(
        self,
        n_agents: int,
        n_cells: int = 8,
        mem_dim: int = 128,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_cells = n_cells
        self.mem_dim = mem_dim

        # Learned initial template — provides structured starting memory
        self.mem_init = nn.Parameter(torch.empty(n_cells, mem_dim))
        nn.init.orthogonal_(self.mem_init)

        # Runtime memory state — actual evolving per-agent memory
        # Initialized by copying mem_init into each agent slot
        self.register_buffer(
            "memory_state",
            self.mem_init.data.unsqueeze(0).expand(n_agents, -1, -1).clone(),
        )

    def reset_state(self) -> None:
        """Re-copy from learned mem_init into runtime state.

        Use at: eval starts, new environments, ablation-controlled resets.
        """
        with torch.no_grad():
            self.memory_state.copy_(
                self.mem_init.data.unsqueeze(0).expand(self.n_agents, -1, -1)
            )

    def detach_state(self) -> None:
        """Detach memory state from computation graph.

        Call at rollout boundaries to truncate gradient flow
        while preserving memory values.
        """
        self.memory_state.detach_()

    def get_state(self) -> torch.Tensor:
        """Return a detached clone of current memory state."""
        return self.memory_state.detach().clone()

    def set_state(self, state: torch.Tensor) -> None:
        """Restore memory state from a saved tensor."""
        with torch.no_grad():
            self.memory_state.copy_(state)

    def update(self, new_state: torch.Tensor) -> None:
        """Replace memory state with new values (from write step).

        Uses in-place copy to preserve the registered buffer.

        Args:
            new_state: (n_agents, K, mem_dim) — output of MemoryWriter
        """
        self.memory_state.data.copy_(new_state.data)

    def forward(self) -> torch.Tensor:
        """Return current memory state: (n_agents, K, mem_dim)."""
        return self.memory_state

    def extra_repr(self) -> str:
        return (
            f"n_agents={self.n_agents}, n_cells={self.n_cells}, "
            f"mem_dim={self.mem_dim}"
        )
