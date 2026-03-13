"""
rollout_buffer.py — GAE rollout storage for MAPPO training.

Stores per-timestep data for all agents across rollout steps.
Computes Generalized Advantage Estimation (GAE) returns.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class RolloutBuffer:
    """Stores rollout data for all agents across steps."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.avail_actions: List[np.ndarray] = []
        self.values: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        avail_actions: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.avail_actions.append(avail_actions)
        self.values.append(values)

    def compute_returns(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages.

        Args:
            last_values: (n_agents,) — bootstrap values for the final state
            gamma: discount factor
            gae_lambda: GAE lambda

        Returns:
            returns: (T, n_agents) — discounted returns
            advantages: (T, n_agents) — GAE advantages
        """
        T = len(self.rewards)
        n_agents = self.rewards[0].shape[0]
        advantages = np.zeros((T, n_agents), dtype=np.float32)
        last_gae = np.zeros(n_agents, dtype=np.float32)

        values_arr = np.array(self.values)
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones)

        for t in reversed(range(T)):
            next_values = last_values if t == T - 1 else values_arr[t + 1]
            delta = rewards_arr[t] + gamma * next_values * (1.0 - dones_arr[t]) - values_arr[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - dones_arr[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr
        return returns, advantages

    def clear(self) -> None:
        for lst in [
            self.obs, self.states, self.actions, self.log_probs,
            self.rewards, self.dones, self.avail_actions, self.values,
        ]:
            lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)
