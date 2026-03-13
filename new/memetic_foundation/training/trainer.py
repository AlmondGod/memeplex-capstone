"""
trainer.py — MAPPO trainer for Memetic Foundation.

Adapts standard MAPPO training loop with:
  - Per-agent persistent memory state management
  - Memory detachment between rollouts
  - Memory state in checkpoints
  - Diagnostics: memory cell norms, write gate values
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from ..models.agent_network import MemeticFoundationAC
from .rollout_buffer import RolloutBuffer
from .env_utils import make_env


class MemeticFoundationTrainer:
    """MAPPO-based trainer with persistent memory management."""

    def __init__(
        self,
        env,
        device: str = "cpu",
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        update_epochs: int = 5,
        num_mini_batches: int = 1,
        hidden_dim: int = 128,
        mem_dim: int = 128,
        comm_dim: int = 128,
        n_mem_cells: int = 8,
        use_memory: bool = True,
        use_comm: bool = True,
    ):
        self.env = env
        self.device = torch.device(device)

        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_shape = env_info["obs_shape"]
        self.state_shape = env_info["state_shape"]

        self.policy = MemeticFoundationAC(
            obs_dim=self.obs_shape,
            state_dim=self.state_shape,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            hidden_dim=hidden_dim,
            mem_dim=mem_dim,
            comm_dim=comm_dim,
            n_mem_cells=n_mem_cells,
            use_memory=use_memory,
            use_comm=use_comm,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_mini_batches = num_mini_batches

        # Track whether env has been reset at least once
        self._started = False
        self._episode_reward = 0.0

        # Print variant info
        variant = self.policy.get_variant_name()
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Variant:    {variant}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Enc dim:    {self.policy.enc_dim}")
        print(f"  Device:     {self.device}")

    def collect_rollout(self, rollout_steps: int) -> Tuple[RolloutBuffer, np.ndarray, dict]:
        """Collect a rollout of transitions.

        Memory state persists across episodes within the rollout.
        Memory is detached at the start of each rollout to truncate gradients.

        Returns: (buffer, last_values, stats)
        """
        buffer = RolloutBuffer()
        episode_rewards = []
        win_count = 0
        episode_count = 0

        if not self._started:
            self.env.reset()
            self._started = True

        # Detach memory at rollout boundary
        self.policy.detach_memory()
        self.policy.eval()

        for step in range(rollout_steps):
            obs_list = self.env.get_obs()
            state = self.env.get_state()

            obs_arr = np.array(obs_list, dtype=np.float32)
            state_arr = np.array(state, dtype=np.float32)

            avail_arr = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            for agent_id in range(self.n_agents):
                avail = self.env.get_avail_agent_actions(agent_id)
                avail_arr[agent_id] = np.array(avail, dtype=np.float32)

            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
                avail_t = torch.tensor(avail_arr, dtype=torch.float32, device=self.device)
                state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device)

                # Full forward step (includes comm + memory write)
                step_out = self.policy.forward_step(obs_t, avail_t)

                # Centralized value
                state_repeated = state_t.unsqueeze(0).expand(self.n_agents, -1)
                values_t = self.policy.get_value(state_repeated)

            actions = step_out["actions"].cpu().numpy()
            log_probs = step_out["log_probs"].cpu().numpy()
            values = values_t.cpu().numpy()

            reward, terminated, info = self.env.step(actions.tolist())

            rewards = np.full(self.n_agents, reward, dtype=np.float32)
            dones = np.full(self.n_agents, float(terminated), dtype=np.float32)

            buffer.add(
                obs_arr,
                np.tile(state_arr, (self.n_agents, 1)),
                actions, log_probs, rewards, dones, avail_arr, values,
            )

            self._episode_reward += reward
            if terminated:
                episode_rewards.append(self._episode_reward)
                if info.get("battle_won", False):
                    win_count += 1
                episode_count += 1
                self._episode_reward = 0.0
                self.env.reset()
                # NOTE: memory is NOT reset between episodes (persistence)

        # Bootstrap value for GAE
        with torch.no_grad():
            state = self.env.get_state()
            state_t = torch.tensor(
                np.array(state, dtype=np.float32), device=self.device
            )
            state_repeated = state_t.unsqueeze(0).expand(self.n_agents, -1)
            last_values = self.policy.get_value(state_repeated).cpu().numpy()

        stats = {
            "episode_rewards": episode_rewards,
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "win_rate": win_count / max(episode_count, 1),
            "episodes": episode_count,
        }

        # Memory diagnostics
        mem_state = self.policy.get_memory_state()
        if mem_state is not None:
            stats["mem_norm"] = float(mem_state.norm().item())
            stats["mem_cell_norms"] = mem_state.norm(dim=-1).mean().item()

        return buffer, last_values, stats

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray) -> dict:
        """Run PPO update on collected rollout."""
        returns, advantages = buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda
        )

        flat_obs = np.concatenate(buffer.obs, axis=0)
        flat_states = np.concatenate(buffer.states, axis=0)
        flat_actions = np.concatenate(buffer.actions, axis=0)
        flat_log_probs = np.concatenate(buffer.log_probs, axis=0)
        flat_avail = np.concatenate(buffer.avail_actions, axis=0)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        flat_advantages = (
            (flat_advantages - flat_advantages.mean())
            / (flat_advantages.std() + 1e-8)
        )

        obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
        states_t = torch.tensor(flat_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(flat_actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.tensor(flat_log_probs, dtype=torch.float32, device=self.device)
        avail_t = torch.tensor(flat_avail, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(flat_returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)

        batch_size = obs_t.shape[0]
        mini_batch_size = max(batch_size // self.num_mini_batches, 1)

        total_pg, total_vf, total_ent, total_aux, n_updates = 0.0, 0.0, 0.0, 0.0, 0

        self.policy.train()
        for _ in range(self.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                idx = indices[start:end]

                new_lp, entropy, values, aux_loss = self.policy.evaluate_actions(
                    obs_t[idx], states_t[idx], actions_t[idx], avail_t[idx]
                )

                ratio = torch.exp(new_lp - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                ) * adv_t[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * ((values - returns_t[idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss + aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                total_aux += aux_loss.item()
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "aux_loss": total_aux / max(n_updates, 1),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "env_info": {
                "n_agents": self.n_agents,
                "n_actions": self.n_actions,
                "obs_shape": self.obs_shape,
                "state_shape": self.state_shape,
            },
            "variant": self.policy.get_variant_name(),
        }
        # Save memory state separately (it's a buffer, included in state_dict,
        # but we also save it explicitly for clarity)
        mem_state = self.policy.get_memory_state()
        if mem_state is not None:
            ckpt["memory_state"] = mem_state
        torch.save(ckpt, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Restore memory state if available
        if "memory_state" in ckpt:
            self.policy.set_memory_state(ckpt["memory_state"])
        print(f"Checkpoint loaded: {path}")


# ===========================================================================
# Plotting
# ===========================================================================

def rolling_mean(values: list, window: int = 10) -> list:
    out = []
    for i in range(len(values)):
        sl = values[max(0, i - window + 1): i + 1]
        out.append(float(np.mean(sl)))
    return out


def plot_training_curves(
    steps, rewards, win_rates, pg_losses, vf_losses, entropies,
    save_path, variant_name: str = "", window: int = 10,
):
    """4-panel training figure."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    title = f"Memetic Foundation ({variant_name}) — Training Curves"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(steps, rewards, color="tab:blue", linewidth=0.8, alpha=0.4, label="raw")
    ax.plot(steps, rolling_mean(rewards, window), color="tab:blue", linewidth=1.8,
            label=f"rolling({window})")
    ax.set_ylabel("Mean Episode Reward")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, win_rates, color="tab:green", linewidth=0.8, alpha=0.4)
    ax.plot(steps, rolling_mean(win_rates, window), color="tab:green", linewidth=1.8)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if pg_losses:
        ax.plot(steps, pg_losses, color="tab:red", linewidth=1.5, label="pg_loss")
        ax.plot(steps, vf_losses, color="tab:orange", linewidth=1.5, label="vf_loss")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if entropies:
        ax.plot(steps, entropies, color="tab:purple", linewidth=1.5)
    ax.set_ylabel("Entropy")
    ax.set_xlabel("Env Steps")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved: {save_path}")
