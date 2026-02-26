"""Standalone MADDPG training & evaluation on SMAC v2.

This script implements MADDPG (Lowe et al., 2017) adapted for discrete actions
using Gumbel-Softmax, designed to work directly with SMACv2's
StarCraftCapabilityEnvWrapper. No AgileRL or other MARL library required.

Key design choices vs. the original MADDPG paper:
  - Discrete actions: actor outputs Gumbel-Softmax one-hot during training
    so the critic can differentiate through action selection. At execution
    time the actor just argmaxes the logits.
  - Centralized critic: sees (global_state, all_agents_one_hot_actions).
  - Parameter sharing: one actor and one critic network shared across all
    agents (reduces parameters; each agent feeds its own obs to the shared
    actor, padded with an agent-id one-hot if needed).
  - Replay buffer: flat circular numpy buffer — more memory efficient than
    the AgileRL dict-based buffer.

Modes:
    train — train MADDPG agents, save checkpoints, plot learning curves
    eval  — load a checkpoint and evaluate over N episodes
    test  — quick random-agent smoke test (verifies env works)

Usage:
    # Training (default: terran 5v5)
    python run_smacv2_maddpg.py --mode train

    # Different scenario
    python run_smacv2_maddpg.py --mode train --race protoss --n-units 10 --n-enemies 10

    # Short run for quick iteration
    python run_smacv2_maddpg.py --mode train --total-steps 200000

    # Evaluate a saved checkpoint
    python run_smacv2_maddpg.py --mode eval --load-path checkpoints/smacv2_maddpg_latest.pt

    # Quick smoke test
    python run_smacv2_maddpg.py --mode test --test-episodes 1

    # See all options
    python run_smacv2_maddpg.py --help

Requires:
    - smacv2 (pip install git+https://github.com/oxwhirl/smacv2.git)
    - StarCraft II installed (see install_sc2.sh)
    - SC2PATH env var pointing to StarCraft II directory
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Mac: default SC2PATH to Blizzard app location if not set
if not os.environ.get("SC2PATH"):
    _mac_default = "/Applications/StarCraft II"
    if os.path.isdir(_mac_default):
        os.environ["SC2PATH"] = _mac_default

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ===========================================================================
# SMAC v2 environment helpers  (identical to run_smacv2_mappo.py)
# ===========================================================================

RACE_CONFIGS = {
    "terran": {
        "unit_types": ["marine", "marauder", "medivac"],
        "exception_unit_types": ["medivac"],
        "weights": [0.45, 0.45, 0.1],
    },
    "protoss": {
        "unit_types": ["stalker", "zealot", "colossus"],
        "exception_unit_types": ["colossus"],
        "weights": [0.45, 0.45, 0.1],
    },
    "zerg": {
        "unit_types": ["zergling", "baneling", "hydralisk"],
        "exception_unit_types": ["baneling"],
        "weights": [0.45, 0.1, 0.45],
    },
}

RACE_MAP_NAMES = {
    "terran": "10gen_terran",
    "protoss": "10gen_protoss",
    "zerg": "10gen_zerg",
}


def build_distribution_config(race: str, n_units: int, n_enemies: int) -> dict:
    """Build the SMACv2 capability/distribution config."""
    race = race.lower()
    if race not in RACE_CONFIGS:
        raise ValueError(f"Unknown race '{race}'. Choose from: {list(RACE_CONFIGS)}")
    rc = RACE_CONFIGS[race]
    return {
        "n_units": n_units,
        "n_enemies": n_enemies,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": rc["unit_types"],
            "exception_unit_types": rc["exception_unit_types"],
            "weights": rc["weights"],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": n_enemies,
            "map_x": 32,
            "map_y": 32,
        },
    }


def make_env(race: str, n_units: int, n_enemies: int, render: bool = False):
    """Create a single SMACv2 environment instance.

    Args:
        render: If True, use a larger window size for better visualization.
                On Mac, SC2 always opens a visible window.
    """
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    dist_config = build_distribution_config(race, n_units, n_enemies)
    window_x, window_y = (1280, 720) if render else (640, 480)
    env = StarCraftCapabilityEnvWrapper(
        capability_config=dist_config,
        map_name=RACE_MAP_NAMES[race],
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        window_size_x=window_x,
        window_size_y=window_y,
    )
    return env


# ===========================================================================
# Neural networks
# ===========================================================================

class MADDPGActor(nn.Module):
    """Shared actor: local observation → action logits.

    During training, actions are produced via Gumbel-Softmax so that the
    critic can differentiate through them. At execution time, argmax is used.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw action logits."""
        return self.net(obs)

    def gumbel_action(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable action (Gumbel-Softmax one-hot) for training."""
        logits = self.forward(obs)
        logits = logits + (avail_actions.float() - 1.0) * 1e10  # mask unavailable
        return F.gumbel_softmax(logits, tau=tau, hard=True)

    @torch.no_grad()
    def greedy_action(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> np.ndarray:
        """Greedy argmax with availability mask."""
        logits = self.forward(obs)
        logits = logits + (avail_actions.float() - 1.0) * 1e10
        return logits.argmax(dim=-1).cpu().numpy()


class MADDPGCritic(nn.Module):
    """Centralized critic: (global_state, all_agent_actions_one_hot) → Q(s,a).

    Input size = state_dim + n_agents * n_actions
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = state_dim + n_agents * n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        actions_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state:          (batch, state_dim)
            actions_onehot: (batch, n_agents * n_actions)  — all agents concatenated

        Returns:
            Q value: (batch,)
        """
        x = torch.cat([state, actions_onehot], dim=-1)
        return self.net(x).squeeze(-1)


# ===========================================================================
# Replay buffer
# ===========================================================================

class ReplayBuffer:
    """Circular replay buffer for MADDPG on SMACv2.

    Stores per-step tuples for all agents together:
        obs          (n_agents, obs_dim)
        state        (state_dim,)
        actions      (n_agents,)          int indices
        avail        (n_agents, n_actions) available action masks
        rewards      (n_agents,)
        next_obs     (n_agents, obs_dim)
        next_state   (state_dim,)
        next_avail   (n_agents, n_actions)
        dones        scalar float
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.ptr = 0
        self.size = 0

        self.obs       = np.zeros((capacity, n_agents, obs_dim),   dtype=np.float32)
        self.state     = np.zeros((capacity, state_dim),           dtype=np.float32)
        self.actions   = np.zeros((capacity, n_agents),            dtype=np.int64)
        self.avail     = np.zeros((capacity, n_agents, n_actions), dtype=np.float32)
        self.rewards   = np.zeros((capacity, n_agents),            dtype=np.float32)
        self.next_obs  = np.zeros((capacity, n_agents, obs_dim),   dtype=np.float32)
        self.next_state= np.zeros((capacity, state_dim),           dtype=np.float32)
        self.next_avail= np.zeros((capacity, n_agents, n_actions), dtype=np.float32)
        self.dones     = np.zeros((capacity,),                     dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        avail: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        next_avail: np.ndarray,
        done: float,
    ):
        i = self.ptr
        self.obs[i]        = obs
        self.state[i]      = state
        self.actions[i]    = actions
        self.avail[i]      = avail
        self.rewards[i]    = rewards
        self.next_obs[i]   = next_obs
        self.next_state[i] = next_state
        self.next_avail[i] = next_avail
        self.dones[i]      = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.tensor(x[idx], dtype=torch.float32, device=device)
        return {
            "obs":        to_t(self.obs),         # (B, n_agents, obs_dim)
            "state":      to_t(self.state),        # (B, state_dim)
            "actions":    torch.tensor(self.actions[idx], dtype=torch.long, device=device),
            "avail":      to_t(self.avail),        # (B, n_agents, n_actions)
            "rewards":    to_t(self.rewards),      # (B, n_agents)
            "next_obs":   to_t(self.next_obs),
            "next_state": to_t(self.next_state),
            "next_avail": to_t(self.next_avail),
            "dones":      to_t(self.dones),        # (B,)
        }

    def __len__(self):
        return self.size


# ===========================================================================
# MADDPG trainer
# ===========================================================================

class MADDPGTrainer:
    """MADDPG training loop for SMAC v2 (discrete actions, parameter sharing)."""

    def __init__(
        self,
        env,
        device: str = "cpu",
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        hidden_dim: int = 128,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50_000,
        gumbel_tau: float = 1.0,
        learn_every: int = 10,
    ):
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gumbel_tau = gumbel_tau
        self.learn_every = learn_every

        env_info = env.get_env_info()
        self.n_agents   = env_info["n_agents"]
        self.n_actions  = env_info["n_actions"]
        self.obs_shape  = env_info["obs_shape"]
        self.state_shape= env_info["state_shape"]

        # --- Networks ---
        self.actor = MADDPGActor(self.obs_shape, self.n_actions, hidden_dim).to(self.device)
        self.actor_target = MADDPGActor(self.obs_shape, self.n_actions, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = MADDPGCritic(
            self.state_shape, self.n_agents, self.n_actions, hidden_dim
        ).to(self.device)
        self.critic_target = MADDPGCritic(
            self.state_shape, self.n_agents, self.n_actions, hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(
            buffer_size, self.n_agents, self.obs_shape, self.state_shape, self.n_actions
        )

        self._step = 0  # total env steps taken (for epsilon schedule)

    # ------------------------------------------------------------------
    def _epsilon(self) -> float:
        progress = min(self._step / max(self.epsilon_decay, 1), 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - progress)

    def _get_obs_and_avail(self):
        obs_list = self.env.get_obs()
        obs = np.array(obs_list, dtype=np.float32)           # (n_agents, obs_dim)
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for a in range(self.n_agents):
            avail[a] = np.array(self.env.get_avail_agent_actions(a), dtype=np.float32)
        return obs, avail

    def select_actions(self, obs: np.ndarray, avail: np.ndarray) -> np.ndarray:
        """ε-greedy action selection (returns int array of shape (n_agents,))."""
        eps = self._epsilon()
        obs_t   = torch.tensor(obs,   dtype=torch.float32, device=self.device)
        avail_t = torch.tensor(avail, dtype=torch.float32, device=self.device)
        greedy = self.actor.greedy_action(obs_t, avail_t)   # (n_agents,) ints
        actions = greedy.copy()
        for a in range(self.n_agents):
            if np.random.random() < eps:
                avail_idx = np.flatnonzero(avail[a])
                if len(avail_idx) > 0:
                    actions[a] = np.random.choice(avail_idx)
        return actions

    # ------------------------------------------------------------------
    def _actions_to_onehot(self, actions_t: torch.Tensor) -> torch.Tensor:
        """(B, n_agents) int → (B, n_agents * n_actions) float one-hot."""
        B = actions_t.shape[0]
        oh = F.one_hot(actions_t, num_classes=self.n_actions).float()  # (B, n_agents, n_actions)
        return oh.view(B, -1)

    def update(self) -> dict:
        """Sample a minibatch and update actor + critic."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, device=str(self.device))
        obs        = batch["obs"]          # (B, n_agents, obs_dim)
        state      = batch["state"]        # (B, state_dim)
        actions    = batch["actions"]      # (B, n_agents) int
        avail      = batch["avail"]        # (B, n_agents, n_actions)
        rewards    = batch["rewards"]      # (B, n_agents)
        next_obs   = batch["next_obs"]
        next_state = batch["next_state"]
        next_avail = batch["next_avail"]
        dones      = batch["dones"]        # (B,)

        B = obs.shape[0]

        # ---- Critic update ----
        with torch.no_grad():
            # Target actor: greedy one-hot per agent using target network
            next_obs_flat = next_obs.view(B * self.n_agents, self.obs_shape)
            next_avail_flat = next_avail.view(B * self.n_agents, self.n_actions)
            next_logits = self.actor_target(next_obs_flat)
            next_logits = next_logits + (next_avail_flat - 1.0) * 1e10
            next_actions_idx = next_logits.argmax(dim=-1)  # (B*n_agents,)
            next_actions_oh = F.one_hot(next_actions_idx, self.n_actions).float()
            next_actions_oh = next_actions_oh.view(B, -1)  # (B, n_agents*n_actions)

            target_q = self.critic_target(next_state, next_actions_oh)  # (B,)
            # Team reward: mean across agents
            team_reward = rewards.mean(dim=1)                            # (B,)
            target_q = team_reward + self.gamma * (1.0 - dones) * target_q

        current_actions_oh = self._actions_to_onehot(actions)           # (B, n_agents*n_actions)
        current_q = self.critic(state, current_actions_oh)              # (B,)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # ---- Actor update (policy gradient through Gumbel-Softmax) ----
        obs_flat   = obs.view(B * self.n_agents, self.obs_shape)
        avail_flat = avail.view(B * self.n_agents, self.n_actions)
        # Gumbel-softmax one-hot — differentiable
        logits = self.actor(obs_flat)
        logits_masked = logits + (avail_flat - 1.0) * 1e10
        actions_gumbel = F.gumbel_softmax(logits_masked, tau=self.gumbel_tau, hard=True)
        actions_gumbel_flat = actions_gumbel.view(B, -1)               # (B, n_agents*n_actions)

        actor_loss = -self.critic(state, actions_gumbel_flat).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ---- Soft target updates ----
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor_state_dict":         self.actor.state_dict(),
            "actor_target_state_dict":  self.actor_target.state_dict(),
            "critic_state_dict":        self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_opt_state_dict":     self.actor_opt.state_dict(),
            "critic_opt_state_dict":    self.critic_opt.state_dict(),
            "step":                     self._step,
            "env_info": {
                "n_agents":    self.n_agents,
                "n_actions":   self.n_actions,
                "obs_shape":   self.obs_shape,
                "state_shape": self.state_shape,
            },
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor_target.load_state_dict(ckpt["actor_target_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        self._step = ckpt.get("step", 0)
        print(f"Checkpoint loaded: {path}")


# ===========================================================================
# Plotting
# ===========================================================================

def rolling_mean(values: list, window: int = 20) -> list:
    out = []
    for i in range(len(values)):
        sl = values[max(0, i - window + 1): i + 1]
        out.append(float(np.mean(sl)))
    return out


def plot_training_curves(
    steps,
    rewards,
    win_rates,
    critic_losses,
    actor_losses,
    epsilons,
    save_path,
    window: int = 20,
):
    """4-panel training figure: reward, win rate, critic/actor loss, epsilon."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("MADDPG on SMAC v2 — Training Curves", fontsize=13, fontweight="bold")

    # Panel 1 — Episode reward
    ax = axes[0]
    ax.plot(steps, rewards, color="tab:orange", linewidth=0.8, alpha=0.4, label="raw")
    ax.plot(steps, rolling_mean(rewards, window), color="tab:orange", linewidth=1.8, label=f"rolling({window})")
    ax.set_ylabel("Mean Episode Reward")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2 — Win rate
    ax = axes[1]
    ax.plot(steps, win_rates, color="tab:green", linewidth=0.8, alpha=0.4)
    ax.plot(steps, rolling_mean(win_rates, window), color="tab:green", linewidth=1.8)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Critic & actor loss
    ax = axes[2]
    valid = [v for v in critic_losses if v is not None]
    if valid:
        ax.plot(steps, critic_losses, color="tab:red",  linewidth=1.5, label="critic_loss")
        ax.plot(steps, actor_losses,  color="tab:blue", linewidth=1.5, label="actor_loss")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Panel 4 — Epsilon
    ax = axes[3]
    ax.plot(steps, epsilons, color="tab:purple", linewidth=1.5)
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Env Steps")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved: {save_path}")


# ===========================================================================
# Run modes
# ===========================================================================

def run_test(args):
    """Quick smoke test with random agents."""
    print(f"=== Smoke Test: {args.race} {args.n_units}v{args.n_enemies} ===")
    env = make_env(args.race, args.n_units, args.n_enemies, render=getattr(args, 'render', False))
    env_info = env.get_env_info()
    print(f"  n_agents:    {env_info['n_agents']}")
    print(f"  n_actions:   {env_info['n_actions']}")
    print(f"  obs_shape:   {env_info['obs_shape']}")
    print(f"  state_shape: {env_info['state_shape']}")

    for ep in range(args.test_episodes):
        env.reset()
        terminated = False
        ep_reward = 0.0
        step = 0
        while not terminated:
            actions = []
            for agent_id in range(env_info["n_agents"]):
                avail = env.get_avail_agent_actions(agent_id)
                avail_idx = np.nonzero(avail)[0]
                actions.append(np.random.choice(avail_idx))
            reward, terminated, info = env.step(actions)
            ep_reward += reward
            step += 1
        won = info.get("battle_won", False)
        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, steps={step}, won={won}")

    env.close()
    print("Smoke test passed!")


def run_train(args):
    """MADDPG training loop."""
    print(f"=== MADDPG Training: {args.race} {args.n_units}v{args.n_enemies} ===")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    env = make_env(args.race, args.n_units, args.n_enemies, render=False)
    trainer = MADDPGTrainer(
        env=env,
        device=device,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        learn_every=args.learn_every,
    )

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)

    os.makedirs(args.save_dir, exist_ok=True)

    total_steps = 0
    log_steps, log_rewards, log_win_rates = [], [], []
    log_critic_losses: list = []
    log_actor_losses:  list = []
    log_epsilons:      list = []
    t_start = time.time()

    episode_reward = 0.0
    episode_count = 0
    win_count = 0
    # Per-episode loss accumulators
    ep_critic_losses: list = []
    ep_actor_losses:  list = []
    last_critic_loss = 0.0
    last_actor_loss  = 0.0

    env.reset()
    obs, avail = trainer._get_obs_and_avail()

    pbar = tqdm(total=args.total_steps, desc="MADDPG")

    while total_steps < args.total_steps:
        # --- Act ---
        actions = trainer.select_actions(obs, avail)

        state = env.get_state()
        reward, terminated, info = env.step(actions.tolist())

        next_obs, next_avail = trainer._get_obs_and_avail()
        next_state = env.get_state()

        # Broadcast scalar team reward to all agents
        rewards = np.full(trainer.n_agents, reward, dtype=np.float32)

        trainer.buffer.add(
            obs, state, actions, avail,
            rewards,
            next_obs, next_state, next_avail,
            float(terminated),
        )

        episode_reward += reward
        total_steps += 1
        trainer._step += 1
        pbar.update(1)

        if terminated:
            episode_count += 1
            won = info.get("battle_won", False)
            if won:
                win_count += 1

            log_steps.append(total_steps)
            log_rewards.append(episode_reward)
            log_win_rates.append(1.0 if won else 0.0)
            # Record mean losses over episode (None if no updates yet)
            log_critic_losses.append(
                float(np.mean(ep_critic_losses)) if ep_critic_losses else last_critic_loss
            )
            log_actor_losses.append(
                float(np.mean(ep_actor_losses)) if ep_actor_losses else last_actor_loss
            )
            log_epsilons.append(trainer._epsilon())

            ep_critic_losses.clear()
            ep_actor_losses.clear()
            episode_reward = 0.0
            env.reset()
            obs, avail = trainer._get_obs_and_avail()
        else:
            obs, avail = next_obs, next_avail

        # --- Learn ---
        if (
            total_steps >= args.learn_start
            and total_steps % args.learn_every == 0
            and len(trainer.buffer) >= args.batch_size
        ):
            stats = trainer.update()
            if stats:
                last_critic_loss = stats["critic_loss"]
                last_actor_loss  = stats["actor_loss"]
                ep_critic_losses.append(last_critic_loss)
                ep_actor_losses.append(last_actor_loss)

        # --- Logging ---
        if total_steps % args.log_interval == 0 and log_rewards:
            window = 20
            mean_rew = float(np.mean(log_rewards[-window:]))
            mean_wr  = float(np.mean(log_win_rates[-window:]))
            elapsed  = time.time() - t_start
            pbar.set_postfix({
                "rew": f"{mean_rew:.2f}",
                "wr":  f"{mean_wr:.2%}",
                "eps": f"{trainer._epsilon():.3f}",
            })
            tqdm.write(
                f"Step {total_steps:>8d} | "
                f"ep={episode_count} | "
                f"reward={mean_rew:>7.2f} | "
                f"win={mean_wr:.2%} | "
                f"eps={trainer._epsilon():.3f} | "
                f"c_loss={last_critic_loss:.4f} | "
                f"a_loss={last_actor_loss:.4f} | "
                f"time={elapsed:.0f}s"
            )

        # --- Checkpoint ---
        if total_steps % args.save_interval == 0:
            ckpt = os.path.join(args.save_dir, f"smacv2_maddpg_step_{total_steps}.pt")
            trainer.save(ckpt)

    pbar.close()
    elapsed = time.time() - t_start

    # Final checkpoint
    final_path = os.path.join(args.save_dir, "smacv2_maddpg_latest.pt")
    trainer.save(final_path)

    # Plot
    if log_steps:
        # Smooth for plot
        plot_path = os.path.join(args.save_dir, "smacv2_maddpg_training.png")
        plot_training_curves(
            log_steps,
            log_rewards,
            log_win_rates,
            log_critic_losses,
            log_actor_losses,
            log_epsilons,
            plot_path,
        )

    # Compute rolling-20 win rate history (per episode)
    roll20_wr = [
        float(np.mean(log_win_rates[max(0, i - 19): i + 1]))
        for i in range(len(log_win_rates))
    ]
    peak_roll20 = float(max(roll20_wr)) if roll20_wr else 0.0

    # Wall-clock time at each logged episode
    log_wall_clock = [elapsed * s / max(total_steps, 1) for s in log_steps]

    # Save results JSON
    results = {
        "algorithm": "MADDPG",
        "total_steps": total_steps,
        "wall_clock_seconds": elapsed,
        "race": args.race,
        "scenario": f"{args.n_units}v{args.n_enemies}",
        "total_episodes": episode_count,
        "rollout_steps": "off-policy",
        "final_mean_reward": float(np.mean(log_rewards[-20:])) if log_rewards else 0.0,
        "final_win_rate":    float(np.mean(log_win_rates[-20:])) if log_win_rates else 0.0,
        "peak_rolling20_win_rate": peak_roll20,
        "win_rate_history": log_win_rates,
        "steps_history":    log_steps,
        "wall_clock_history": log_wall_clock,
        "rewards_history":  log_rewards,
    }
    results_path = os.path.join(args.save_dir, "smacv2_maddpg_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    env.close()
    print(f"\nTraining complete! {total_steps} steps in {elapsed:.1f}s")
    print(f"Final win rate (last 20 eps): {results['final_win_rate']:.2%}")


def run_eval(args):
    """Evaluate a saved checkpoint."""
    if not args.load_path:
        print("Error: --load-path required for eval mode")
        return

    print(f"=== MADDPG Evaluation: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"Loading: {args.load_path}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    render = getattr(args, 'render', False)
    if render:
        print("Render mode ON — StarCraft II window will open.")
    env = make_env(args.race, args.n_units, args.n_enemies, render=render)
    trainer = MADDPGTrainer(env=env, device=device, hidden_dim=args.hidden_dim)
    trainer.load(args.load_path)
    trainer.actor.eval()

    total_reward = 0.0
    wins = 0

    env_info = env.get_env_info()
    for ep in range(args.eval_episodes):
        env.reset()
        terminated = False
        ep_reward = 0.0

        while not terminated:
            obs, avail = trainer._get_obs_and_avail()
            # Force epsilon=0 (greedy)
            obs_t   = torch.tensor(obs,   dtype=torch.float32, device=trainer.device)
            avail_t = torch.tensor(avail, dtype=torch.float32, device=trainer.device)
            actions = trainer.actor.greedy_action(obs_t, avail_t)
            reward, terminated, info = env.step(actions.tolist())
            ep_reward += reward

        won = info.get("battle_won", False)
        total_reward += ep_reward
        if won:
            wins += 1
        print(f"  Episode {ep+1}/{args.eval_episodes}: reward={ep_reward:.2f}, won={won}")

    env.close()
    print(f"\nEvaluation Results ({args.eval_episodes} episodes):")
    print(f"  Mean reward: {total_reward / args.eval_episodes:.2f}")
    print(f"  Win rate:    {wins / args.eval_episodes:.2%}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MADDPG Training & Evaluation on SMAC v2 (discrete actions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", choices=["train", "eval", "test"], default="train",
                        help="Run mode")

    # Environment
    parser.add_argument("--race", type=str, default="terran",
                        choices=["terran", "protoss", "zerg"])
    parser.add_argument("--n-units",   type=int, default=5,  help="Number of allied units")
    parser.add_argument("--n-enemies", type=int, default=5,  help="Number of enemy units")

    # Training
    parser.add_argument("--total-steps",    type=int,   default=2_000_000)
    parser.add_argument("--buffer-size",    type=int,   default=100_000)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--lr-actor",       type=float, default=1e-3)
    parser.add_argument("--lr-critic",      type=float, default=1e-3)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--tau",            type=float, default=0.01,
                        help="Soft target update coefficient")
    parser.add_argument("--hidden-dim",     type=int,   default=128)
    parser.add_argument("--epsilon-start",  type=float, default=1.0)
    parser.add_argument("--epsilon-end",    type=float, default=0.05)
    parser.add_argument("--epsilon-decay",  type=int,   default=50_000,
                        help="Steps over which epsilon is annealed")
    parser.add_argument("--learn-start",    type=int,   default=5_000,
                        help="Steps before first gradient update")
    parser.add_argument("--learn-every",    type=int,   default=10,
                        help="Update networks every N env steps")

    # Logging & checkpointing
    parser.add_argument("--log-interval",  type=int, default=1_000,
                        help="Log every N env steps")
    parser.add_argument("--save-interval", type=int, default=50_000,
                        help="Checkpoint every N env steps")
    parser.add_argument("--save-dir",      type=str, default="checkpoints")

    # Eval / test
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--load-path",     type=str, default="")

    # Misc
    parser.add_argument("--cpu",  action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "test":
        run_test(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
