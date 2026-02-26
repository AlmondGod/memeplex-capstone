"""
run_smacv2_tarmac.py — TarMAC on SMAC v2
==========================================

TarMAC: Targeted Multi-Agent Communication (Das et al. 2019, arXiv:1810.11187)
Backbone: PPO with CTDE (centralised training, decentralised execution).

Architecture per step (per agent i, n_agents total):

  1. Encode observation:   h_i = encoder(obs_i)          [hidden_dim]
  2. Produce message:
       key_i   = W_k · h_i                               [comm_dim]
       value_i = W_v · h_i                               [comm_dim]
  3. Produce query:        query_i = W_q · h_i           [comm_dim]
  4. Targeted aggregation (soft attention, j ≠ i):
       α_{i←j} = softmax_j( query_i · key_j / √d )
       context_i = Σ_{j≠i} α_{i←j} · value_j           [comm_dim]
  5. Act:  logits_i = actor([h_i ‖ context_i])
  6. Value (centralised): V = critic(global_state)

Multi-round comm: repeat steps 2-4 R times before acting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from torch.distributions import Categorical
from tqdm import tqdm


# ===========================================================================
# Environment helpers  (identical to MAPPO script)
# ===========================================================================

RACE_MAP_NAMES = {
    "terran":  "10gen_terran",
    "protoss": "10gen_protoss",
    "zerg":    "10gen_zerg",
}


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


def build_distribution_config(race: str, n_units: int, n_enemies: int) -> dict:
    rc = RACE_CONFIGS[race]
    return {
        "n_units":   n_units,
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
    """Create a SMACv2 environment. render=True uses a larger window on Mac."""
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
# TarMAC Actor-Critic Network
# ===========================================================================

class TarMACActorCritic(nn.Module):
    """
    Shared-parameter TarMAC network for all agents.

    Components
    ----------
    encoder   : obs_dim  → hidden_dim
    key_head  : hidden_dim → comm_dim   (message "signature")
    value_head: hidden_dim → comm_dim   (message "content")
    query_head: hidden_dim → comm_dim   (what I'm looking for)
    actor     : hidden_dim + comm_dim → n_actions
    critic    : state_dim → 1           (centralised V)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        comm_dim: int = 16,
        comm_rounds: int = 1,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.comm_dim    = comm_dim
        self.comm_rounds = comm_rounds
        self.scale       = math.sqrt(comm_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Communication heads
        self.key_head   = nn.Linear(hidden_dim, comm_dim)
        self.value_head = nn.Linear(hidden_dim, comm_dim)
        self.query_head = nn.Linear(hidden_dim, comm_dim)

        # Actor uses hidden + aggregated context
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Centralised critic (uses global state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def _communicate(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Perform one round of targeted communication.

        Parameters
        ----------
        hidden : (n_agents, hidden_dim)  or  (batch, n_agents, hidden_dim)

        Returns
        -------
        context : same leading shape, comm_dim last dim
        """
        # Support both (N, D) and (B, N, D)
        squeeze = hidden.dim() == 2
        if squeeze:
            hidden = hidden.unsqueeze(0)   # (1, N, D)

        B, N, _ = hidden.shape

        keys    = self.key_head(hidden)    # (B, N, C)
        values  = self.value_head(hidden)  # (B, N, C)
        queries = self.query_head(hidden)  # (B, N, C)

        # Attention scores:  (B, N_query, N_key)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale

        # Mask self-attention (diagonal): set to -inf
        mask = torch.eye(N, device=hidden.device, dtype=torch.bool)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)   # (B, N, N)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard all-inf rows (N=1 edge case)

        context = torch.bmm(attn, values)  # (B, N, C)

        if squeeze:
            context = context.squeeze(0)   # (N, C)

        return context

    # ------------------------------------------------------------------
    def communicate_and_encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hidden, context) after comm_rounds rounds.
        obs : (n_agents, obs_dim)  or  (B, n_agents, obs_dim)
        """
        hidden = self.encoder(obs)
        context = torch.zeros(*hidden.shape[:-1], self.comm_dim,
                               device=obs.device, dtype=obs.dtype)
        for _ in range(self.comm_rounds):
            context = self._communicate(hidden)
        return hidden, context

    # ------------------------------------------------------------------
    def get_actor_output(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (n_agents, obs_dim)"""
        hidden, context = self.communicate_and_encode(obs)
        return self.actor(torch.cat([hidden, context], dim=-1))

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,         # (B, N, obs_dim)
        state: torch.Tensor,       # (B, state_dim)
        actions: torch.Tensor,     # (B, N)
        avail_actions: torch.Tensor,  # (B, N, n_actions)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, context = self.communicate_and_encode(obs)
        logits = self.actor(torch.cat([hidden, context], dim=-1))  # (B, N, A)
        logits = logits.masked_fill(avail_actions == 0, -1e10)
        dist   = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)   # (B, N)
        entropy   = dist.entropy()           # (B, N)
        values    = self.get_value(state)    # (B,)
        return log_probs, entropy, values


# ===========================================================================
# Rollout buffer  (same structure as MAPPO)
# ===========================================================================

class RolloutBuffer:
    def __init__(self):
        self.obs:          List[np.ndarray] = []
        self.states:       List[np.ndarray] = []
        self.actions:      List[np.ndarray] = []
        self.log_probs:    List[np.ndarray] = []
        self.rewards:      List[float]      = []
        self.dones:        List[bool]       = []
        self.avail_actions:List[np.ndarray] = []
        self.values:       List[np.ndarray] = []

    def add(self, obs, state, actions, log_probs, reward, done, avail_actions, values):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.avail_actions.append(avail_actions)
        self.values.append(values)

    def compute_returns(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        T       = len(self.rewards)
        n_agents= self.obs[0].shape[0]
        advantages = np.zeros((T, n_agents), dtype=np.float32)
        last_gae   = np.zeros(n_agents, dtype=np.float32)
        values_arr = np.array(self.values)       # (T, n_agents)
        rewards_arr= np.array(self.rewards)      # (T,)
        dones_arr  = np.array(self.dones)        # (T,)

        for t in reversed(range(T)):
            next_val   = last_values if t == T - 1 else values_arr[t + 1]
            mask       = 1.0 - float(dones_arr[t])
            delta      = rewards_arr[t] + gamma * next_val * mask - values_arr[t]
            last_gae   = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr
        return returns, advantages

    def clear(self):
        self.__init__()


# ===========================================================================
# TarMAC Trainer
# ===========================================================================

class TarMACTrainer:
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
        update_epochs: int = 4,
        num_mini_batches: int = 4,
        hidden_dim: int = 128,
        comm_dim: int = 16,
        comm_rounds: int = 1,
    ):
        self.env            = env
        self.device         = device
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_coef      = clip_coef
        self.ent_coef       = ent_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
        self.update_epochs  = update_epochs
        self.num_mini_batches = num_mini_batches

        env_info   = env.get_env_info()
        obs_dim    = env_info["obs_shape"]
        state_dim  = env_info["state_shape"]
        n_actions  = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

        self.policy = TarMACActorCritic(
            obs_dim    = obs_dim,
            state_dim  = state_dim,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            comm_dim   = comm_dim,
            comm_rounds= comm_rounds,
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _step(self, render: bool = False):
        """Collect one environment step. Returns (reward, done, info)."""
        env_info   = self.env.get_env_info()
        obs_list   = self.env.get_obs()
        state      = np.array(self.env.get_state(), dtype=np.float32)
        obs_arr    = np.array(obs_list, dtype=np.float32)   # (N, obs_dim)
        avail_arr  = np.array(
            [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)],
            dtype=np.float32)                                # (N, A)

        obs_t   = torch.tensor(obs_arr,   device=self.device)
        avail_t = torch.tensor(avail_arr, device=self.device)
        state_t = torch.tensor(state,     device=self.device)

        hidden, context  = self.policy.communicate_and_encode(obs_t)
        logits = self.policy.actor(torch.cat([hidden, context], dim=-1))
        logits = logits.masked_fill(avail_t == 0, -1e10)
        dist   = Categorical(logits=logits)
        actions= dist.sample()               # (N,)
        log_probs = dist.log_prob(actions)   # (N,)
        values = self.policy.get_value(state_t.unsqueeze(0)).squeeze(0)  # scalar -> (1,)

        reward, terminated, info = self.env.step(actions.cpu().numpy().tolist())

        self.buffer.add(
            obs         = obs_arr,
            state       = state,
            actions     = actions.cpu().numpy(),
            log_probs   = log_probs.cpu().numpy(),
            reward      = reward,
            done        = terminated,
            avail_actions = avail_arr,
            values      = values.cpu().numpy() * np.ones(self.n_agents, dtype=np.float32),
        )

        return reward, terminated, info

    # ------------------------------------------------------------------
    def collect_rollout(self, n_steps: int) -> Tuple[float, float]:
        """
        Collect up to n_steps steps (may span multiple episodes).
        Returns (ep_reward, win_rate) where win_rate is the fraction of
        completed episodes within this rollout that were won.
        """
        self.buffer.clear()
        self.env.reset()
        ep_reward   = 0.0
        win_count   = 0
        episode_count = 0
        terminated  = False

        for _ in range(n_steps):
            reward, terminated, info = self._step()
            ep_reward += reward
            if terminated:
                episode_count += 1
                if info.get("battle_won", False):
                    win_count += 1
                self.env.reset()
                terminated = False

        # Bootstrap value for last state
        with torch.no_grad():
            obs_list = self.env.get_obs()
            state    = np.array(self.env.get_state(), dtype=np.float32)
            state_t  = torch.tensor(state, device=self.device)
            last_val = self.policy.get_value(state_t.unsqueeze(0)).squeeze(0)
            last_val_np = last_val.cpu().numpy() * np.ones(self.n_agents, dtype=np.float32)

        returns, advantages = self.buffer.compute_returns(last_val_np, self.gamma, self.gae_lambda)
        win_rate = win_count / max(episode_count, 1)
        return ep_reward, win_rate

    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """One PPO update over the collected rollout."""
        T          = len(self.buffer.obs)
        N          = self.n_agents
        obs        = torch.tensor(np.array(self.buffer.obs),          device=self.device)   # (T,N,o)
        states     = torch.tensor(np.array(self.buffer.states),       device=self.device)   # (T,s)
        actions    = torch.tensor(np.array(self.buffer.actions),      device=self.device)   # (T,N)
        old_lp     = torch.tensor(np.array(self.buffer.log_probs),    device=self.device)   # (T,N)
        avail      = torch.tensor(np.array(self.buffer.avail_actions),device=self.device)   # (T,N,A)

        # Recompute returns & advantages
        with torch.no_grad():
            last_state = states[-1]
            last_val   = self.policy.get_value(last_state.unsqueeze(0)).squeeze(0)
            last_val_np = last_val.cpu().numpy() * np.ones(N, dtype=np.float32)
        returns_np, adv_np = self.buffer.compute_returns(last_val_np, self.gamma, self.gae_lambda)

        returns    = torch.tensor(returns_np, device=self.device)   # (T,N)
        advantages = torch.tensor(adv_np,     device=self.device)   # (T,N)
        adv_flat   = advantages.view(-1)      # (T*N,)
        adv_flat   = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.update_epochs):
            indices = torch.randperm(T)
            mb_size = max(1, T // self.num_mini_batches)

            for start in range(0, T, mb_size):
                mb_idx = indices[start:start + mb_size]
                mb_obs    = obs[mb_idx]        # (mb, N, o)
                mb_states = states[mb_idx]     # (mb, s)
                mb_acts   = actions[mb_idx]    # (mb, N)
                mb_old_lp = old_lp[mb_idx]     # (mb, N)
                mb_avail  = avail[mb_idx]      # (mb, N, A)
                mb_ret    = returns[mb_idx]    # (mb, N)
                mb_adv    = adv_flat[mb_idx.repeat_interleave(N) if False else
                                     torch.arange(start, min(start + mb_size, T), device=self.device)
                                     .unsqueeze(1).expand(-1, N).reshape(-1)]  # simplification below

                # Simpler: just slice the advantage array
                mb_adv_2d = advantages[mb_idx]           # (mb, N)
                mb_adv_n  = (mb_adv_2d.view(-1) - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                new_lp, entropy, values = self.policy.evaluate_actions(
                    mb_obs, mb_states, mb_acts, mb_avail
                )
                # values: (mb,) → broadcast over agents with mean return
                mb_ret_mean = mb_ret.mean(dim=-1)   # (mb,)

                ratio  = torch.exp(new_lp - mb_old_lp)   # (mb, N)
                mb_adv_2d_norm = mb_adv_2d.view(-1)
                mb_adv_2d_norm = (mb_adv_2d_norm - mb_adv_2d_norm.mean()) / (mb_adv_2d_norm.std() + 1e-8)
                mb_adv_2d_norm = mb_adv_2d_norm.view(ratio.shape)

                pg_loss1 = -mb_adv_2d_norm * ratio
                pg_loss2 = -mb_adv_2d_norm * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = F.mse_loss(values, mb_ret_mean)
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(-ent_loss.item())

        return {
            "pg_loss":  float(np.mean(pg_losses)),
            "vf_loss":  float(np.mean(vf_losses)),
            "entropy":  float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Checkpoint loaded: {path}")


# ===========================================================================
# Plotting
# ===========================================================================

def save_plot(rewards, win_rates, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(rewards);      ax1.set_title("Episode Reward"); ax1.set_xlabel("Update")
    ax2.plot(win_rates);    ax2.set_title("Win Rate");       ax2.set_xlabel("Update")
    fig.tight_layout()
    path = os.path.join(save_dir, "smacv2_tarmac_training.png")
    fig.savefig(path);  plt.close(fig)
    print(f"Training plot saved: {path}")


# ===========================================================================
# Run modes
# ===========================================================================

def run_test(args):
    """Quick smoke test with random agents + comm."""
    print(f"=== TarMAC Smoke Test: {args.race} {args.n_units}v{args.n_enemies} ===")
    env      = make_env(args.race, args.n_units, args.n_enemies, render=getattr(args, "render", False))
    env_info = env.get_env_info()
    print(f"  n_agents:    {env_info['n_agents']}")
    print(f"  n_actions:   {env_info['n_actions']}")
    print(f"  obs_shape:   {env_info['obs_shape']}")
    print(f"  state_shape: {env_info['state_shape']}")

    # Quick forward pass to verify comm shapes
    trainer = TarMACTrainer(
        env=env,
        comm_dim   =args.comm_dim,
        comm_rounds=args.comm_rounds,
        hidden_dim =args.hidden_dim,
    )
    comm_params = sum(p.numel() for p in trainer.policy.parameters() if p.requires_grad)
    print(f"  comm_dim:    {args.comm_dim}  comm_rounds: {args.comm_rounds}")
    print(f"  total params:{comm_params:,}")

    for ep in range(args.test_episodes):
        env.reset()
        terminated = False
        ep_reward  = 0.0
        step       = 0
        while not terminated:
            obs_list = env.get_obs()
            obs_t    = torch.tensor(np.array(obs_list, dtype=np.float32))
            avail    = np.array([env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])],
                                dtype=np.float32)
            with torch.no_grad():
                hidden, context = trainer.policy.communicate_and_encode(obs_t)
                logits = trainer.policy.actor(torch.cat([hidden, context], dim=-1))
                logits = logits.masked_fill(torch.tensor(avail) == 0, -1e10)
                # Random-ish: sample from slightly-uniform logits
                actions = Categorical(logits=torch.zeros_like(logits)).sample()
            reward, terminated, _ = env.step(actions.numpy().tolist())
            ep_reward += reward
            step += 1
        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, steps={step}")

    env.close()
    print("Smoke test passed!")


# ---------------------------------------------------------------------------

def run_train(args):
    """TarMAC training loop with PPO."""
    print(f"=== TarMAC Training: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"  comm_dim={args.comm_dim}  comm_rounds={args.comm_rounds}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"  device: {device}")

    env = make_env(args.race, args.n_units, args.n_enemies, render=False)
    trainer = TarMACTrainer(
        env           = env,
        device        = device,
        lr            = args.lr,
        gamma         = args.gamma,
        gae_lambda    = args.gae_lambda,
        clip_coef     = args.clip_coef,
        ent_coef      = args.ent_coef,
        vf_coef       = args.vf_coef,
        max_grad_norm = args.max_grad_norm,
        update_epochs = args.update_epochs,
        num_mini_batches = args.num_mini_batches,
        hidden_dim    = args.hidden_dim,
        comm_dim      = args.comm_dim,
        comm_rounds   = args.comm_rounds,
    )

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)

    os.makedirs(args.save_dir, exist_ok=True)
    all_rewards, all_wins, all_steps, all_times = [], [], [], []
    t_start = time.time()
    steps_done = 0

    pbar = tqdm(total=args.total_steps, desc="TarMAC")
    update_idx = 0

    while steps_done < args.total_steps:
        ep_reward, win_rate = trainer.collect_rollout(args.rollout_steps)
        metrics = trainer.update()
        steps_done += args.rollout_steps
        update_idx += 1
        pbar.update(args.rollout_steps)

        all_rewards.append(ep_reward)
        all_wins.append(float(win_rate))
        all_steps.append(steps_done)
        all_times.append(time.time() - t_start)

        if update_idx % args.log_interval == 0:
            rolling_wr = np.mean(all_wins[-20:]) if all_wins else 0.0
            elapsed    = time.time() - t_start
            pbar.write(
                f"Step {steps_done:6d} | reward={ep_reward:7.2f} | "
                f"win_rate={rolling_wr:.2%} | "
                f"pg_loss={metrics['pg_loss']:.4f} | "
                f"vf_loss={metrics['vf_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"time={elapsed:.0f}s"
            )

        if update_idx % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"smacv2_tarmac_step_{steps_done}.pt")
            trainer.save(path)

    pbar.close()
    trainer.save(os.path.join(args.save_dir, "smacv2_tarmac_latest.pt"))

    elapsed_total = time.time() - t_start
    save_plot(all_rewards, all_wins, args.save_dir)
    results = {
        "algorithm": "TarMAC",
        "race": args.race,
        "scenario": f"{args.n_units}v{args.n_enemies}",
        "total_steps": steps_done,
        "wall_clock_seconds": elapsed_total,
        "rollout_steps": args.rollout_steps,
        "comm_dim": args.comm_dim,
        "comm_rounds": args.comm_rounds,
        "final_win_rate": float(np.mean(all_wins[-10:])) if all_wins else 0.0,
        "final_mean_reward": float(np.mean(all_rewards[-10:])) if all_rewards else 0.0,
        "peak_rolling20_win_rate": float(max(
            np.mean(all_wins[max(0, i-19):i+1])
            for i in range(len(all_wins))
        )) if all_wins else 0.0,
        "rewards": all_rewards,
        "win_rate_history": all_wins,
        "steps_history": all_steps,
        "wall_clock_history": all_times,
    }
    with open(os.path.join(args.save_dir, "smacv2_tarmac_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete! {steps_done} steps in {elapsed_total:.1f}s")
    env.close()


# ---------------------------------------------------------------------------

def run_eval(args):
    """Evaluate a saved TarMAC checkpoint."""
    if not args.load_path:
        print("Error: --load-path required for eval mode")
        return

    print(f"=== TarMAC Evaluation: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"Loading: {args.load_path}")
    render = getattr(args, "render", False)
    if render:
        print("Render mode ON — StarCraft II window will open.")

    device  = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env     = make_env(args.race, args.n_units, args.n_enemies, render=render)
    env_info= env.get_env_info()

    trainer = TarMACTrainer(
        env        = env,
        device     = device,
        hidden_dim = args.hidden_dim,
        comm_dim   = args.comm_dim,
        comm_rounds= args.comm_rounds,
    )
    trainer.load(args.load_path)
    trainer.policy.eval()

    total_reward, wins = 0.0, 0

    for ep in range(args.eval_episodes):
        env.reset()
        terminated = False
        ep_reward  = 0.0

        while not terminated:
            obs_list = env.get_obs()
            obs_t    = torch.tensor(np.array(obs_list, dtype=np.float32), device=device)
            avail    = np.array([env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])],
                                dtype=np.float32)
            avail_t  = torch.tensor(avail, device=device)
            with torch.no_grad():
                hidden, context = trainer.policy.communicate_and_encode(obs_t)
                logits = trainer.policy.actor(torch.cat([hidden, context], dim=-1))
                logits = logits.masked_fill(avail_t == 0, -1e10)
                actions = logits.argmax(dim=-1)   # greedy

            reward, terminated, info = env.step(actions.cpu().numpy().tolist())
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
# Argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TarMAC on SMACv2")

    # Mode
    p.add_argument("--mode", choices=["train","eval","test"], default="train")

    # Environment
    p.add_argument("--race",     choices=["terran","protoss","zerg"], default="terran")
    p.add_argument("--n-units",  type=int, default=5)
    p.add_argument("--n-enemies",type=int, default=5)

    # TarMAC-specific
    p.add_argument("--comm-dim",   type=int, default=16,
                   help="Dimensionality of communication messages (key/value/query)")
    p.add_argument("--comm-rounds",type=int, default=1,
                   help="Number of communication rounds before acting")

    # PPO hyperparams
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae-lambda",      type=float, default=0.95)
    p.add_argument("--clip-coef",       type=float, default=0.2)
    p.add_argument("--ent-coef",        type=float, default=0.01)
    p.add_argument("--vf-coef",         type=float, default=0.5)
    p.add_argument("--max-grad-norm",   type=float, default=10.0)
    p.add_argument("--update-epochs",   type=int,   default=4)
    p.add_argument("--num-mini-batches",type=int,   default=4)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--total-steps",     type=int,   default=500_000)
    p.add_argument("--rollout-steps",   type=int,   default=200)

    # Logging / checkpointing
    p.add_argument("--save-dir",      type=str, default="checkpoints")
    p.add_argument("--load-path",     type=str, default="")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--test-episodes", type=int, default=3)
    p.add_argument("--log-interval",  type=int, default=5)
    p.add_argument("--save-interval", type=int, default=10)

    # Misc
    p.add_argument("--cpu",    action="store_true")
    p.add_argument("--render", action="store_true",
                   help="Open SC2 window for visualization (eval/test modes)")
    p.add_argument("--seed",   type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "test":
        run_test(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
