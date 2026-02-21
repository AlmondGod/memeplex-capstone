"""Standalone MAPPO training & evaluation on SMAC v2.

This script implements Multi-Agent PPO (MAPPO) with parameter sharing,
designed to work directly with SMACv2's StarCraftCapabilityEnvWrapper.

Modes:
    train — train MAPPO agents, save checkpoints, plot learning curves
    eval  — load a checkpoint and evaluate over N episodes
    test  — quick random-agent smoke test (verifies env works)

Usage:
    # Training (default: terran 5v5)
    python run_smacv2_mappo.py --mode train

    # Different scenario
    python run_smacv2_mappo.py --mode train --race protoss --n-units 10 --n-enemies 10

    # Evaluate a saved checkpoint
    python run_smacv2_mappo.py --mode eval --load-path checkpoints/smacv2_mappo_latest.pt

    # Quick smoke test (random agents, 1 episode)
    python run_smacv2_mappo.py --mode test --test-episodes 1

    # See all options
    python run_smacv2_mappo.py --help

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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm


# ===========================================================================
# SMAC v2 environment helpers
# ===========================================================================

# Race -> unit configs (matching the official SMACv2 paper configs)
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


def make_env(race: str, n_units: int, n_enemies: int):
    """Create a single SMACv2 environment instance."""
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    dist_config = build_distribution_config(race, n_units, n_enemies)
    env = StarCraftCapabilityEnvWrapper(
        capability_config=dist_config,
        map_name=RACE_MAP_NAMES[race],
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
    )
    return env


# ===========================================================================
# Neural network: Actor-Critic with parameter sharing
# ===========================================================================

class MAPPOActorCritic(nn.Module):
    """Shared actor-critic network for all agents (parameter sharing).

    Actor: obs -> action logits
    Critic: global state -> V(s)   (centralized value function)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Critic (centralized, uses global state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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
                nn.init.constant_(m.bias, 0)

    def get_actor_output(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.get_actor_output(obs)
        logits[avail_actions == 0] = -1e10
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.get_value(state)
        return log_probs, entropy, values


# ===========================================================================
# Rollout buffer
# ===========================================================================

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

    def add(self, obs, state, actions, log_probs, rewards, dones, avail_actions, values):
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
        """Compute GAE returns and advantages."""
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

    def clear(self):
        for lst in [self.obs, self.states, self.actions, self.log_probs,
                     self.rewards, self.dones, self.avail_actions, self.values]:
            lst.clear()


# ===========================================================================
# MAPPO trainer
# ===========================================================================

class MAPPOTrainer:
    """MAPPO training loop for SMAC v2."""

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
    ):
        self.env = env
        self.device = torch.device(device)

        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_shape = env_info["obs_shape"]
        self.state_shape = env_info["state_shape"]

        self.policy = MAPPOActorCritic(
            obs_dim=self.obs_shape,
            state_dim=self.state_shape,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
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

    def collect_rollout(self, rollout_steps: int):
        """Collect a rollout of `rollout_steps` transitions.

        The env is kept alive across rollout calls (no full restart each time).
        A reset only happens at the very start and when an episode terminates.
        """
        buffer = RolloutBuffer()
        episode_rewards = []
        win_count = 0
        episode_count = 0

        # Only reset at the very first call; after that continue mid-episode
        if not self._started:
            self.env.reset()
            self._started = True

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
                state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device)
                avail_t = torch.tensor(avail_arr, dtype=torch.float32, device=self.device)

                logits = self.policy.get_actor_output(obs_t)
                logits[avail_t == 0] = -1e10
                dist = Categorical(logits=logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

                state_repeated = state_t.unsqueeze(0).expand(self.n_agents, -1)
                values_t = self.policy.get_value(state_repeated)

            actions = actions_t.cpu().numpy()
            log_probs = log_probs_t.cpu().numpy()
            values = values_t.cpu().numpy()

            reward, terminated, info = self.env.step(actions.tolist())

            rewards = np.full(self.n_agents, reward, dtype=np.float32)
            dones = np.full(self.n_agents, float(terminated), dtype=np.float32)

            buffer.add(
                obs_arr,
                state_arr[np.newaxis].repeat(self.n_agents, axis=0),
                actions, log_probs, rewards, dones, avail_arr, values,
            )

            self._episode_reward += reward
            if terminated:
                episode_rewards.append(self._episode_reward)
                if info.get("battle_won", False):
                    win_count += 1
                episode_count += 1
                self._episode_reward = 0.0
                self.env.reset()   # start next episode inside same SC2 process

        # Last values for GAE
        with torch.no_grad():
            state = self.env.get_state()
            state_t = torch.tensor(np.array(state, dtype=np.float32), device=self.device)
            state_repeated = state_t.unsqueeze(0).expand(self.n_agents, -1)
            last_values = self.policy.get_value(state_repeated).cpu().numpy()

        stats = {
            "episode_rewards": episode_rewards,
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "win_rate": win_count / max(episode_count, 1),
            "episodes": episode_count,
        }
        return buffer, last_values, stats

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray) -> dict:
        """Run PPO update on collected rollout."""
        returns, advantages = buffer.compute_returns(last_values, self.gamma, self.gae_lambda)

        flat_obs = np.concatenate(buffer.obs, axis=0)
        flat_states = np.concatenate(buffer.states, axis=0)
        flat_actions = np.concatenate(buffer.actions, axis=0)
        flat_log_probs = np.concatenate(buffer.log_probs, axis=0)
        flat_avail = np.concatenate(buffer.avail_actions, axis=0)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
        states_t = torch.tensor(flat_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(flat_actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.tensor(flat_log_probs, dtype=torch.float32, device=self.device)
        avail_t = torch.tensor(flat_avail, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(flat_returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)

        batch_size = obs_t.shape[0]
        mini_batch_size = max(batch_size // self.num_mini_batches, 1)

        total_pg, total_vf, total_ent, n_updates = 0.0, 0.0, 0.0, 0

        self.policy.train()
        for _ in range(self.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                idx = indices[start:end]

                new_lp, entropy, values = self.policy.evaluate_actions(
                    obs_t[idx], states_t[idx], actions_t[idx], avail_t[idx]
                )

                ratio = torch.exp(new_lp - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * ((values - returns_t[idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "env_info": {
                "n_agents": self.n_agents,
                "n_actions": self.n_actions,
                "obs_shape": self.obs_shape,
                "state_shape": self.state_shape,
            },
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Checkpoint loaded: {path}")


# ===========================================================================
# Plotting
# ===========================================================================

def rolling_mean(values: list, window: int = 10) -> list:
    """Simple trailing rolling mean."""
    out = []
    for i in range(len(values)):
        sl = values[max(0, i - window + 1): i + 1]
        out.append(float(np.mean(sl)))
    return out


def plot_training_curves(
    steps,
    rewards,
    win_rates,
    pg_losses,
    vf_losses,
    entropies,
    save_path,
    window: int = 10,
):
    """4-panel training figure: reward, win rate, pg/vf loss, entropy."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("MAPPO on SMAC v2 — Training Curves", fontsize=13, fontweight="bold")

    # Panel 1 — Episode reward
    ax = axes[0]
    ax.plot(steps, rewards, color="tab:blue", linewidth=0.8, alpha=0.4, label="raw")
    ax.plot(steps, rolling_mean(rewards, window), color="tab:blue", linewidth=1.8, label=f"rolling({window})")
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

    # Panel 3 — Policy loss & value loss
    ax = axes[2]
    if pg_losses:
        ax.plot(steps, pg_losses, color="tab:red",    linewidth=1.5, label="pg_loss")
        ax.plot(steps, vf_losses, color="tab:orange", linewidth=1.5, label="vf_loss")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Panel 4 — Entropy
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


# ===========================================================================
# Run modes
# ===========================================================================

def run_test(args):
    """Quick smoke test with random agents."""
    print(f"=== Smoke Test: {args.race} {args.n_units}v{args.n_enemies} ===")
    env = make_env(args.race, args.n_units, args.n_enemies)
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
    """MAPPO training loop."""
    print(f"=== MAPPO Training: {args.race} {args.n_units}v{args.n_enemies} ===")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    env = make_env(args.race, args.n_units, args.n_enemies)
    trainer = MAPPOTrainer(
        env=env, device=device, lr=args.lr, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, update_epochs=args.update_epochs,
        num_mini_batches=args.num_mini_batches, hidden_dim=args.hidden_dim,
    )

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)

    total_steps = 0
    log_steps, log_rewards, log_win_rates = [], [], []
    log_pg_losses, log_vf_losses, log_entropies = [], [], []
    t_start = time.time()

    n_iters = args.total_steps // args.rollout_steps
    pbar = tqdm(total=args.total_steps, desc="MAPPO")

    for iteration in range(n_iters):
        buffer, last_values, stats = trainer.collect_rollout(args.rollout_steps)
        update_stats = trainer.update(buffer, last_values)
        buffer.clear()

        total_steps += args.rollout_steps
        pbar.update(args.rollout_steps)

        if stats["episodes"] > 0:
            log_steps.append(total_steps)
            log_rewards.append(stats["mean_reward"])
            log_win_rates.append(stats["win_rate"])
            log_pg_losses.append(update_stats["pg_loss"])
            log_vf_losses.append(update_stats["vf_loss"])
            log_entropies.append(update_stats["entropy"])

        if (iteration + 1) % args.log_interval == 0 and stats["episodes"] > 0:
            elapsed = time.time() - t_start
            pbar.set_postfix({
                "rew": f"{stats['mean_reward']:.2f}",
                "wr":  f"{stats['win_rate']:.2%}",
                "pg":  f"{update_stats['pg_loss']:.4f}",
            })
            tqdm.write(
                f"Step {total_steps:>8d} | "
                f"reward={stats['mean_reward']:>7.2f} | "
                f"win_rate={stats['win_rate']:.2%} | "
                f"pg_loss={update_stats['pg_loss']:.4f} | "
                f"vf_loss={update_stats['vf_loss']:.4f} | "
                f"entropy={update_stats['entropy']:.4f} | "
                f"time={elapsed:.0f}s"
            )

        if (iteration + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.save_dir, f"smacv2_mappo_step_{total_steps}.pt")
            trainer.save(ckpt)

    pbar.close()
    elapsed = time.time() - t_start

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, "smacv2_mappo_latest.pt")
    trainer.save(final_path)

    # Plot
    if log_steps:
        plot_path = os.path.join(args.save_dir, "smacv2_mappo_training.png")
        plot_training_curves(
            log_steps, log_rewards, log_win_rates,
            log_pg_losses, log_vf_losses, log_entropies,
            plot_path,
        )

    # Save results JSON
    results = {
        "algorithm": "MAPPO",
        "total_steps": total_steps,
        "wall_clock_seconds": elapsed,
        "race": args.race,
        "scenario": f"{args.n_units}v{args.n_enemies}",
        "final_mean_reward": float(np.mean(log_rewards[-10:])) if log_rewards else 0.0,
        "final_win_rate":    float(np.mean(log_win_rates[-10:])) if log_win_rates else 0.0,
        "final_pg_loss":     float(np.mean(log_pg_losses[-10:])) if log_pg_losses else 0.0,
        "final_vf_loss":     float(np.mean(log_vf_losses[-10:])) if log_vf_losses else 0.0,
        "final_entropy":     float(np.mean(log_entropies[-10:])) if log_entropies else 0.0,
        "rewards_history":   log_rewards,
        "win_rate_history":  log_win_rates,
        "pg_loss_history":   log_pg_losses,
        "vf_loss_history":   log_vf_losses,
        "entropy_history":   log_entropies,
        "steps_history":     log_steps,
    }
    results_path = os.path.join(args.save_dir, "smacv2_mappo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    env.close()
    print(f"\nTraining complete! {total_steps} steps in {elapsed:.1f}s")


def run_eval(args):
    """Evaluate a saved checkpoint."""
    if not args.load_path:
        print("Error: --load-path required for eval mode")
        return

    print(f"=== MAPPO Evaluation: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"Loading: {args.load_path}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_env(args.race, args.n_units, args.n_enemies)
    trainer = MAPPOTrainer(env=env, device=device, hidden_dim=args.hidden_dim)
    trainer.load(args.load_path)
    trainer.policy.eval()

    env_info = env.get_env_info()
    total_reward = 0.0
    wins = 0

    for ep in range(args.eval_episodes):
        env.reset()
        terminated = False
        ep_reward = 0.0

        while not terminated:
            obs_list = env.get_obs()
            obs_arr = np.array(obs_list, dtype=np.float32)
            avail_arr = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
            for aid in range(env_info["n_agents"]):
                avail_arr[aid] = np.array(env.get_avail_agent_actions(aid), dtype=np.float32)

            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, device=trainer.device)
                avail_t = torch.tensor(avail_arr, device=trainer.device)
                logits = trainer.policy.get_actor_output(obs_t)
                logits[avail_t == 0] = -1e10
                actions = logits.argmax(dim=-1).cpu().numpy()

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
        description="MAPPO Training & Evaluation on SMAC v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", choices=["train", "eval", "test"], default="train",
                        help="Run mode")

    # Environment
    parser.add_argument("--race", type=str, default="terran",
                        choices=["terran", "protoss", "zerg"],
                        help="SC2 race for unit generation")
    parser.add_argument("--n-units", type=int, default=5,
                        help="Number of allied units")
    parser.add_argument("--n-enemies", type=int, default=5,
                        help="Number of enemy units")

    # Training hyperparameters
    parser.add_argument("--total-steps", type=int, default=2_000_000,
                        help="Total environment steps")
    parser.add_argument("--rollout-steps", type=int, default=400,
                        help="Steps per rollout before update")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Logging & checkpointing
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint every N iterations")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints & results")

    # Eval / test
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--load-path", type=str, default="",
                        help="Path to checkpoint to load")

    # Misc
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA available")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "test":
        run_test(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
