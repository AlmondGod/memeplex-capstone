"""EGGROLL on SMACv2: Evolution Guided GeneRal Optimisation via Low-rank Learning.

Implements Algorithm 1 from "Evolution Strategies at the Hyperscale"
(Sarkar et al., arXiv:2511.16652).

Core idea
---------
Standard ES perturbs all d parameters independently: E ~ N(0, I_d).
For a weight matrix W ∈ R^{m×n}, this requires storing and computing
with full m×n noise matrices — expensive at scale.

EGGROLL instead samples A ∈ R^{m×r}, B ∈ R^{n×r} and forms:
    E = (1/√r) A B^T           (rank-r, r << min(m,n))

The update rule (Gaussian approximate score function Ŝ(E) = E) is:
    M_{t+1} ← M_t + α · (1/N) Σ_i  E_i · f̃_i

where f̃_i is fitness-shaped (global z-score normalisation across the population).

For SMACv2 we run N sequential "workers" (episodes) rather than true
GPU parallelism (CPU-only Mac), but the algorithm is identical.

Modes
-----
    train  — train an EGGROLL policy, save checkpoints, plot curves
    eval   — evaluate a saved checkpoint
    test   — quick smoke test (random actions, verifies env works)

Usage
-----
    python run_smacv2_eggroll.py --mode train
    python run_smacv2_eggroll.py --mode train --rank 2 --pop-size 16 --sigma 0.02
    python run_smacv2_eggroll.py --mode test --test-episodes 1
    python run_smacv2_eggroll.py --mode eval --load-path checkpoints/smacv2_eggroll_latest.pt

References
----------
    Sarkar et al. (2026). Evolution Strategies at the Hyperscale.
        arXiv:2511.16652.
    Salimans et al. (2017). Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning. arXiv:1703.03864.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if not os.environ.get("SC2PATH"):
    _mac_default = "/Applications/StarCraft II"
    if os.path.isdir(_mac_default):
        os.environ["SC2PATH"] = _mac_default

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ===========================================================================
# SMACv2 environment helpers (identical to the other run_smacv2_*.py scripts)
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
    "terran":  "10gen_terran",
    "protoss": "10gen_protoss",
    "zerg":    "10gen_zerg",
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
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    dist_config = build_distribution_config(race, n_units, n_enemies)
    window_x, window_y = (1280, 720) if render else (640, 480)
    return StarCraftCapabilityEnvWrapper(
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


# ===========================================================================
# Policy network
# ===========================================================================

class EGGROLLPolicy(nn.Module):
    """Simple shared MLP policy for all agents.

    Input:  local observation (obs_dim,)
    Output: action logits (n_actions,)

    Parameter sharing: all agents use the same network weights.
    EGGROLL perturbs these shared weights for each population member.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits for a batch of observations."""
        return self.net(obs)

    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a single flat vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Set all parameters from a flat vector."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view(p.shape))
            idx += n

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Low-rank perturbation helpers  (core EGGROLL contribution)
# ===========================================================================

def sample_lowrank_perturbation(
    param_shapes: List[Tuple[int, ...]],
    rank: int,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Sample one rank-r perturbation E = (1/√r) A B^T for each parameter tensor.

    For 2-D parameters (weight matrices) we sample:
        A ∈ R^{m×r},  B ∈ R^{n×r}
        E = (1/√r) A B^T  ∈ R^{m×n}

    For 1-D parameters (bias vectors of length d) we treat them as (d,1)
    matrices:
        A ∈ R^{d×r},  B ∈ R^{1×r}
        E = (1/√r) A B^T  ∈ R^{d×1}  -> reshaped to (d,)

    Returns
    -------
    factors : list of (A, B) tensors, one per parameter tensor
    flat_E  : the full concatenated flat perturbation vector (same shape as
              get_flat_params()) — used for the parameter update step.
    """
    factors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    flat_parts: List[torch.Tensor] = []

    for shape in param_shapes:
        if len(shape) >= 2:
            m = shape[0]
            n = int(np.prod(shape[1:]))          # flatten remaining dims
        else:
            m = shape[0]
            n = 1

        A = torch.randn(m, rank, device=device)
        B = torch.randn(n, rank, device=device)
        E = (1.0 / (rank ** 0.5)) * (A @ B.T)   # (m, n)
        factors.append((A, B))
        flat_parts.append(E.view(-1))

    flat_E = torch.cat(flat_parts)               # (total_params,)
    return factors, flat_E


# ===========================================================================
# Episode rollout
# ===========================================================================

def rollout_episode(
    env,
    policy: EGGROLLPolicy,
    n_agents: int,
    n_actions: int,
    device: torch.device,
    max_steps: int = 200,
) -> Tuple[float, bool]:
    """Run one full episode with the given policy weights.

    Returns (episode_reward, battle_won).
    """
    env.reset()
    episode_reward = 0.0
    won = False

    for _ in range(max_steps):
        obs_list = env.get_obs()
        avail_list = [env.get_avail_agent_actions(i) for i in range(n_agents)]

        obs_t = torch.tensor(np.array(obs_list, dtype=np.float32), device=device)
        avail_t = torch.tensor(np.array(avail_list, dtype=np.float32), device=device)

        with torch.no_grad():
            logits = policy(obs_t)                     # (N, n_actions)
            logits = logits.masked_fill(avail_t == 0, -1e9)
            actions = logits.argmax(dim=-1).cpu().numpy()   # greedy at eval

        reward, terminated, info = env.step(actions.tolist())
        episode_reward += reward

        if terminated:
            won = bool(info.get("battle_won", False))
            break

    return episode_reward, won


# ===========================================================================
# EGGROLL trainer
# ===========================================================================

class EGGROLLTrainer:
    """EGGROLL trainer for SMACv2.

    Implements Algorithm 1 from Sarkar et al. (2026), adapted for sequential
    episode-based fitness evaluation on CPU.

    Parameters
    ----------
    env            : SMACv2 environment
    pop_size       : N — number of population members per update (= N_workers)
    rank           : r — rank of each perturbation matrix (r=1 recommended)
    sigma          : σ — perturbation noise scale
    lr             : α — learning rate for the mean parameter update
    hidden_dim     : width of the shared MLP policy
    device         : torch device
    fitness_norm   : how to normalize fitnesses before the weighted update.
                     "zscore"  — global z-score across population (used in §6.3)
                     "rank"    — rank-based normalization (OpenAI-ES style)
                     "none"    — raw fitnesses
    antithetic     : if True, use antithetic sampling: for each (A,B) also
                     evaluate (-A, B), halving variance. pop_size must be even.
    """

    def __init__(
        self,
        env,
        pop_size: int = 16,
        rank: int = 1,
        sigma: float = 0.02,
        lr: float = 0.01,
        hidden_dim: int = 64,
        device: torch.device = torch.device("cpu"),
        fitness_norm: str = "zscore",
        antithetic: bool = True,
    ):
        self.env = env
        self.pop_size = pop_size
        self.rank = rank
        self.sigma = sigma
        self.lr = lr
        self.device = device
        self.fitness_norm = fitness_norm
        self.antithetic = antithetic

        env_info = env.get_env_info()
        self.n_agents  = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_dim   = env_info["obs_shape"]

        # Mean parameters θ (the policy we optimize)
        self.policy = EGGROLLPolicy(self.obs_dim, self.n_actions, hidden_dim).to(device)
        self._param_shapes = [p.shape for p in self.policy.parameters()]
        self._n_params = self.policy.n_params

        if antithetic:
            assert pop_size % 2 == 0, "pop_size must be even when antithetic=True"

    # ------------------------------------------------------------------
    def _normalize_fitnesses(self, fitnesses: np.ndarray) -> np.ndarray:
        """Shape fitnesses for the ES gradient estimate."""
        if self.fitness_norm == "zscore":
            std = fitnesses.std()
            if std < 1e-8:
                return np.zeros_like(fitnesses)
            return (fitnesses - fitnesses.mean()) / std

        elif self.fitness_norm == "rank":
            # Rank-based normalization (Salimans et al. 2017)
            n = len(fitnesses)
            ranks = np.argsort(np.argsort(fitnesses))   # 0..n-1
            return (ranks / (n - 1) - 0.5)              # in [-0.5, 0.5]

        elif self.fitness_norm == "none":
            return fitnesses

        else:
            raise ValueError(f"Unknown fitness_norm: {self.fitness_norm}")

    # ------------------------------------------------------------------
    def update(self, max_episode_steps: int = 200) -> Dict[str, float]:
        """Run one EGGROLL update step.

        For each of the N population members:
          1. Sample rank-r perturbation E_i = (1/√r) A_i B_i^T
          2. Evaluate fitness f_i = episode_reward(θ + σ E_i)
        Then shape fitnesses and update:
          θ ← θ + α · (1/N) Σ_i E_i · f̃_i

        Returns a dict of logging stats.
        """
        base_params = self.policy.get_flat_params().clone()  # θ_t

        # --- 1. Sample perturbations and evaluate fitnesses ---
        flat_Es: List[torch.Tensor] = []
        fitnesses: List[float] = []
        win_count = 0
        episode_count = 0

        n_positive = self.pop_size if not self.antithetic else self.pop_size // 2

        for i in range(n_positive):
            # Sample rank-r perturbation for each parameter tensor
            _, flat_E = sample_lowrank_perturbation(
                self._param_shapes, self.rank, self.device
            )

            # Positive perturbation: θ + σ E
            perturbed = base_params + self.sigma * flat_E
            self.policy.set_flat_params(perturbed)
            f_pos, won_pos = rollout_episode(
                self.env, self.policy, self.n_agents,
                self.n_actions, self.device, max_episode_steps
            )
            flat_Es.append(flat_E)
            fitnesses.append(f_pos)
            win_count += int(won_pos)
            episode_count += 1

            if self.antithetic:
                # Antithetic: θ - σ E  (mirrors the same noise direction)
                perturbed_neg = base_params - self.sigma * flat_E
                self.policy.set_flat_params(perturbed_neg)
                f_neg, won_neg = rollout_episode(
                    self.env, self.policy, self.n_agents,
                    self.n_actions, self.device, max_episode_steps
                )
                flat_Es.append(-flat_E)
                fitnesses.append(f_neg)
                win_count += int(won_neg)
                episode_count += 1

        # Restore mean params (evaluation shouldn't change θ)
        self.policy.set_flat_params(base_params)

        # --- 2. Shape fitnesses ---
        fitnesses_arr = np.array(fitnesses, dtype=np.float32)
        shaped = self._normalize_fitnesses(fitnesses_arr)

        # --- 3. EGGROLL parameter update ---
        # M_{t+1} = M_t + α · (1/N) Σ_i E_i · f̃_i
        update_vec = torch.zeros(self._n_params, device=self.device)
        for flat_E, f_shaped in zip(flat_Es, shaped):
            update_vec += flat_E * float(f_shaped)
        update_vec /= len(flat_Es)

        new_params = base_params + self.lr * update_vec
        self.policy.set_flat_params(new_params)

        win_rate = win_count / max(episode_count, 1)
        return {
            "mean_fitness":  float(fitnesses_arr.mean()),
            "max_fitness":   float(fitnesses_arr.max()),
            "min_fitness":   float(fitnesses_arr.min()),
            "std_fitness":   float(fitnesses_arr.std()),
            "win_rate":      win_rate,
            "update_norm":   float(update_vec.norm().item()),
            "param_norm":    float(new_params.norm().item()),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "policy_state": self.policy.state_dict(),
            "obs_dim":      self.obs_dim,
            "n_actions":    self.n_actions,
            "n_agents":     self.n_agents,
            "rank":         self.rank,
            "sigma":        self.sigma,
            "lr":           self.lr,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])


# ===========================================================================
# Plot helper
# ===========================================================================

def plot_training(
    steps: List[int],
    win_rates: List[float],
    fitnesses: List[float],
    save_dir: str,
) -> None:
    roll_n = min(20, len(win_rates))
    roll_wr  = np.convolve(win_rates,  np.ones(roll_n) / roll_n, mode="same")
    roll_fit = np.convolve(fitnesses,  np.ones(roll_n) / roll_n, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(steps, win_rates,  alpha=0.3, color="tab:green", label="raw")
    axes[0].plot(steps, roll_wr,    linewidth=2, color="tab:green", label=f"rolling-{roll_n}")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("EGGROLL on SMACv2 — Win Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, fitnesses,  alpha=0.3, color="tab:blue", label="raw")
    axes[1].plot(steps, roll_fit,   linewidth=2, color="tab:blue", label=f"rolling-{roll_n}")
    axes[1].set_ylabel("Mean Episode Fitness")
    axes[1].set_xlabel("Environment Steps")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "smacv2_eggroll_training.png"), dpi=120)
    plt.close()


# ===========================================================================
# Training loop
# ===========================================================================

def run_train(args) -> None:
    print(f"=== EGGROLL Training: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"  rank={args.rank}  pop_size={args.pop_size}  sigma={args.sigma}"
          f"  lr={args.lr}  antithetic={not args.no_antithetic}")

    device = torch.device("cpu")
    env = make_env(args.race, args.n_units, args.n_enemies)

    trainer = EGGROLLTrainer(
        env        = env,
        pop_size   = args.pop_size,
        rank       = args.rank,
        sigma      = args.sigma,
        lr         = args.lr,
        hidden_dim = args.hidden_dim,
        device     = device,
        fitness_norm = args.fitness_norm,
        antithetic = not args.no_antithetic,
    )

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)
        print(f"  Loaded checkpoint: {args.load_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Each EGGROLL update consumes pop_size episodes.
    # We track steps as pop_size * max_episode_steps * update_idx
    # (approximate — actual steps depend on episode length).
    # For a fair comparison with on-policy methods we log cumulative
    # environment steps (sum of actual steps taken in all episodes).

    all_win_rates: List[float]  = []
    all_fitnesses: List[float]  = []
    all_steps:     List[int]    = []
    all_times:     List[float]  = []

    t_start      = time.time()
    total_steps  = 0          # cumulative env steps across all episodes
    update_idx   = 0

    # Each update: pop_size episodes * up to max_episode_steps steps each.
    # We estimate steps per update and stop when >= args.total_steps.
    pbar = tqdm(total=args.total_steps, desc="EGGROLL")

    while total_steps < args.total_steps:
        metrics = trainer.update(max_episode_steps=args.max_episode_steps)
        update_idx += 1

        # Each update runs pop_size episodes; approximate step count
        steps_this_update = args.pop_size * args.max_episode_steps
        total_steps += steps_this_update
        pbar.update(min(steps_this_update, args.total_steps - (total_steps - steps_this_update)))

        all_win_rates.append(metrics["win_rate"])
        all_fitnesses.append(metrics["mean_fitness"])
        all_steps.append(total_steps)
        all_times.append(time.time() - t_start)

        if update_idx % args.log_interval == 0:
            roll20_wr = float(np.mean(all_win_rates[-20:]))
            elapsed   = time.time() - t_start
            pbar.write(
                f"Update {update_idx:5d} | steps~{total_steps:8d} | "
                f"fitness={metrics['mean_fitness']:7.2f} | "
                f"win_rate={metrics['win_rate']:.2%} | "
                f"roll20_wr={roll20_wr:.2%} | "
                f"update_norm={metrics['update_norm']:.4f} | "
                f"time={elapsed:.0f}s"
            )

        if update_idx % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"smacv2_eggroll_update_{update_idx}.pt")
            trainer.save(ckpt_path)

    pbar.close()
    elapsed_total = time.time() - t_start

    # Save final checkpoint
    trainer.save(os.path.join(args.save_dir, "smacv2_eggroll_latest.pt"))

    # Plot
    plot_training(all_steps, all_win_rates, all_fitnesses, args.save_dir)

    # Compute rolling-20 win rate history
    roll20_wr = [
        float(np.mean(all_win_rates[max(0, i - 19): i + 1]))
        for i in range(len(all_win_rates))
    ]

    results = {
        "algorithm":               "EGGROLL",
        "race":                    args.race,
        "scenario":                f"{args.n_units}v{args.n_enemies}",
        "rank":                    args.rank,
        "pop_size":                args.pop_size,
        "sigma":                   args.sigma,
        "lr":                      args.lr,
        "antithetic":              not args.no_antithetic,
        "fitness_norm":            args.fitness_norm,
        "total_steps_approx":      total_steps,
        "total_updates":           update_idx,
        "wall_clock_seconds":      elapsed_total,
        "final_win_rate":          float(np.mean(all_win_rates[-10:])),
        "final_mean_fitness":      float(np.mean(all_fitnesses[-10:])),
        "peak_rolling20_win_rate": float(max(roll20_wr)) if roll20_wr else 0.0,
        "win_rate_history":        all_win_rates,
        "fitness_history":         all_fitnesses,
        "steps_history":           all_steps,
        "wall_clock_history":      all_times,
        "rolling20_win_rate":      roll20_wr,
    }

    results_path = os.path.join(args.save_dir, "smacv2_eggroll_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! ~{total_steps:,} steps ({update_idx} updates) "
          f"in {elapsed_total:.1f}s")
    print(f"Final win rate (last 10 updates): {results['final_win_rate']:.1%}")
    print(f"Peak rolling-20 win rate: {results['peak_rolling20_win_rate']:.1%}")
    print(f"Results: {results_path}")
    env.close()


# ===========================================================================
# Evaluation
# ===========================================================================

def run_eval(args) -> None:
    if not args.load_path:
        print("Error: --load-path required for eval mode")
        return

    print(f"=== EGGROLL Eval: {args.race} {args.n_units}v{args.n_enemies} ===")
    device = torch.device("cpu")
    env    = make_env(args.race, args.n_units, args.n_enemies, render=args.render)

    # Reconstruct policy shape from checkpoint
    ckpt = torch.load(args.load_path, map_location=device)
    policy = EGGROLLPolicy(
        obs_dim    = ckpt["obs_dim"],
        n_actions  = ckpt["n_actions"],
        hidden_dim = args.hidden_dim,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    n_agents  = ckpt["n_agents"]
    n_actions = ckpt["n_actions"]

    total_reward = 0.0
    wins = 0
    for ep in range(args.eval_episodes):
        r, won = rollout_episode(
            env, policy, n_agents, n_actions, device, args.max_episode_steps
        )
        total_reward += r
        wins += int(won)
        print(f"  Episode {ep + 1}: reward={r:.2f}, won={won}")

    print(f"\nMean reward: {total_reward / args.eval_episodes:.2f}")
    print(f"Win rate:    {wins / args.eval_episodes:.1%}")
    env.close()


# ===========================================================================
# Smoke test
# ===========================================================================

def run_test(args) -> None:
    """Verify env and one EGGROLL update work without crashing."""
    print(f"=== EGGROLL Smoke Test: {args.race} {args.n_units}v{args.n_enemies} ===")
    device = torch.device("cpu")
    env    = make_env(args.race, args.n_units, args.n_enemies)

    # Minimal trainer: pop=4, rank=1, 1 update of up to 10 steps per episode
    trainer = EGGROLLTrainer(
        env        = env,
        pop_size   = 4,
        rank       = args.rank,
        sigma      = args.sigma,
        lr         = args.lr,
        hidden_dim = args.hidden_dim,
        device     = device,
        fitness_norm = args.fitness_norm,
        antithetic = False,
    )

    print(f"  Policy params: {trainer._n_params:,}")
    print(f"  obs_dim={trainer.obs_dim}  n_actions={trainer.n_actions}"
          f"  n_agents={trainer.n_agents}")
    print("  Running 1 EGGROLL update (4 episodes, max 20 steps each)...")

    metrics = trainer.update(max_episode_steps=20)

    print(f"  mean_fitness = {metrics['mean_fitness']:.3f}")
    print(f"  win_rate     = {metrics['win_rate']:.2%}")
    print(f"  update_norm  = {metrics['update_norm']:.6f}")
    print(f"  param_norm   = {metrics['param_norm']:.4f}")
    print("Smoke test PASSED ✓")
    env.close()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="EGGROLL on SMACv2")
    p.add_argument("--mode", choices=["train", "eval", "test"], default="train")

    # Environment
    p.add_argument("--race",     default="terran", choices=["terran", "protoss", "zerg"])
    p.add_argument("--n-units",   type=int, default=5)
    p.add_argument("--n-enemies", type=int, default=5)

    # EGGROLL hyperparameters
    p.add_argument("--rank",       type=int,   default=1,
                   help="Rank r of each perturbation matrix (r=1 is the paper default)")
    p.add_argument("--pop-size",   type=int,   default=16,
                   help="Population size N (number of episodes per update)")
    p.add_argument("--sigma",      type=float, default=0.02,
                   help="Perturbation noise scale σ")
    p.add_argument("--lr",         type=float, default=0.01,
                   help="Learning rate α for parameter update")
    p.add_argument("--no-antithetic", action="store_true",
                   help="Disable antithetic sampling (enabled by default)")
    p.add_argument("--fitness-norm", default="zscore",
                   choices=["zscore", "rank", "none"],
                   help="Fitness normalization before the ES update")

    # Network
    p.add_argument("--hidden-dim", type=int, default=64)

    # Training
    p.add_argument("--total-steps",       type=int, default=1_000_000,
                   help="Approximate total env steps (= pop_size * max_ep_steps * updates)")
    p.add_argument("--max-episode-steps", type=int, default=200,
                   help="Max steps per episode during fitness evaluation")
    p.add_argument("--log-interval",      type=int, default=10,
                   help="Log every N updates")
    p.add_argument("--save-interval",     type=int, default=50,
                   help="Save checkpoint every N updates")
    p.add_argument("--save-dir",          default="checkpoints/smacv2_eggroll")
    p.add_argument("--load-path",         default=None)

    # Eval
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--render",        action="store_true")

    # Smoke test
    p.add_argument("--test-episodes", type=int, default=1)

    args = p.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "test":
        run_test(args)


if __name__ == "__main__":
    main()
