"""Independent PPO (baseline) on MPE2 simple_tag_v3 (predator-prey) with self-play.

Key evaluation metrics (competitive setting):
- Rolling opponent pool of frozen policies
- Elo vs the pool
- Exploitability approximation (min win-rate vs pool)
- Win-rate matrix (periodic) to detect cycles

Usage:
    python train_ppo.py
    python train_ppo.py --max-steps 200000 --num-envs 8
    python train_ppo.py --max-time 780 --results-key ppo_wall_clock_match
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpe2 import simple_tag_v3
from tqdm import tqdm

from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from method_i import LatentAlignedIPPO


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(
    max_cycles: int,
    continuous_actions: bool,
    num_good: int,
    num_adversaries: int,
    num_obstacles: int,
    render_mode: str | None = None,
):
    return simple_tag_v3.parallel_env(
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        render_mode=render_mode,
    )


def split_teams(agent_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    predators = [a for a in agent_ids if a.startswith("adversary")]
    prey = [a for a in agent_ids if not a.startswith("adversary")]
    return predators, prey


def build_dones(
    terminations: Dict[str, np.ndarray],
    truncations: Dict[str, np.ndarray],
    agent_ids: List[str],
) -> Dict[str, np.ndarray]:
    dones: Dict[str, np.ndarray] = {}
    for aid in agent_ids:
        terminated = terminations.get(aid, True)
        truncated = truncations.get(aid, False)
        terminated = np.where(np.isnan(terminated), True, terminated).astype(bool)
        truncated = np.where(np.isnan(truncated), False, truncated).astype(bool)
        dones[aid] = terminated | truncated
    return dones


def team_step_rewards(
    rewards: Dict[str, np.ndarray],
    team_ids: List[str],
    num_envs: int,
) -> np.ndarray:
    if not team_ids:
        return np.zeros(num_envs, dtype=np.float32)
    total = np.zeros(num_envs, dtype=np.float32)
    for aid in team_ids:
        r = rewards.get(aid)
        if r is None:
            continue
        r = np.where(np.isnan(r), 0.0, r).astype(np.float32)
        total += r
    return total


# ---------------------------------------------------------------------------
# Competitive metrics (Elo / exploitability / win-rate matrix)
# ---------------------------------------------------------------------------

def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> tuple[float, float]:
    exp_a = elo_expected(r_a, r_b)
    exp_b = 1.0 - exp_a
    new_a = r_a + k * (score_a - exp_a)
    new_b = r_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b


def run_episode(
    pred_agent: LatentAlignedIPPO | None,
    prey_agent: LatentAlignedIPPO | None,
    env_kwargs: dict,
    seed: int | None = None,
) -> tuple[float, float, float]:
    env = make_env(render_mode=None, **env_kwargs)
    obs, info = env.reset(seed=seed)
    predators, prey = split_teams(env.agents)

    pred_score = 0.0
    prey_score = 0.0
    done = False

    if pred_agent is not None:
        pred_agent.set_training_mode(False)
    if prey_agent is not None:
        prey_agent.set_training_mode(False)

    while not done:
        if pred_agent is None:
            pred_actions = {a: env.action_space(a).sample() for a in predators}
        else:
            pred_obs = {a: obs[a][np.newaxis, :] for a in predators}
            pred_action, _ = pred_agent.get_action(obs=pred_obs)
            pred_actions = {a: pred_action[a].squeeze() for a in predators}

        if prey_agent is None:
            prey_actions = {a: env.action_space(a).sample() for a in prey}
        else:
            prey_obs = {a: obs[a][np.newaxis, :] for a in prey}
            prey_action, _ = prey_agent.get_action(obs=prey_obs)
            prey_actions = {a: prey_action[a].squeeze() for a in prey}

        action = {**pred_actions, **prey_actions}
        obs, reward, termination, truncation, info = env.step(action)
        pred_score += sum(reward[a] for a in predators)
        prey_score += sum(reward[a] for a in prey)
        done = all(termination.values()) or all(truncation.values())

    env.close()

    pred_avg = pred_score / max(len(predators), 1)
    prey_avg = prey_score / max(len(prey), 1)
    if pred_score > prey_score:
        win = 1.0
    elif pred_score < prey_score:
        win = 0.0
    else:
        win = 0.5
    return win, pred_avg, prey_avg


def compute_win_rate(
    pred_agent: LatentAlignedIPPO | None,
    prey_agent: LatentAlignedIPPO | None,
    env_kwargs: dict,
    episodes: int,
    seed: int | None = None,
) -> float:
    wins = []
    for ep in range(episodes):
        win, _, _ = run_episode(
            pred_agent,
            prey_agent,
            env_kwargs,
            seed=None if seed is None else seed + ep,
        )
        wins.append(win)
    return float(np.mean(wins)) if wins else 0.0


def load_agent(
    path: str,
    observation_spaces,
    action_spaces,
    device: str = "cpu",
) -> LatentAlignedIPPO:
    agent = LatentAlignedIPPO.load(
        path,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=device,
    )
    agent.set_training_mode(False)
    return agent


def update_pool(
    pool: list[dict],
    new_entry: dict,
    pool_size: int,
) -> list[dict]:
    pool.append(new_entry)
    checkpoint_entries = [e for e in pool if e["type"] == "checkpoint"]
    if len(checkpoint_entries) > pool_size:
        for idx, entry in enumerate(pool):
            if entry["type"] == "checkpoint":
                to_remove = pool.pop(idx)
                path = to_remove.get("path")
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                break
    return pool


def evaluate_predator_vs_pool(
    pred_agent: LatentAlignedIPPO,
    prey_pool: list[dict],
    env_kwargs: dict,
    episodes: int,
    seed: int,
    elo_pred: float,
    elo_prey_pool: dict,
    k: float,
    prey_obs_spaces,
    prey_action_spaces,
) -> tuple[float, dict, list[float], float]:
    win_rates = []
    pred_agent.set_training_mode(False)

    for idx, opponent in enumerate(prey_pool):
        opp_id = opponent["id"]
        opp_rating = float(elo_prey_pool.get(opp_id, 1000.0))
        if opponent["type"] == "random":
            prey_agent = None
        else:
            prey_agent = load_agent(
                opponent["path"],
                observation_spaces=prey_obs_spaces,
                action_spaces=prey_action_spaces,
                device="cpu",
            )
        win_rate = compute_win_rate(
            pred_agent,
            prey_agent,
            env_kwargs,
            episodes=episodes,
            seed=seed + idx * 1000,
        )
        win_rates.append(win_rate)
        elo_pred, opp_rating = elo_update(elo_pred, opp_rating, win_rate, k)
        elo_prey_pool[opp_id] = opp_rating

    exploitability = 0.0
    if win_rates:
        exploitability = max(0.0, 0.5 - min(win_rates))

    pred_agent.set_training_mode(True)
    return elo_pred, elo_prey_pool, win_rates, exploitability


def evaluate_prey_vs_pool(
    prey_agent: LatentAlignedIPPO,
    pred_pool: list[dict],
    env_kwargs: dict,
    episodes: int,
    seed: int,
    elo_prey: float,
    elo_pred_pool: dict,
    k: float,
    pred_obs_spaces,
    pred_action_spaces,
) -> tuple[float, dict, list[float], float]:
    win_rates = []
    prey_agent.set_training_mode(False)

    for idx, opponent in enumerate(pred_pool):
        opp_id = opponent["id"]
        opp_rating = float(elo_pred_pool.get(opp_id, 1000.0))
        if opponent["type"] == "random":
            pred_agent = None
        else:
            pred_agent = load_agent(
                opponent["path"],
                observation_spaces=pred_obs_spaces,
                action_spaces=pred_action_spaces,
                device="cpu",
            )
        pred_win = compute_win_rate(
            pred_agent,
            prey_agent,
            env_kwargs,
            episodes=episodes,
            seed=seed + idx * 1000,
        )
        prey_score = 1.0 - pred_win
        win_rates.append(prey_score)
        elo_prey, opp_rating = elo_update(elo_prey, opp_rating, prey_score, k)
        elo_pred_pool[opp_id] = opp_rating

    exploitability = 0.0
    if win_rates:
        exploitability = max(0.0, 0.5 - min(win_rates))

    prey_agent.set_training_mode(True)
    return elo_prey, elo_pred_pool, win_rates, exploitability


def compute_winrate_matrix(
    pred_pool: list[dict],
    prey_pool: list[dict],
    env_kwargs: dict,
    episodes: int,
    seed: int,
    pred_obs_spaces,
    pred_action_spaces,
    prey_obs_spaces,
    prey_action_spaces,
) -> tuple[np.ndarray, list[str], list[str]]:
    pred_labels = [p["id"] for p in pred_pool]
    prey_labels = [p["id"] for p in prey_pool]
    matrix = np.zeros((len(pred_pool), len(prey_pool)), dtype=np.float32)

    pred_cache: dict[str, LatentAlignedIPPO | None] = {}
    prey_cache: dict[str, LatentAlignedIPPO | None] = {}

    for pred in pred_pool:
        if pred["type"] == "random":
            pred_cache[pred["id"]] = None
        else:
            pred_cache[pred["id"]] = load_agent(
                pred["path"],
                observation_spaces=pred_obs_spaces,
                action_spaces=pred_action_spaces,
                device="cpu",
            )

    for prey in prey_pool:
        if prey["type"] == "random":
            prey_cache[prey["id"]] = None
        else:
            prey_cache[prey["id"]] = load_agent(
                prey["path"],
                observation_spaces=prey_obs_spaces,
                action_spaces=prey_action_spaces,
                device="cpu",
            )

    for i, pred in enumerate(pred_pool):
        for j, prey in enumerate(prey_pool):
            pred_agent = pred_cache[pred["id"]]
            prey_agent = prey_cache[prey["id"]]
            win_rate = compute_win_rate(
                pred_agent,
                prey_agent,
                env_kwargs,
                episodes=episodes,
                seed=seed + i * 1000 + j * 10,
            )
            matrix[i, j] = win_rate

    return matrix, pred_labels, prey_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(
    eval_steps: List[int],
    elo_pred: List[float],
    elo_prey: List[float],
    exploit_pred: List[float],
    exploit_prey: List[float],
    save_path: str,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(eval_steps, elo_pred, color="tab:red", label="predator Elo")
    ax1.plot(eval_steps, elo_prey, color="tab:green", label="prey Elo")
    ax1.set_ylabel("Elo vs Opponent Pool")
    ax1.set_title("PPO on MPE2 simple_tag_v3 — Elo vs Pool")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(eval_steps, exploit_pred, color="tab:red", label="predator exploitability")
    ax2.plot(eval_steps, exploit_prey, color="tab:green", label="prey exploitability")
    ax2.set_xlabel("Env Steps")
    ax2.set_ylabel("Exploitability (approx)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {save_path}")


def plot_winrate_matrix(
    matrix: np.ndarray,
    pred_labels: list[str],
    prey_labels: list[str],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(prey_labels)))
    ax.set_yticks(range(len(pred_labels)))
    ax.set_xticklabels(prey_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pred_labels, fontsize=7)
    ax.set_xlabel("Prey policies")
    ax.set_ylabel("Predator policies")
    ax.set_title("Win-rate Matrix (predator win rate)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Win-rate matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--continuous-actions", action="store_true", default=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--max-time", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--elo-episodes", type=int, default=5)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--matrix-interval", type=int, default=100_000)
    parser.add_argument("--matrix-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-json", default="experiment_results.json")
    parser.add_argument("--results-key", default="")
    parser.add_argument("--plot-path", default="")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_kwargs = dict(
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous_actions,
        num_good=args.num_good,
        num_adversaries=args.num_adversaries,
        num_obstacles=args.num_obstacles,
    )

    # --- PPO hyperparameters ---
    num_envs = args.num_envs
    max_steps = args.max_steps
    rollout_steps = 128
    eval_interval = args.eval_interval
    max_time = args.max_time

    results_key = (args.results_key or "").strip()
    if not results_key:
        results_key = "ppo_wall_clock_match" if max_time > 0 else "ppo"

    lr = 3e-4
    gamma = 0.95
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    batch_size = 64

    latent_dim = 64
    hidden_dims = [128, 128]

    # Distillation disabled for PPO baseline
    distill_interval = 10
    distill_weight = 0.0
    distill_lr = 1e-3
    distill_batch_size = 64
    obs_buffer_size = 512

    distill_anneal = False
    distill_anneal_end = 0.0
    est_learn_steps = max_steps // (rollout_steps * num_envs)
    est_distill_events = max(est_learn_steps // distill_interval, 1)
    distill_anneal_steps = max(int(est_distill_events * 0.6), 1)

    lr_anneal = True
    lr_anneal_total_steps = max(est_learn_steps, 1)

    # --- Environment setup ---
    env_fns = [partial(make_env, **env_kwargs) for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)
    obs, info = env.reset(seed=args.seed)

    all_agent_ids = list(env.agents)
    predators, prey = split_teams(all_agent_ids)
    if not predators or not prey:
        raise ValueError("simple_tag_v3 requires both predators and prey.")

    pred_obs_spaces = [env.single_observation_space(a) for a in predators]
    pred_action_spaces = [env.single_action_space(a) for a in predators]
    prey_obs_spaces = [env.single_observation_space(a) for a in prey]
    prey_action_spaces = [env.single_action_space(a) for a in prey]

    print(f"Agents: {all_agent_ids}")
    print(f"Predators: {predators}")
    print(f"Prey: {prey}")

    # --- Create agents ---
    pred_agent = LatentAlignedIPPO(
        observation_spaces=pred_obs_spaces,
        action_spaces=pred_action_spaces,
        agent_ids=predators,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        update_epochs=update_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        distill_interval=distill_interval,
        distill_weight=distill_weight,
        distill_lr=distill_lr,
        distill_batch_size=distill_batch_size,
        obs_buffer_size=obs_buffer_size,
        distill_anneal=distill_anneal,
        distill_anneal_end=distill_anneal_end,
        distill_anneal_steps=distill_anneal_steps,
        lr_anneal=lr_anneal,
        lr_anneal_total_steps=lr_anneal_total_steps,
        device=str(device),
    )

    prey_agent = LatentAlignedIPPO(
        observation_spaces=prey_obs_spaces,
        action_spaces=prey_action_spaces,
        agent_ids=prey,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        update_epochs=update_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        distill_interval=distill_interval,
        distill_weight=distill_weight,
        distill_lr=distill_lr,
        distill_batch_size=distill_batch_size,
        obs_buffer_size=obs_buffer_size,
        distill_anneal=distill_anneal,
        distill_anneal_end=distill_anneal_end,
        distill_anneal_steps=distill_anneal_steps,
        lr_anneal=lr_anneal,
        lr_anneal_total_steps=lr_anneal_total_steps,
        device=str(device),
    )

    # --- Opponent pools ---
    pool_pred: list[dict] = [{"id": "random_pred", "type": "random", "path": None}]
    pool_prey: list[dict] = [{"id": "random_prey", "type": "random", "path": None}]
    elo_pred_pool = {"random_pred": 1000.0}
    elo_prey_pool = {"random_prey": 1000.0}
    elo_pred = 1000.0
    elo_prey = 1000.0

    pool_pred_dir = Path(f"checkpoints/pool_pred_{results_key}")
    pool_prey_dir = Path(f"checkpoints/pool_prey_{results_key}")
    pool_pred_dir.mkdir(parents=True, exist_ok=True)
    pool_prey_dir.mkdir(parents=True, exist_ok=True)

    time_tag = f", max {max_time:.0f}s" if max_time > 0 else ""
    print(
        f"\nStarting PPO self-play for {max_steps} env steps with {num_envs} envs{time_tag}..."
    )

    total_steps = 0
    last_eval = 0
    last_matrix = 0
    t_start = time.time()

    eval_steps: List[int] = []
    elo_pred_hist: List[float] = []
    elo_prey_hist: List[float] = []
    exploit_pred_hist: List[float] = []
    exploit_prey_hist: List[float] = []
    winrate_matrix_paths: List[str] = []

    episode_scores = {
        "predator": np.zeros(num_envs, dtype=np.float32),
        "prey": np.zeros(num_envs, dtype=np.float32),
    }
    completed_scores = {"predator": [], "prey": []}

    pbar = tqdm(total=max_steps, desc="PPO")

    while total_steps < max_steps:
        if max_time > 0 and (time.time() - t_start) >= max_time:
            break

        pred_agent.set_training_mode(True)
        prey_agent.set_training_mode(True)

        for _ in range(rollout_steps):
            if max_time > 0 and (time.time() - t_start) >= max_time:
                break

            pred_obs = {a: obs[a] for a in predators}
            prey_obs = {a: obs[a] for a in prey}
            pred_info = {a: info[a] for a in predators} if info else None
            prey_info = {a: info[a] for a in prey} if info else None

            pred_actions, pred_raw = pred_agent.get_action(obs=pred_obs, infos=pred_info)
            prey_actions, prey_raw = prey_agent.get_action(obs=prey_obs, infos=prey_info)
            actions = {**pred_actions, **prey_actions}

            next_obs, reward, termination, truncation, info = env.step(actions)
            total_steps += num_envs
            pbar.update(num_envs)

            dones_all = build_dones(termination, truncation, all_agent_ids)

            pred_agent.store_transition(
                pred_obs,
                pred_raw,
                {a: reward[a] for a in predators},
                {a: dones_all[a] for a in predators},
            )
            prey_agent.store_transition(
                prey_obs,
                prey_raw,
                {a: reward[a] for a in prey},
                {a: dones_all[a] for a in prey},
            )

            pred_step = team_step_rewards(reward, predators, num_envs)
            prey_step = team_step_rewards(reward, prey, num_envs)
            episode_scores["predator"] += pred_step
            episode_scores["prey"] += prey_step

            for idx in range(num_envs):
                if all(dones_all[aid][idx] for aid in all_agent_ids):
                    completed_scores["predator"].append(
                        float(episode_scores["predator"][idx]) / max(len(predators), 1)
                    )
                    completed_scores["prey"].append(
                        float(episode_scores["prey"][idx]) / max(len(prey), 1)
                    )
                    episode_scores["predator"][idx] = 0.0
                    episode_scores["prey"][idx] = 0.0

            obs = next_obs

            if total_steps >= max_steps:
                break

        pred_agent.learn()
        prey_agent.learn()

        if total_steps - last_eval >= eval_interval:
            pred_ckpt = pool_pred_dir / f"pred_step_{total_steps}.pt"
            prey_ckpt = pool_prey_dir / f"prey_step_{total_steps}.pt"
            pred_agent.save_checkpoint(pred_ckpt.as_posix())
            prey_agent.save_checkpoint(prey_ckpt.as_posix())

            pool_pred = update_pool(
                pool_pred,
                {"id": pred_ckpt.stem, "type": "checkpoint", "path": pred_ckpt.as_posix()},
                args.pool_size,
            )
            pool_prey = update_pool(
                pool_prey,
                {"id": prey_ckpt.stem, "type": "checkpoint", "path": prey_ckpt.as_posix()},
                args.pool_size,
            )

            elo_pred, elo_prey_pool, _, exploit_pred = evaluate_predator_vs_pool(
                pred_agent,
                pool_prey,
                env_kwargs,
                episodes=args.elo_episodes,
                seed=args.seed,
                elo_pred=elo_pred,
                elo_prey_pool=elo_prey_pool,
                k=args.elo_k,
                prey_obs_spaces=prey_obs_spaces,
                prey_action_spaces=prey_action_spaces,
            )
            elo_prey, elo_pred_pool, _, exploit_prey = evaluate_prey_vs_pool(
                prey_agent,
                pool_pred,
                env_kwargs,
                episodes=args.elo_episodes,
                seed=args.seed,
                elo_prey=elo_prey,
                elo_pred_pool=elo_pred_pool,
                k=args.elo_k,
                pred_obs_spaces=pred_obs_spaces,
                pred_action_spaces=pred_action_spaces,
            )

            eval_steps.append(total_steps)
            elo_pred_hist.append(elo_pred)
            elo_prey_hist.append(elo_prey)
            exploit_pred_hist.append(exploit_pred)
            exploit_prey_hist.append(exploit_prey)
            last_eval = total_steps

            pbar.set_postfix(
                {
                    "elo_pred": f"{elo_pred:.1f}",
                    "elo_prey": f"{elo_prey:.1f}",
                    "expl_pred": f"{exploit_pred:.3f}",
                }
            )

        if total_steps - last_matrix >= args.matrix_interval:
            matrix, pred_labels, prey_labels = compute_winrate_matrix(
                pool_pred,
                pool_prey,
                env_kwargs,
                episodes=args.matrix_episodes,
                seed=args.seed,
                pred_obs_spaces=pred_obs_spaces,
                pred_action_spaces=pred_action_spaces,
                prey_obs_spaces=prey_obs_spaces,
                prey_action_spaces=prey_action_spaces,
            )
            matrix_path = Path("checkpoints") / f"{results_key}_winrate_matrix_{total_steps}.png"
            plot_winrate_matrix(matrix, pred_labels, prey_labels, matrix_path.as_posix())
            winrate_matrix_paths.append(matrix_path.as_posix())
            last_matrix = total_steps

    pbar.close()
    elapsed = time.time() - t_start
    env.close()

    print("\nTraining complete!")
    print(f"Total steps: {total_steps}")
    print(f"Wall-clock time: {elapsed:.1f}s")

    # Save latest checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    pred_latest = f"checkpoints/{results_key}_predator_latest.pt"
    prey_latest = f"checkpoints/{results_key}_prey_latest.pt"
    pred_agent.save_checkpoint(pred_latest)
    prey_agent.save_checkpoint(prey_latest)

    # Plot metrics
    plot_path = (args.plot_path or "").strip()
    if not plot_path:
        plot_path = f"{results_key}_training_plot.png"

    if eval_steps:
        plot_training(
            eval_steps,
            elo_pred_hist,
            elo_prey_hist,
            exploit_pred_hist,
            exploit_prey_hist,
            save_path=plot_path,
        )

    # Optional results JSON
    results_path = (args.results_json or "").strip()
    if results_path:
        results = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except Exception:
                results = {}

        results.update(
            {
                "environment": "simple_tag_v3",
                "train_team": "both",
                "mpe2": True,
                "num_envs": args.num_envs,
                "max_cycles": args.max_cycles,
            }
        )
        results[results_key] = {
            "description": "Independent PPO self-play with Elo vs rolling opponent pool",
            "eval_steps": eval_steps,
            "elo_predator": elo_pred_hist,
            "elo_prey": elo_prey_hist,
            "exploitability_predator": exploit_pred_hist,
            "exploitability_prey": exploit_prey_hist,
            "final_elo_predator": round(elo_pred_hist[-1], 2) if elo_pred_hist else 0.0,
            "final_elo_prey": round(elo_prey_hist[-1], 2) if elo_prey_hist else 0.0,
            "final_exploitability_predator": round(exploit_pred_hist[-1], 4)
            if exploit_pred_hist
            else 0.0,
            "final_exploitability_prey": round(exploit_prey_hist[-1], 4)
            if exploit_prey_hist
            else 0.0,
            "training_time_s": round(elapsed, 1),
            "total_env_steps": total_steps,
            "wall_clock_limit_s": round(max_time, 1) if max_time > 0 else 0.0,
            "winrate_matrix_paths": winrate_matrix_paths,
            "plot_path": plot_path,
            "predator_checkpoint": pred_latest,
            "prey_checkpoint": prey_latest,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    if args.render:
        print("\nRendering trained policies...")
        env = make_env(render_mode="human", **env_kwargs)
        pred_agent.set_training_mode(False)
        prey_agent.set_training_mode(False)
        predators, prey = split_teams(env.agents)
        obs, info = env.reset()
        done = False
        while not done:
            pred_obs = {a: obs[a][np.newaxis, :] for a in predators}
            prey_obs = {a: obs[a][np.newaxis, :] for a in prey}
            pred_action, _ = pred_agent.get_action(obs=pred_obs)
            prey_action, _ = prey_agent.get_action(obs=prey_obs)
            actions = {
                **{a: pred_action[a].squeeze() for a in predators},
                **{a: prey_action[a].squeeze() for a in prey},
            }
            obs, reward, termination, truncation, info = env.step(actions)
            done = all(termination.values()) or all(truncation.values())
        env.close()


if __name__ == "__main__":
    main()
