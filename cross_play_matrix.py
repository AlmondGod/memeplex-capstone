"""Compute cross-play win-rate matrices between MADDPG and Method I.

Produces:
- MADDPG predators vs Method I prey
- Method I predators vs MADDPG prey
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from mpe2 import simple_tag_v3

from agilerl.algorithms.maddpg import MADDPG
from method_i import LatentAlignedIPPO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


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


def list_checkpoints(dir_path: Path, max_per_algo: int) -> list[Path]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.iterdir() if p.suffix == ".pt"]
    def step_key(p: Path) -> int:
        m = re.search(r"_(\d+)$", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=step_key)
    if max_per_algo > 0:
        files = files[-max_per_algo:]
    return files


def load_policy(
    algo: str,
    path: str,
    pred_obs_spaces,
    pred_action_spaces,
    prey_obs_spaces,
    prey_action_spaces,
    team: str,
):
    if algo == "maddpg":
        agent = MADDPG.load(path, device="cpu")
        agent.set_training_mode(False)
        return agent
    if algo == "method_i":
        obs_spaces = pred_obs_spaces if team == "predator" else prey_obs_spaces
        act_spaces = pred_action_spaces if team == "predator" else prey_action_spaces
        agent = LatentAlignedIPPO.load(
            path,
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            device="cpu",
        )
        agent.set_training_mode(False)
        return agent
    raise ValueError(f"Unknown algo: {algo}")


def run_episode(pred_agent, prey_agent, env_kwargs: dict, seed: int | None = None) -> float:
    env = make_env(render_mode=None, **env_kwargs)
    obs, info = env.reset(seed=seed)
    predators, prey = split_teams(env.agents)

    pred_score = 0.0
    prey_score = 0.0
    done = False

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

    if pred_score > prey_score:
        return 1.0
    if pred_score < prey_score:
        return 0.0
    return 0.5


def compute_win_rate(pred_agent, prey_agent, env_kwargs: dict, episodes: int, seed: int) -> float:
    wins = []
    for ep in range(episodes):
        wins.append(run_episode(pred_agent, prey_agent, env_kwargs, seed=seed + ep))
    return float(np.mean(wins)) if wins else 0.0


def plot_matrix(matrix: np.ndarray, pred_labels: list[str], prey_labels: list[str], path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(prey_labels)))
    ax.set_yticks(range(len(pred_labels)))
    ax.set_xticklabels(prey_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pred_labels, fontsize=7)
    ax.set_xlabel("Prey policies")
    ax.set_ylabel("Predator policies")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-set", choices=["step", "wallclock"], default="step")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-per-algo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-json", default="experiment_results.json")
    parser.add_argument("--num-good", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--continuous-actions", action="store_true", default=True)
    args = parser.parse_args()

    env_kwargs = dict(
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous_actions,
        num_good=args.num_good,
        num_adversaries=args.num_adversaries,
        num_obstacles=args.num_obstacles,
    )

    env = make_env(render_mode=None, **env_kwargs)
    env.reset(seed=args.seed)
    predators, prey = split_teams(env.agents)
    pred_obs_spaces = [env.observation_space(a) for a in predators]
    pred_action_spaces = [env.action_space(a) for a in predators]
    prey_obs_spaces = [env.observation_space(a) for a in prey]
    prey_action_spaces = [env.action_space(a) for a in prey]
    env.close()

    base = Path("checkpoints")
    if args.pool_set == "step":
        maddpg_pred_dir = base / "pool_pred"
        maddpg_prey_dir = base / "pool_prey"
        method_pred_dir = base / "pool_pred_method_i"
        method_prey_dir = base / "pool_prey_method_i"
    else:
        maddpg_pred_dir = base / "pool_pred"
        maddpg_prey_dir = base / "pool_prey"
        method_pred_dir = base / "pool_pred_method_i_wall_clock_match"
        method_prey_dir = base / "pool_prey_method_i_wall_clock_match"

    maddpg_pred = list_checkpoints(maddpg_pred_dir, args.max_per_algo)
    maddpg_prey = list_checkpoints(maddpg_prey_dir, args.max_per_algo)
    method_pred = list_checkpoints(method_pred_dir, args.max_per_algo)
    method_prey = list_checkpoints(method_prey_dir, args.max_per_algo)

    # Build caches
    maddpg_pred_agents = [
        load_policy("maddpg", p.as_posix(), pred_obs_spaces, pred_action_spaces, prey_obs_spaces, prey_action_spaces, "predator")
        for p in maddpg_pred
    ]
    maddpg_prey_agents = [
        load_policy("maddpg", p.as_posix(), pred_obs_spaces, pred_action_spaces, prey_obs_spaces, prey_action_spaces, "prey")
        for p in maddpg_prey
    ]
    method_pred_agents = [
        load_policy("method_i", p.as_posix(), pred_obs_spaces, pred_action_spaces, prey_obs_spaces, prey_action_spaces, "predator")
        for p in method_pred
    ]
    method_prey_agents = [
        load_policy("method_i", p.as_posix(), pred_obs_spaces, pred_action_spaces, prey_obs_spaces, prey_action_spaces, "prey")
        for p in method_prey
    ]

    # MADDPG predators vs Method I prey
    matrix_a = np.zeros((len(maddpg_pred_agents), len(method_prey_agents)), dtype=np.float32)
    for i, pred_agent in enumerate(maddpg_pred_agents):
        for j, prey_agent in enumerate(method_prey_agents):
            matrix_a[i, j] = compute_win_rate(pred_agent, prey_agent, env_kwargs, args.episodes, args.seed + i * 1000 + j * 10)

    # Method I predators vs MADDPG prey
    matrix_b = np.zeros((len(method_pred_agents), len(maddpg_prey_agents)), dtype=np.float32)
    for i, pred_agent in enumerate(method_pred_agents):
        for j, prey_agent in enumerate(maddpg_prey_agents):
            matrix_b[i, j] = compute_win_rate(pred_agent, prey_agent, env_kwargs, args.episodes, args.seed + i * 1000 + j * 10)

    out_a = base / f"maddpg_vs_method_i_pred_winrate_{args.pool_set}.png"
    out_b = base / f"method_i_vs_maddpg_pred_winrate_{args.pool_set}.png"

    plot_matrix(
        matrix_a,
        [p.stem for p in maddpg_pred],
        [p.stem for p in method_prey],
        out_a.as_posix(),
        "MADDPG predators vs Method I prey (win rate)",
    )
    plot_matrix(
        matrix_b,
        [p.stem for p in method_pred],
        [p.stem for p in maddpg_prey],
        out_b.as_posix(),
        "Method I predators vs MADDPG prey (win rate)",
    )

    results = {}
    if os.path.exists(args.results_json):
        try:
            with open(args.results_json, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    results[f"cross_play_{args.pool_set}"] = {
        "maddpg_pred_vs_method_i_prey": out_a.as_posix(),
        "method_i_pred_vs_maddpg_prey": out_b.as_posix(),
        "episodes": args.episodes,
        "max_per_algo": args.max_per_algo,
    }

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Cross-play matrices saved to {out_a} and {out_b}")


if __name__ == "__main__":
    main()
