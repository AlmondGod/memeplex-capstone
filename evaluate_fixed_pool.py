"""Evaluate trained policies vs a shared fixed opponent pool (Elo).

This produces comparable Elo numbers across algorithms by holding opponent
policies and their ratings fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from mpe2 import simple_tag_v3

from agilerl.algorithms.maddpg import MADDPG
from method_i import LatentAlignedIPPO


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
    if algo in {"method_i", "ppo"}:
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


def run_episode(
    pred_agent,
    prey_agent,
    env_kwargs: dict,
    seed: int | None = None,
) -> tuple[float, float, float]:
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

    pred_avg = pred_score / max(len(predators), 1)
    prey_avg = prey_score / max(len(prey), 1)
    if pred_score > prey_score:
        win = 1.0
    elif pred_score < prey_score:
        win = 0.0
    else:
        win = 0.5
    return win, pred_avg, prey_avg


def compute_win_rate(pred_agent, prey_agent, env_kwargs: dict, episodes: int, seed: int) -> float:
    wins = []
    for ep in range(episodes):
        win, _, _ = run_episode(pred_agent, prey_agent, env_kwargs, seed=seed + ep)
        wins.append(win)
    return float(np.mean(wins)) if wins else 0.0


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_candidate_elo(rating: float, opponent_rating: float, score: float, k: float) -> float:
    exp = elo_expected(rating, opponent_rating)
    return rating + k * (score - exp)


def build_pool_entries(pool_set: str, max_per_algo: int) -> tuple[list[dict], list[dict], dict]:
    base = Path("checkpoints")
    if pool_set == "step":
        sources = {
            "maddpg": (base / "pool_pred", base / "pool_prey"),
            "method_i": (base / "pool_pred_method_i", base / "pool_prey_method_i"),
            "ppo": (base / "pool_pred_ppo", base / "pool_prey_ppo"),
        }
    elif pool_set == "wallclock":
        sources = {
            "maddpg": (base / "pool_pred", base / "pool_prey"),
            "method_i": (
                base / "pool_pred_method_i_wall_clock_match",
                base / "pool_prey_method_i_wall_clock_match",
            ),
            "ppo": (
                base / "pool_pred_ppo_wall_clock_match",
                base / "pool_prey_ppo_wall_clock_match",
            ),
        }
    else:
        raise ValueError("pool_set must be 'step' or 'wallclock'")

    pred_pool = [{"id": "random_pred", "type": "random", "algo": "random", "path": None}]
    prey_pool = [{"id": "random_prey", "type": "random", "algo": "random", "path": None}]

    for algo, (pred_dir, prey_dir) in sources.items():
        for p in list_checkpoints(pred_dir, max_per_algo):
            pred_pool.append({
                "id": f"{algo}:{p.stem}",
                "type": "checkpoint",
                "algo": algo,
                "path": p.as_posix(),
            })
        for p in list_checkpoints(prey_dir, max_per_algo):
            prey_pool.append({
                "id": f"{algo}:{p.stem}",
                "type": "checkpoint",
                "algo": algo,
                "path": p.as_posix(),
            })

    return pred_pool, prey_pool, {k: (str(v[0]), str(v[1])) for k, v in sources.items()}


def evaluate_fixed_pool(
    algo_key: str,
    algo_type: str,
    pred_path: str,
    prey_path: str,
    pred_pool: list[dict],
    prey_pool: list[dict],
    env_kwargs: dict,
    episodes: int,
    k: float,
    pred_obs_spaces,
    pred_action_spaces,
    prey_obs_spaces,
    prey_action_spaces,
    seed: int,
):
    pred_agent = load_policy(
        algo_type,
        pred_path,
        pred_obs_spaces,
        pred_action_spaces,
        prey_obs_spaces,
        prey_action_spaces,
        team="predator",
    )
    prey_agent = load_policy(
        algo_type,
        prey_path,
        pred_obs_spaces,
        pred_action_spaces,
        prey_obs_spaces,
        prey_action_spaces,
        team="prey",
    )

    # Candidate Elo vs fixed pool (opponent ratings fixed at 1000)
    pred_rating = 1000.0
    pred_win_rates = []
    for idx, opp in enumerate(prey_pool):
        if opp["type"] == "random":
            opp_agent = None
        else:
            opp_agent = load_policy(
                opp["algo"],
                opp["path"],
                pred_obs_spaces,
                pred_action_spaces,
                prey_obs_spaces,
                prey_action_spaces,
                team="prey",
            )
        win_rate = compute_win_rate(pred_agent, opp_agent, env_kwargs, episodes, seed + idx * 1000)
        pred_win_rates.append(win_rate)
        pred_rating = update_candidate_elo(pred_rating, 1000.0, win_rate, k)

    pred_exploit = max(0.0, 0.5 - min(pred_win_rates)) if pred_win_rates else 0.0

    prey_rating = 1000.0
    prey_win_rates = []
    for idx, opp in enumerate(pred_pool):
        if opp["type"] == "random":
            opp_agent = None
        else:
            opp_agent = load_policy(
                opp["algo"],
                opp["path"],
                pred_obs_spaces,
                pred_action_spaces,
                prey_obs_spaces,
                prey_action_spaces,
                team="predator",
            )
        pred_win = compute_win_rate(opp_agent, prey_agent, env_kwargs, episodes, seed + idx * 1000)
        prey_score = 1.0 - pred_win
        prey_win_rates.append(prey_score)
        prey_rating = update_candidate_elo(prey_rating, 1000.0, prey_score, k)

    prey_exploit = max(0.0, 0.5 - min(prey_win_rates)) if prey_win_rates else 0.0

    return {
        "elo_predator": round(pred_rating, 2),
        "elo_prey": round(prey_rating, 2),
        "exploitability_predator": round(pred_exploit, 4),
        "exploitability_prey": round(prey_exploit, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-set", choices=["step", "wallclock"], default="step")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--elo-k", type=float, default=32.0)
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

    pred_pool, prey_pool, pool_sources = build_pool_entries(args.pool_set, args.max_per_algo)

    if args.pool_set == "step":
        eval_algos = {
            "maddpg": "maddpg",
            "method_i": "method_i",
            "ppo": "ppo",
        }
    else:
        eval_algos = {
            "maddpg": "maddpg",
            "method_i_wall_clock_match": "method_i",
            "ppo_wall_clock_match": "ppo",
        }

    results = {}
    if os.path.exists(args.results_json):
        try:
            with open(args.results_json, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    fixed_key = f"fixed_pool_{args.pool_set}"
    results[fixed_key] = {
        "pool_set": args.pool_set,
        "episodes": args.episodes,
        "elo_k": args.elo_k,
        "max_per_algo": args.max_per_algo,
        "pool_sources": pool_sources,
        "algorithms": {},
    }

    for algo_key, algo_type in eval_algos.items():
        pred_path = f"checkpoints/{algo_key}_predator_latest.pt"
        prey_path = f"checkpoints/{algo_key}_prey_latest.pt"
        if not (Path(pred_path).exists() and Path(prey_path).exists()):
            continue
        metrics = evaluate_fixed_pool(
            algo_key,
            algo_type,
            pred_path,
            prey_path,
            pred_pool,
            prey_pool,
            env_kwargs,
            args.episodes,
            args.elo_k,
            pred_obs_spaces,
            pred_action_spaces,
            prey_obs_spaces,
            prey_action_spaces,
            args.seed,
        )
        results[fixed_key]["algorithms"][algo_key] = metrics

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fixed-pool results saved to {args.results_json} ({fixed_key})")


if __name__ == "__main__":
    main()
