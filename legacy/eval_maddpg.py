"""Evaluate MADDPG predator/prey policies on MPE2 simple_tag_v3.

Usage:
    python eval_maddpg.py
    python eval_maddpg.py --pred-checkpoint checkpoints/maddpg_predator_latest.pt
    python eval_maddpg.py --prey-checkpoint checkpoints/maddpg_prey_latest.pt
    python eval_maddpg.py --episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from mpe2 import simple_tag_v3

from agilerl.algorithms.maddpg import MADDPG


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


def split_teams(agent_ids):
    predators = [a for a in agent_ids if a.startswith("adversary")]
    prey = [a for a in agent_ids if not a.startswith("adversary")]
    return predators, prey


def load_agent(checkpoint_path, device="cpu"):
    if checkpoint_path is None:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_maddpg.py first to generate checkpoints."
        )
    agent = MADDPG.load(checkpoint_path, device=device)
    agent.set_training_mode(False)
    print(f"Loaded agent from {checkpoint_path}")
    return agent


def run_episode(pred_agent, prey_agent, env_kwargs):
    env = make_env(render_mode=None, **env_kwargs)
    obs, info = env.reset()
    predators, prey = split_teams(env.agents)

    pred_score = 0.0
    prey_score = 0.0
    done = False
    steps = 0

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

        obs, reward, termination, truncation, info = env.step(
            {**pred_actions, **prey_actions}
        )
        pred_score += sum(reward[a] for a in predators)
        prey_score += sum(reward[a] for a in prey)
        steps += 1
        done = all(termination.values()) or all(truncation.values())

    env.close()

    pred_score /= max(len(predators), 1)
    prey_score /= max(len(prey), 1)
    return pred_score, prey_score, steps


def load_meta(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG agents")
    parser.add_argument(
        "--pred-checkpoint",
        default="checkpoints/maddpg_predator_latest.pt",
        help="Path to predator checkpoint",
    )
    parser.add_argument(
        "--prey-checkpoint",
        default="checkpoints/maddpg_prey_latest.pt",
        help="Path to prey checkpoint",
    )
    parser.add_argument("--random-pred", action="store_true")
    parser.add_argument("--random-prey", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-good", type=int, default=None)
    parser.add_argument("--num-adversaries", type=int, default=None)
    parser.add_argument("--num-obstacles", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--continuous-actions", action="store_true", default=True)
    args = parser.parse_args()

    meta = load_meta("checkpoints/maddpg_meta.json")

    num_good = args.num_good if args.num_good is not None else (meta.get("num_good") if meta else 1)
    num_adv = args.num_adversaries if args.num_adversaries is not None else (meta.get("num_adversaries") if meta else 3)
    num_obs = args.num_obstacles if args.num_obstacles is not None else (meta.get("num_obstacles") if meta else 2)
    max_cycles = args.max_cycles if args.max_cycles is not None else (meta.get("max_cycles") if meta else 25)

    env_kwargs = dict(
        max_cycles=max_cycles,
        continuous_actions=args.continuous_actions,
        num_good=num_good,
        num_adversaries=num_adv,
        num_obstacles=num_obs,
    )

    pred_agent = None if args.random_pred else load_agent(args.pred_checkpoint, device="cpu")
    prey_agent = None if args.random_prey else load_agent(args.prey_checkpoint, device="cpu")

    predator_scores = []
    prey_scores = []
    for ep in range(args.episodes):
        pred, prey, steps = run_episode(pred_agent, prey_agent, env_kwargs)
        predator_scores.append(pred)
        prey_scores.append(prey)
        print(
            f"  Episode {ep + 1}/{args.episodes}: predator={pred:.2f}, prey={prey:.2f} ({steps} steps)"
        )

    print(
        f"\nMean predator: {np.mean(predator_scores):.2f} +/- {np.std(predator_scores):.2f}"
    )
    print(
        f"Mean prey: {np.mean(prey_scores):.2f} +/- {np.std(prey_scores):.2f}"
    )


if __name__ == "__main__":
    main()
