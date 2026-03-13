"""Run Method I v3 (200k steps) and generate updated comparison plot.

This script:
1. Trains Method I v3 predators only for 200k steps (fixed prey policy)
2. Saves eval curve data to experiment_results.json
3. Generates comparison_plot.png with MADDPG vs Method I v3
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3
from tqdm import trange

from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
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


def split_teams(agent_ids):
    predators = [a for a in agent_ids if a.startswith("adversary")]
    prey = [a for a in agent_ids if not a.startswith("adversary")]
    return predators, prey


def build_dones(terminations, truncations, agent_ids):
    dones = {}
    for aid in agent_ids:
        t = terminations.get(aid, True)
        tr = truncations.get(aid, False)
        t = np.where(np.isnan(t), True, t).astype(bool)
        tr = np.where(np.isnan(tr), False, tr).astype(bool)
        dones[aid] = t | tr
    return dones


def team_step_rewards(rewards, team_ids, num_envs):
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


def fixed_policy_actions(env, agent_ids, num_envs, policy: str):
    actions = {}
    for aid in agent_ids:
        space = env.single_action_space(aid)
        if policy == "random":
            actions[aid] = np.stack([space.sample() for _ in range(num_envs)])
        elif policy == "zeros":
            actions[aid] = np.zeros((num_envs, *space.shape), dtype=space.dtype)
        else:
            raise ValueError(f"Unknown fixed policy: {policy}")
    return actions


def evaluate_agent(agent, env_kwargs, episodes, prey_policy, seed=None):
    agent.set_training_mode(False)
    predator_returns = []
    prey_returns = []

    for ep in range(episodes):
        env = make_env(render_mode=None, **env_kwargs)
        obs, info = env.reset(seed=None if seed is None else seed + ep)
        predators, prey = split_teams(env.agents)

        pred_score = 0.0
        prey_score = 0.0
        done = False
        while not done:
            pred_obs = {a: obs[a][np.newaxis, :] for a in predators}
            pred_action, _ = agent.get_action(obs=pred_obs)
            pred_action = {a: pred_action[a].squeeze() for a in predators}
            prey_action = {a: env.action_space(a).sample() for a in prey}
            if prey_policy == "zeros":
                prey_action = {
                    a: np.zeros(env.action_space(a).shape, dtype=env.action_space(a).dtype)
                    for a in prey
                }
            action = {**pred_action, **prey_action}

            obs, reward, termination, truncation, info = env.step(action)
            pred_score += sum(reward[a] for a in predators)
            prey_score += sum(reward[a] for a in prey)
            done = all(termination.values()) or all(truncation.values())

        predator_returns.append(pred_score / max(len(predators), 1))
        prey_returns.append(prey_score / max(len(prey), 1))
        env.close()

    agent.set_training_mode(True)
    return float(np.mean(predator_returns)), float(np.mean(prey_returns))


def _moving_avg(data, window=3):
    if len(data) < window:
        return data
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    smoothed = list(data[:window - 1])
    smoothed += list(cumsum[window - 1:] / window)
    return smoothed


def run_v3_200k(prey_policy: str = "random"):
    """Train Method I v3 for 200k steps and return eval data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_envs = 8
    max_steps = 200_000
    rollout_steps = 128
    eval_interval = 10_000

    env_kwargs = dict(
        max_cycles=25,
        continuous_actions=True,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
    )

    lr = 3e-4
    gamma = 0.95
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    batch_size = 64

    latent_dim = 64
    hidden_dims = [64, 64]
    distill_interval = 10
    distill_weight = 0.1
    distill_lr = 1e-3
    distill_batch_size = 64
    obs_buffer_size = 512

    est_learn_steps = max_steps // (rollout_steps * num_envs)
    lr_anneal = True
    lr_anneal_total_steps = est_learn_steps

    env = AsyncPettingZooVecEnv([lambda: make_env(**env_kwargs) for _ in range(num_envs)])
    env.reset()

    all_agent_ids = list(env.agents)
    predators, prey = split_teams(all_agent_ids)
    train_agent_ids = predators

    observation_spaces = [env.single_observation_space(a) for a in train_agent_ids]
    action_spaces = [env.single_action_space(a) for a in train_agent_ids]

    agent = LatentAlignedIPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=train_agent_ids,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, ent_coef=ent_coef, vf_coef=vf_coef,
        update_epochs=update_epochs, batch_size=batch_size,
        latent_dim=latent_dim, hidden_dims=hidden_dims,
        distill_interval=distill_interval, distill_weight=distill_weight,
        distill_lr=distill_lr, distill_batch_size=distill_batch_size,
        obs_buffer_size=obs_buffer_size,
        distill_anneal=False, distill_anneal_end=0.0, distill_anneal_steps=1,
        lr_anneal=lr_anneal, lr_anneal_total_steps=lr_anneal_total_steps,
        device=str(device),
    )

    print(f"LR annealing: {lr} → 0 over {lr_anneal_total_steps} learn steps")
    print(f"Distill weight: {distill_weight} (constant)")
    print(f"\nStarting Method I v3 training for {max_steps} total steps...")

    total_steps = 0
    last_eval = 0
    eval_steps_list = []
    eval_scores_list = []
    eval_fitnesses_list = []
    pbar = trange(max_steps, desc="Method I v3")
    t_start = time.time()

    obs, info = env.reset()
    episode_scores = {
        "predator": np.zeros(num_envs, dtype=np.float32),
        "prey": np.zeros(num_envs, dtype=np.float32),
    }
    completed_scores = {"predator": [], "prey": []}

    while total_steps < max_steps:
        agent.set_training_mode(True)

        for _ in range(rollout_steps):
            pred_obs = {a: obs[a] for a in predators}
            pred_info = {a: info[a] for a in predators} if info else None
            pred_actions, raw_actions = agent.get_action(obs=pred_obs, infos=pred_info)
            prey_actions = fixed_policy_actions(env, prey, num_envs=num_envs, policy=prey_policy)
            actions = {**pred_actions, **prey_actions}

            next_obs, reward, termination, truncation, info = env.step(actions)
            total_steps += num_envs

            dones_all = build_dones(termination, truncation, all_agent_ids)

            pred_step = team_step_rewards(reward, predators, num_envs)
            prey_step = team_step_rewards(reward, prey, num_envs)
            episode_scores["predator"] += pred_step
            episode_scores["prey"] += prey_step

            agent.store_transition(
                pred_obs,
                raw_actions,
                {a: reward[a] for a in predators},
                {a: dones_all[a] for a in predators},
            )
            obs = next_obs

            for idx in range(num_envs):
                if all(dones_all[aid][idx] for aid in all_agent_ids):
                    pred_ep = float(episode_scores["predator"][idx]) / max(len(predators), 1)
                    prey_ep = float(episode_scores["prey"][idx]) / max(len(prey), 1)
                    completed_scores["predator"].append(pred_ep)
                    completed_scores["prey"].append(prey_ep)
                    agent.scores.append(pred_ep)
                    episode_scores["predator"][idx] = 0.0
                    episode_scores["prey"][idx] = 0.0

        agent.learn()
        pbar.update(rollout_steps * num_envs)

        if total_steps - last_eval >= eval_interval:
            pred_eval, prey_eval = evaluate_agent(
                agent,
                env_kwargs,
                episodes=10,
                prey_policy=prey_policy,
            )

            if completed_scores["predator"]:
                mean_ep = float(np.mean(completed_scores["predator"][-100:]))
            else:
                mean_ep = 0.0

            eval_steps_list.append(total_steps)
            eval_scores_list.append(mean_ep)
            eval_fitnesses_list.append(pred_eval)
            last_eval = total_steps

            pbar.set_postfix({
                "mean_pred": f"{mean_ep:.2f}",
                "pred_eval": f"{pred_eval:.2f}",
                "prey_eval": f"{prey_eval:.2f}",
            })

    pbar.close()
    elapsed = time.time() - t_start
    env.close()

    print(f"\nTraining complete! Total steps: {total_steps}, Time: {elapsed:.1f}s")

    return {
        "eval_steps": eval_steps_list,
        "eval_scores": eval_scores_list,
        "eval_fitnesses": eval_fitnesses_list,
        "training_time_s": round(elapsed, 1),
        "total_env_steps": total_steps,
        "prey_policy": prey_policy,
    }


def run_v3_wallclock(time_budget=446, prey_policy: str = "random"):
    """Train Method I v3 for same wall-clock time as MADDPG."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    num_envs = 8
    max_steps = 2_000_000  # large cap, time-limited
    rollout_steps = 128
    eval_interval = 10_000

    env_kwargs = dict(
        max_cycles=25,
        continuous_actions=True,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
    )

    lr = 3e-4
    gamma = 0.95
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    batch_size = 64

    latent_dim = 64
    hidden_dims = [64, 64]
    distill_interval = 10
    distill_weight = 0.1
    distill_lr = 1e-3
    distill_batch_size = 64
    obs_buffer_size = 512

    # For wallclock, estimate with large max_steps
    est_learn_steps = max_steps // (rollout_steps * num_envs)
    lr_anneal = True
    lr_anneal_total_steps = est_learn_steps

    env = AsyncPettingZooVecEnv([lambda: make_env(**env_kwargs) for _ in range(num_envs)])
    env.reset()

    all_agent_ids = list(env.agents)
    predators, prey = split_teams(all_agent_ids)
    train_agent_ids = predators

    observation_spaces = [env.single_observation_space(a) for a in train_agent_ids]
    action_spaces = [env.single_action_space(a) for a in train_agent_ids]

    agent = LatentAlignedIPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=train_agent_ids,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, ent_coef=ent_coef, vf_coef=vf_coef,
        update_epochs=update_epochs, batch_size=batch_size,
        latent_dim=latent_dim, hidden_dims=hidden_dims,
        distill_interval=distill_interval, distill_weight=distill_weight,
        distill_lr=distill_lr, distill_batch_size=distill_batch_size,
        obs_buffer_size=obs_buffer_size,
        distill_anneal=False, distill_anneal_end=0.0, distill_anneal_steps=1,
        lr_anneal=lr_anneal, lr_anneal_total_steps=lr_anneal_total_steps,
        device=str(device),
    )

    print(f"Starting Method I v3 wall-clock run ({time_budget}s budget)...")

    total_steps = 0
    last_eval = 0
    eval_steps_list = []
    eval_scores_list = []
    eval_fitnesses_list = []
    t_start = time.time()

    obs, info = env.reset()
    episode_scores = {
        "predator": np.zeros(num_envs, dtype=np.float32),
        "prey": np.zeros(num_envs, dtype=np.float32),
    }
    completed_scores = {"predator": [], "prey": []}

    while total_steps < max_steps:
        if (time.time() - t_start) >= time_budget:
            break
        agent.set_training_mode(True)

        for _ in range(rollout_steps):
            pred_obs = {a: obs[a] for a in predators}
            pred_info = {a: info[a] for a in predators} if info else None
            pred_actions, raw_actions = agent.get_action(obs=pred_obs, infos=pred_info)
            prey_actions = fixed_policy_actions(env, prey, num_envs=num_envs, policy=prey_policy)
            actions = {**pred_actions, **prey_actions}

            next_obs, reward, termination, truncation, info = env.step(actions)
            total_steps += num_envs

            dones_all = build_dones(termination, truncation, all_agent_ids)

            pred_step = team_step_rewards(reward, predators, num_envs)
            prey_step = team_step_rewards(reward, prey, num_envs)
            episode_scores["predator"] += pred_step
            episode_scores["prey"] += prey_step

            agent.store_transition(
                pred_obs,
                raw_actions,
                {a: reward[a] for a in predators},
                {a: dones_all[a] for a in predators},
            )
            obs = next_obs

            for idx in range(num_envs):
                if all(dones_all[aid][idx] for aid in all_agent_ids):
                    pred_ep = float(episode_scores["predator"][idx]) / max(len(predators), 1)
                    prey_ep = float(episode_scores["prey"][idx]) / max(len(prey), 1)
                    completed_scores["predator"].append(pred_ep)
                    completed_scores["prey"].append(prey_ep)
                    agent.scores.append(pred_ep)
                    episode_scores["predator"][idx] = 0.0
                    episode_scores["prey"][idx] = 0.0

        agent.learn()

        if total_steps - last_eval >= eval_interval:
            pred_eval, prey_eval = evaluate_agent(
                agent,
                env_kwargs,
                episodes=10,
                prey_policy=prey_policy,
            )

            if completed_scores["predator"]:
                mean_ep = float(np.mean(completed_scores["predator"][-100:]))
            else:
                mean_ep = 0.0

            eval_steps_list.append(total_steps)
            eval_scores_list.append(mean_ep)
            eval_fitnesses_list.append(pred_eval)
            last_eval = total_steps

            if len(eval_steps_list) % 10 == 0:
                print(
                    f"  Steps: {total_steps}, Mean pred: {mean_ep:.2f}, "
                    f"Pred eval: {pred_eval:.2f}, Elapsed: {time.time()-t_start:.0f}s"
                )

    elapsed = time.time() - t_start
    env.close()

    print(f"\nWall-clock run complete! Steps: {total_steps}, Time: {elapsed:.1f}s")

    return {
        "eval_steps": eval_steps_list,
        "eval_scores": eval_scores_list,
        "eval_fitnesses": eval_fitnesses_list,
        "training_time_s": round(elapsed, 1),
        "total_env_steps": total_steps,
        "prey_policy": prey_policy,
    }


def generate_comparison_plot(results_path="experiment_results.json",
                              save_path="comparison_plot.png"):
    """Generate updated comparison plot from experiment data."""
    with open(results_path) as f:
        data = json.load(f)

    maddpg = data["maddpg"]
    mi_v3 = data["method_i_v3_200k"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

    # --- Panel 1: Mean Predator Episode Return (training) ---
    m_steps = np.array(maddpg["eval_steps"]) / 1000
    m_scores = np.array(maddpg["eval_scores"])
    v3_steps = np.array(mi_v3["eval_steps"]) / 1000
    v3_scores = np.array(mi_v3["eval_scores"])

    ax1.plot(m_steps, m_scores, color="tab:blue", alpha=0.35, linewidth=1)
    ax1.plot(m_steps, _moving_avg(m_scores, 3), color="tab:blue", linewidth=2.5,
             label=f"MADDPG (final: {m_scores[-1]:.1f})")
    ax1.plot(v3_steps, v3_scores, color="tab:green", alpha=0.35, linewidth=1)
    ax1.plot(v3_steps, _moving_avg(v3_scores, 3), color="tab:green", linewidth=2.5,
             label=f"Method I v3 (final: {v3_scores[-1]:.1f})")
    ax1.set_ylabel("Mean Predator Episode Return")
    ax1.set_xlabel("Env Steps (×1000)")
    ax1.set_title("MADDPG vs Method I v3 — Mean Predator Return (200k env steps)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Eval Predator Return (200k steps comparison) ---
    m_fit = np.array(maddpg["eval_fitnesses"])
    v3_fit = np.array(mi_v3["eval_fitnesses"])

    ax2.plot(m_steps, m_fit, color="tab:blue", alpha=0.35, linewidth=1)
    ax2.plot(m_steps, _moving_avg(m_fit, 3), color="tab:blue", linewidth=2.5,
             label=f"MADDPG (final: {m_fit[-1]:.1f})")
    ax2.plot(v3_steps, v3_fit, color="tab:green", alpha=0.35, linewidth=1)
    ax2.plot(v3_steps, _moving_avg(v3_fit, 3), color="tab:green", linewidth=2.5,
             label=f"Method I v3 (final: {v3_fit[-1]:.1f})")
    ax2.set_ylabel("Eval Predator Return (fixed prey)")
    ax2.set_xlabel("Env Steps (×1000)")
    ax2.set_title("MADDPG vs Method I v3 — Eval Predator Return (200k env steps)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def generate_wallclock_plot(results_path="experiment_results.json",
                             save_path="wallclock_comparison_plot.png"):
    """Generate wall-clock comparison plot."""
    with open(results_path) as f:
        data = json.load(f)

    maddpg = data["maddpg"]
    mi_v3_wc = data["method_i_v3_wallclock"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # --- Panel 1: Mean Predator Return over env steps ---
    m_steps = np.array(maddpg["eval_steps"]) / 1000
    m_scores = np.array(maddpg["eval_scores"])
    v3_steps = np.array(mi_v3_wc["eval_steps"]) / 1000
    v3_scores = np.array(mi_v3_wc["eval_scores"])

    ax1.plot(m_steps, m_scores, color="tab:blue", alpha=0.35, linewidth=1)
    ax1.plot(m_steps, _moving_avg(m_scores, 3), color="tab:blue", linewidth=2.5,
             label=f"MADDPG ({maddpg['training_time_s']:.0f}s, {maddpg['total_env_steps']/1000:.0f}k steps)")
    ax1.plot(v3_steps, v3_scores, color="tab:green", alpha=0.2, linewidth=1)
    ax1.plot(v3_steps, _moving_avg(v3_scores, 5), color="tab:green", linewidth=2.5,
             label=f"Method I v3 ({mi_v3_wc['training_time_s']:.0f}s, {mi_v3_wc['total_env_steps']/1000:.0f}k steps)")
    ax1.set_ylabel("Mean Predator Episode Return")
    ax1.set_xlabel("Env Steps (×1000)")
    ax1.set_title("Wall-Clock Comparison — Mean Predator Return")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Eval Predator Return ---
    m_fit = np.array(maddpg["eval_fitnesses"])
    v3_fit = np.array(mi_v3_wc["eval_fitnesses"])

    ax2.plot(m_steps, m_fit, color="tab:blue", alpha=0.35, linewidth=1)
    ax2.plot(m_steps, _moving_avg(m_fit, 3), color="tab:blue", linewidth=2.5,
             label="MADDPG")
    ax2.plot(v3_steps, v3_fit, color="tab:green", alpha=0.15, linewidth=1)
    ax2.plot(v3_steps, _moving_avg(v3_fit, 5), color="tab:green", linewidth=2.5,
             label="Method I v3")
    ax2.set_ylabel("Eval Predator Return (fixed prey)")
    ax2.set_xlabel("Env Steps (×1000)")
    ax2.set_title("Wall-Clock Comparison — Eval Predator Return")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Wall-clock comparison plot saved to {save_path}")


if __name__ == "__main__":
    results_path = "experiment_results.json"
    prey_policy = "random"

    # Load existing results to reuse MADDPG timing for wall-clock match
    data = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    time_budget = data.get("maddpg", {}).get("training_time_s", 446)

    # --- Run v3 200k experiment ---
    print("=" * 60)
    print("RUNNING METHOD I v3 — 200k STEPS")
    print("=" * 60)
    v3_200k = run_v3_200k(prey_policy=prey_policy)

    # --- Run v3 wall-clock experiment ---
    print("\n" + "=" * 60)
    print(f"RUNNING METHOD I v3 — WALL-CLOCK ({time_budget:.0f}s)")
    print("=" * 60)
    v3_wc = run_v3_wallclock(time_budget=time_budget, prey_policy=prey_policy)

    # --- Update experiment_results.json ---
    last5_scores = v3_200k["eval_scores"][-5:]
    data.update(
        {
            "environment": "simple_tag_v3",
            "train_team": "predator",
            "prey_policy": prey_policy,
            "num_envs": 8,
        }
    )
    data["method_i_v3_200k"] = {
        "description": "Method I v3 predators only vs fixed prey",
        "eval_steps": v3_200k["eval_steps"],
        "eval_scores": v3_200k["eval_scores"],
        "eval_fitnesses": v3_200k["eval_fitnesses"],
        "final_mean_score": round(v3_200k["eval_scores"][-1], 2) if v3_200k["eval_scores"] else 0,
        "final_eval_fitness": round(v3_200k["eval_fitnesses"][-1], 2) if v3_200k["eval_fitnesses"] else 0,
        "last5_mean_score": round(float(np.mean(last5_scores)), 2) if last5_scores else 0,
        "training_time_s": v3_200k["training_time_s"],
        "total_env_steps": v3_200k["total_env_steps"],
        "prey_policy": v3_200k.get("prey_policy", prey_policy),
    }

    wc_last5 = v3_wc["eval_scores"][-5:]
    data["method_i_v3_wallclock"] = {
        "description": "Method I v3 wall-clock match vs MADDPG",
        "eval_steps": v3_wc["eval_steps"],
        "eval_scores": v3_wc["eval_scores"],
        "eval_fitnesses": v3_wc["eval_fitnesses"],
        "final_mean_score": round(v3_wc["eval_scores"][-1], 2) if v3_wc["eval_scores"] else 0,
        "best_mean_score": round(max(v3_wc["eval_scores"]) if v3_wc["eval_scores"] else 0, 2),
        "last5_mean_score": round(float(np.mean(wc_last5)), 2) if wc_last5 else 0,
        "training_time_s": v3_wc["training_time_s"],
        "total_env_steps": v3_wc["total_env_steps"],
        "prey_policy": v3_wc.get("prey_policy", prey_policy),
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Generate plots ---
    generate_comparison_plot(results_path)
    generate_wallclock_plot(results_path)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    m_last5 = np.mean(data["maddpg"]["eval_scores"][-5:])
    v3_last5 = np.mean(data["method_i_v3_200k"]["eval_scores"][-5:])
    wc_v3_last5 = np.mean(data["method_i_v3_wallclock"]["eval_scores"][-5:])

    print(f"\n200k Step-Matched Comparison (predators only):")
    print(f"  MADDPG        — Last-5 mean: {m_last5:.2f}, Final: {data['maddpg']['eval_scores'][-1]:.2f}")
    print(f"  Method I v3   — Last-5 mean: {v3_last5:.2f}, Final: {data['method_i_v3_200k']['eval_scores'][-1]:.2f}")

    print(f"\nWall-Clock Comparison (~{time_budget:.0f}s):")
    print(f"  MADDPG        — {data['maddpg']['total_env_steps']/1000:.0f}k steps, Last-5 mean: {m_last5:.2f}")
    print(f"  Method I v3   — {data['method_i_v3_wallclock']['total_env_steps']/1000:.0f}k steps, Last-5 mean: {wc_v3_last5:.2f}")
