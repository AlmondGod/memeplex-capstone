"""
run.py — CLI entry point for Memetic Foundation training and evaluation.

Usage:
    # Full model (memory + communication) training
    python -m new.memetic_foundation.run --mode train

    # Ablation: communication only
    python -m new.memetic_foundation.run --mode train --no-memory

    # Ablation: memory only
    python -m new.memetic_foundation.run --mode train --no-comm

    # Ablation: baseline (no memory, no communication)
    python -m new.memetic_foundation.run --mode train --no-memory --no-comm

    # Smoke test
    python -m new.memetic_foundation.run --mode test

    # Evaluate checkpoint
    python -m new.memetic_foundation.run --mode eval --load-path checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from tqdm import tqdm

from .training.env_utils import make_env
from .training.trainer import MemeticFoundationTrainer, plot_training_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memetic Foundation: Multi-agent architecture with persistent memory and targeted communication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", choices=["train", "eval", "test"], default="train")

    # Environment
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--n-units", type=int, default=5)
    parser.add_argument("--n-enemies", type=int, default=5)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mem-dim", type=int, default=128,
                        help="Memory cell dimension")
    parser.add_argument("--comm-dim", type=int, default=128,
                        help="Communication message dimension")
    parser.add_argument("--n-mem-cells", type=int, default=8,
                        help="Number of persistent memory cells per agent")

    # Ablation flags
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable persistent memory (comm-only or baseline)")
    parser.add_argument("--no-comm", action="store_true",
                        help="Disable communication (memory-only or baseline)")

    # Training hyperparameters
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=1)

    # Logging / checkpointing
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint every N iterations")
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    # Eval / test
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--load-path", type=str, default="")

    # Misc
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true",
                        help="Open SC2 window (eval/test modes)")

    return parser


def get_variant_name(args) -> str:
    use_mem = not args.no_memory
    use_comm = not args.no_comm
    if use_mem and use_comm:
        return "full"
    elif use_comm:
        return "comm_only"
    elif use_mem:
        return "memory_only"
    else:
        return "baseline"


def run_test(args):
    """Quick smoke test with random agents."""
    print(f"=== Smoke Test: {args.race} {args.n_units}v{args.n_enemies} ===")
    env = make_env(args.race, args.n_units, args.n_enemies,
                   render=getattr(args, 'render', False))
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
    """Training loop."""
    variant = get_variant_name(args)
    print(f"=== Memetic Foundation Training ({variant}) ===")
    print(f"  Scenario: {args.race} {args.n_units}v{args.n_enemies}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"  Device: {device}")

    env = make_env(args.race, args.n_units, args.n_enemies, render=False)
    trainer = MemeticFoundationTrainer(
        env=env,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_mini_batches=args.num_mini_batches,
        hidden_dim=args.hidden_dim,
        mem_dim=args.mem_dim,
        comm_dim=args.comm_dim,
        n_mem_cells=args.n_mem_cells,
        use_memory=not args.no_memory,
        use_comm=not args.no_comm,
    )

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)

    total_steps = 0
    log_steps, log_rewards, log_win_rates = [], [], []
    log_pg_losses, log_vf_losses, log_entropies = [], [], []
    t_start = time.time()

    n_iters = args.total_steps // args.rollout_steps
    pbar = tqdm(total=args.total_steps, desc=f"MF-{variant}")

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
            mem_info = ""
            if "mem_norm" in stats:
                mem_info = f" | mem_norm={stats['mem_norm']:.2f}"
            pbar.set_postfix({
                "rew": f"{stats['mean_reward']:.2f}",
                "wr": f"{stats['win_rate']:.2%}",
                "pg": f"{update_stats['pg_loss']:.4f}",
            })
            tqdm.write(
                f"Step {total_steps:>8d} | "
                f"reward={stats['mean_reward']:>7.2f} | "
                f"win_rate={stats['win_rate']:.2%} | "
                f"pg_loss={update_stats['pg_loss']:.4f} | "
                f"vf_loss={update_stats['vf_loss']:.4f} | "
                f"entropy={update_stats['entropy']:.4f}{mem_info} | "
                f"time={elapsed:.0f}s"
            )

        if (iteration + 1) % args.save_interval == 0:
            ckpt = os.path.join(
                args.save_dir, f"memfound_{variant}_step_{total_steps}.pt"
            )
            trainer.save(ckpt)

    pbar.close()
    elapsed = time.time() - t_start

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, f"memfound_{variant}_latest.pt")
    trainer.save(final_path)

    # Plot
    if log_steps:
        plot_path = os.path.join(args.save_dir, f"memfound_{variant}_training.png")
        plot_training_curves(
            log_steps, log_rewards, log_win_rates,
            log_pg_losses, log_vf_losses, log_entropies,
            plot_path, variant_name=variant,
        )

    # Save results JSON
    results = {
        "algorithm": f"memetic_foundation_{variant}",
        "total_steps": total_steps,
        "wall_clock_seconds": elapsed,
        "race": args.race,
        "scenario": f"{args.n_units}v{args.n_enemies}",
        "variant": variant,
        "use_memory": not args.no_memory,
        "use_comm": not args.no_comm,
        "hidden_dim": args.hidden_dim,
        "mem_dim": args.mem_dim,
        "comm_dim": args.comm_dim,
        "n_mem_cells": args.n_mem_cells,
        "final_mean_reward": float(np.mean(log_rewards[-10:])) if log_rewards else 0.0,
        "final_win_rate": float(np.mean(log_win_rates[-10:])) if log_win_rates else 0.0,
        "final_pg_loss": float(np.mean(log_pg_losses[-10:])) if log_pg_losses else 0.0,
        "final_vf_loss": float(np.mean(log_vf_losses[-10:])) if log_vf_losses else 0.0,
        "final_entropy": float(np.mean(log_entropies[-10:])) if log_entropies else 0.0,
        "rewards_history": log_rewards,
        "win_rate_history": log_win_rates,
        "steps_history": log_steps,
    }
    results_path = os.path.join(args.save_dir, f"memfound_{variant}_results.json")
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

    variant = get_variant_name(args)
    print(f"=== Memetic Foundation Evaluation ({variant}) ===")
    print(f"Loading: {args.load_path}")
    render = getattr(args, 'render', False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_env(args.race, args.n_units, args.n_enemies, render=render)
    trainer = MemeticFoundationTrainer(
        env=env,
        device=device,
        hidden_dim=args.hidden_dim,
        mem_dim=args.mem_dim,
        comm_dim=args.comm_dim,
        n_mem_cells=args.n_mem_cells,
        use_memory=not args.no_memory,
        use_comm=not args.no_comm,
    )
    trainer.load(args.load_path)
    trainer.policy.eval()

    # Reset memory to learned initial state for clean eval
    trainer.policy.reset_memory()

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
            avail_arr = np.zeros(
                (env_info["n_agents"], env_info["n_actions"]), dtype=np.float32
            )
            for aid in range(env_info["n_agents"]):
                avail_arr[aid] = np.array(
                    env.get_avail_agent_actions(aid), dtype=np.float32
                )

            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, device=trainer.device)
                avail_t = torch.tensor(avail_arr, device=trainer.device)

                step_out = trainer.policy.forward_step(obs_t, avail_t)
                # Use greedy actions for eval
                actions = step_out["logits"].argmax(dim=-1).cpu().numpy()

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


def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create save dir
    if args.mode == "train":
        variant = get_variant_name(args)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(args.save_dir, f"memfound_{variant}_{timestamp}")
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Save dir: {args.save_dir}")

    if args.mode == "test":
        run_test(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
