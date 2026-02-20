"""Unified SMACv2 training entry point.

Dispatches to MAPPO or MADDPG based on the --algorithm flag.
All other arguments are passed through to the chosen algorithm.

Usage:
    # MAPPO (default)
    python run_smacv2.py --algorithm mappo --mode train

    # MADDPG
    python run_smacv2.py --algorithm maddpg --mode train

    # Short run
    python run_smacv2.py --algorithm mappo  --mode train --total-steps 200000
    python run_smacv2.py --algorithm maddpg --mode train --total-steps 200000

    # Smoke test
    python run_smacv2.py --algorithm mappo  --mode test
    python run_smacv2.py --algorithm maddpg --mode test

    # Evaluate a checkpoint
    python run_smacv2.py --algorithm mappo  --mode eval --load-path checkpoints/smacv2_mappo_latest.pt
    python run_smacv2.py --algorithm maddpg --mode eval --load-path checkpoints/smacv2_maddpg_latest.pt
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMACv2 MARL training — MAPPO or MADDPG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Algorithm selector ────────────────────────────────────────────────
    parser.add_argument(
        "--algorithm", "-a",
        choices=["mappo", "maddpg"],
        default="mappo",
        help="RL algorithm to run",
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=["train", "eval", "test"], default="train",
        help="Run mode",
    )

    # ── Environment ───────────────────────────────────────────────────────
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--n-units",   type=int, default=5,  help="Number of allied units")
    parser.add_argument("--n-enemies", type=int, default=5,  help="Number of enemy units")

    # ── Shared training hyperparams ───────────────────────────────────────
    parser.add_argument("--total-steps",  type=int,   default=2_000_000)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--cpu",          action="store_true", help="Force CPU")

    # ── Logging / checkpointing ───────────────────────────────────────────
    parser.add_argument("--save-dir",     type=str, default="checkpoints")
    parser.add_argument("--load-path",    type=str, default="")
    parser.add_argument("--eval-episodes",type=int, default=32)
    parser.add_argument("--test-episodes",type=int, default=3)

    # ── MAPPO-specific ────────────────────────────────────────────────────
    mappo = parser.add_argument_group("MAPPO-specific")
    mappo.add_argument("--rollout-steps",    type=int,   default=400)
    mappo.add_argument("--lr",               type=float, default=5e-4)
    mappo.add_argument("--gae-lambda",       type=float, default=0.95)
    mappo.add_argument("--clip-coef",        type=float, default=0.2)
    mappo.add_argument("--ent-coef",         type=float, default=0.01)
    mappo.add_argument("--vf-coef",          type=float, default=0.5)
    mappo.add_argument("--max-grad-norm",    type=float, default=10.0)
    mappo.add_argument("--update-epochs",    type=int,   default=5)
    mappo.add_argument("--num-mini-batches", type=int,   default=1)
    mappo.add_argument("--log-interval",     type=int,   default=10,
                       help="(MAPPO) log every N iterations")
    mappo.add_argument("--save-interval",    type=int,   default=50,
                       help="(MAPPO) checkpoint every N iterations")

    # ── MADDPG-specific ───────────────────────────────────────────────────
    maddpg = parser.add_argument_group("MADDPG-specific")
    maddpg.add_argument("--buffer-size",    type=int,   default=100_000)
    maddpg.add_argument("--batch-size",     type=int,   default=256)
    maddpg.add_argument("--lr-actor",       type=float, default=1e-3)
    maddpg.add_argument("--lr-critic",      type=float, default=1e-3)
    maddpg.add_argument("--tau",            type=float, default=0.01,
                        help="Soft target update coefficient")
    maddpg.add_argument("--epsilon-start",  type=float, default=1.0)
    maddpg.add_argument("--epsilon-end",    type=float, default=0.05)
    maddpg.add_argument("--epsilon-decay",  type=int,   default=50_000)
    maddpg.add_argument("--learn-start",    type=int,   default=5_000)
    maddpg.add_argument("--learn-every",    type=int,   default=10)
    maddpg.add_argument("--maddpg-log-interval",  type=int, default=1_000,
                        help="(MADDPG) log every N env steps")
    maddpg.add_argument("--maddpg-save-interval", type=int, default=50_000,
                        help="(MADDPG) checkpoint every N env steps")

    return parser


def run_mappo(args: argparse.Namespace):
    """Dispatch to the MAPPO run functions in run_smacv2_mappo.py."""
    import run_smacv2_mappo as mappo_mod

    # Patch log/save-interval onto args using MAPPO's naming
    if args.mode == "train":
        mappo_mod.run_train(args)
    elif args.mode == "eval":
        mappo_mod.run_eval(args)
    elif args.mode == "test":
        mappo_mod.run_test(args)


def run_maddpg(args: argparse.Namespace):
    """Dispatch to the MADDPG run functions in run_smacv2_maddpg.py."""
    import run_smacv2_maddpg as maddpg_mod

    # Map unified log/save-interval names to MADDPG's internal names
    if not hasattr(args, "log_interval"):
        args.log_interval = args.maddpg_log_interval
    if not hasattr(args, "save_interval"):
        args.save_interval = args.maddpg_save_interval

    if args.mode == "train":
        maddpg_mod.run_train(args)
    elif args.mode == "eval":
        maddpg_mod.run_eval(args)
    elif args.mode == "test":
        maddpg_mod.run_test(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Normalise dashes→underscores (argparse does this, but be explicit)
    # Also alias MADDPG log/save interval args onto the unified names
    args.log_interval  = getattr(args, "maddpg_log_interval",  args.log_interval  if hasattr(args, "log_interval")  else 1000)
    args.save_interval = getattr(args, "maddpg_save_interval", args.save_interval if hasattr(args, "save_interval") else 50000)

    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Algorithm : {args.algorithm.upper()}")
    print(f"Mode      : {args.mode}")
    print(f"Scenario  : {args.race} {args.n_units}v{args.n_enemies}")
    print(f"Steps     : {args.total_steps:,}")
    print()

    if args.algorithm == "mappo":
        run_mappo(args)
    else:
        run_maddpg(args)


if __name__ == "__main__":
    main()
