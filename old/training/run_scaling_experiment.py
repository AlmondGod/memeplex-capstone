#!/usr/bin/env python3.9
"""
M-Scaling Experiment for Memeplex
=================================
Runs Memeplex with varying numbers of memes M ∈ {2, 4, 8, 16, 32}
to test how performance scales with meme bank capacity.

Each run: 200k steps, seed=42, terran 5v5.
Results collected into a single JSON + scaling plot.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
M_VALUES      = [2, 4, 8, 16, 32]
TOTAL_STEPS   = 200_000
SEED          = 42
RACE          = "terran"
N_UNITS       = 5
N_ENEMIES     = 5
MEME_DIM      = 16
INFECTION_INT = 50
MUTATION_SIGMA= 0.1
BASE_DIR      = "checkpoints/scaling_M"

PYTHON        = "python3.9"
SCRIPT        = "run_smacv2_memeplex.py"
# ───────────────────────────────────────────────────────────────────────────


def run_single(m: int) -> dict:
    """Run a single Memeplex training with n_memes=m and return results."""
    save_dir = os.path.join(BASE_DIR, f"M{m}_seed{SEED}")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        PYTHON, SCRIPT,
        "--mode", "train",
        "--total-steps", str(TOTAL_STEPS),
        "--n-memes", str(m),
        "--meme-dim", str(MEME_DIM),
        "--infection-interval", str(INFECTION_INT),
        "--mutation-sigma", str(MUTATION_SIGMA),
        "--race", RACE,
        "--n-units", str(N_UNITS),
        "--n-enemies", str(N_ENEMIES),
        "--seed", str(SEED),
        "--save-dir", save_dir,
        "--log-interval", "5",
        "--save-interval", "50",   # save less often to save disk
    ]

    print(f"\n{'='*60}")
    print(f"  M-SCALING: n_memes={m}  (seed={SEED})")
    print(f"  save_dir: {save_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env.setdefault("SC2PATH", "/Applications/StarCraft II")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        capture_output=False,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ⚠ Run M={m} exited with code {proc.returncode}")
        return {"n_memes": m, "error": f"exit code {proc.returncode}"}

    # Load results JSON
    results_path = os.path.join(save_dir, "smacv2_memeplex_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        results["actual_wall_clock"] = elapsed
        print(f"  ✓ M={m}: WR={results['final_win_rate']:.1%}, "
              f"peak={results['peak_rolling20_win_rate']:.1%}, "
              f"time={elapsed:.0f}s")
        return results
    else:
        print(f"  ⚠ No results file found for M={m}")
        return {"n_memes": m, "error": "no results file"}


def plot_scaling(all_results: list, out_dir: str):
    """Create scaling law plots from all results."""
    ms = [r["n_memes"] for r in all_results if "error" not in r]
    peak_wrs = [r["peak_rolling20_win_rate"] for r in all_results if "error" not in r]
    final_wrs = [r["final_win_rate"] for r in all_results if "error" not in r]
    final_rewards = [r["final_mean_reward"] for r in all_results if "error" not in r]
    infections = [r.get("total_infections", 0) for r in all_results if "error" not in r]
    diversities = [r.get("final_meme_diversity", 0) for r in all_results if "error" not in r]
    wall_clocks = [r.get("wall_clock_seconds", 0) for r in all_results if "error" not in r]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Peak Win Rate vs M (log scale)
    ax = axes[0, 0]
    ax.plot(ms, [wr * 100 for wr in peak_wrs], 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Peak Rolling-20 Win Rate (%)')
    ax.set_title('Peak Win Rate vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    # 2. Final Win Rate vs M
    ax = axes[0, 1]
    ax.plot(ms, [wr * 100 for wr in final_wrs], 's-', linewidth=2, markersize=8, color='#4CAF50')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Final Win Rate (%)')
    ax.set_title('Final Win Rate vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    # 3. Final Mean Reward vs M
    ax = axes[0, 2]
    ax.plot(ms, final_rewards, 'D-', linewidth=2, markersize=8, color='#FF9800')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Final Mean Reward')
    ax.set_title('Mean Reward vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    # 4. Total Infections vs M
    ax = axes[1, 0]
    ax.plot(ms, infections, '^-', linewidth=2, markersize=8, color='#F44336')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Total Infections')
    ax.set_title('Infections vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    # 5. Meme Diversity vs M
    ax = axes[1, 1]
    ax.plot(ms, diversities, 'v-', linewidth=2, markersize=8, color='#9C27B0')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Final Meme Diversity')
    ax.set_title('Diversity vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    # 6. Wall-clock time vs M
    ax = axes[1, 2]
    ax.plot(ms, [t / 60 for t in wall_clocks], 'p-', linewidth=2, markersize=8, color='#607D8B')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Memes (M)')
    ax.set_ylabel('Wall-Clock Time (min)')
    ax.set_title('Training Time vs Meme Count')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, alpha=0.3)

    fig.suptitle('Memeplex M-Scaling Laws (200k steps, Terran 5v5, seed=42)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, "m_scaling_results.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nScaling plot saved: {path}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    all_results = []

    t_total = time.time()
    for m in M_VALUES:
        result = run_single(m)
        all_results.append(result)

    elapsed_total = time.time() - t_total

    # Save combined results
    summary = {
        "experiment": "M-scaling",
        "m_values": M_VALUES,
        "total_steps_per_run": TOTAL_STEPS,
        "seed": SEED,
        "total_wall_clock": elapsed_total,
        "results": all_results,
    }
    summary_path = os.path.join(BASE_DIR, "m_scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Plot
    plot_scaling(all_results, BASE_DIR)

    # Print table
    print(f"\n{'='*70}")
    print(f"  M-SCALING RESULTS  ({elapsed_total/60:.1f} min total)")
    print(f"{'='*70}")
    print(f"{'M':>4}  {'Peak WR':>10}  {'Final WR':>10}  {'Reward':>10}  {'Infections':>12}  {'Time (s)':>10}")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['n_memes']:>4}  {'ERROR':>10}")
            continue
        print(f"{r['n_memes']:>4}  "
              f"{r['peak_rolling20_win_rate']*100:>9.1f}%  "
              f"{r['final_win_rate']*100:>9.1f}%  "
              f"{r['final_mean_reward']:>10.2f}  "
              f"{r.get('total_infections', 0):>12}  "
              f"{r.get('wall_clock_seconds', 0):>10.0f}")


if __name__ == "__main__":
    main()
