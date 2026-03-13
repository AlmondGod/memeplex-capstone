#!/usr/bin/env python3.9
"""
N-Scaling Experiment: Memeplex vs TarMAC vs MAPPO
=================================================
Tests how each algorithm scales with agent count N ∈ {3, 5, 8, 10}.
All runs: 200k steps, terran NvN, seed=42.

This is the "killer experiment" — if Memeplex degrades less than
TarMAC/MAPPO as N grows, that validates the epidemic dynamics thesis.
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
N_VALUES      = [3, 5, 8, 10]
TOTAL_STEPS   = 200_000
SEED          = 42
RACE          = "terran"
BASE_DIR      = "checkpoints/scaling_N"
PYTHON        = "python3.9"

ALGORITHMS = {
    "memeplex": {
        "script": "run_smacv2_memeplex.py",
        "extra_args": ["--n-memes", "8", "--meme-dim", "16",
                       "--infection-interval", "50", "--mutation-sigma", "0.1"],
        "results_file": "smacv2_memeplex_results.json",
        "color": "#2196F3",
        "marker": "o",
    },
    "tarmac": {
        "script": "run_smacv2_tarmac.py",
        "extra_args": [],
        "results_file": "smacv2_tarmac_results.json",
        "color": "#4CAF50",
        "marker": "s",
    },
    "mappo": {
        "script": "run_smacv2_mappo.py",
        "extra_args": [],
        "results_file": "smacv2_mappo_results.json",
        "color": "#FF9800",
        "marker": "D",
    },
}
# ───────────────────────────────────────────────────────────────────────────


def run_single(algo_name: str, algo_cfg: dict, n: int) -> dict:
    """Run a single training at NvN and return results dict."""
    save_dir = os.path.join(BASE_DIR, f"{algo_name}_N{n}_seed{SEED}")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        PYTHON, algo_cfg["script"],
        "--mode", "train",
        "--total-steps", str(TOTAL_STEPS),
        "--race", RACE,
        "--n-units", str(n),
        "--n-enemies", str(n),
        "--seed", str(SEED),
        "--save-dir", save_dir,
        "--log-interval", "5",
        "--save-interval", "50",
    ] + algo_cfg["extra_args"]

    print(f"\n{'='*60}")
    print(f"  N-SCALING: {algo_name} @ {n}v{n} (seed={SEED})")
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
        print(f"  ⚠ {algo_name} N={n} exited with code {proc.returncode}")
        return {"algorithm": algo_name, "n_agents": n, "error": f"exit code {proc.returncode}"}

    results_path = os.path.join(save_dir, algo_cfg["results_file"])
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        results["actual_wall_clock"] = elapsed
        results["n_agents_config"] = n
        print(f"  ✓ {algo_name} N={n}: WR={results.get('final_win_rate', 0):.1%}, "
              f"peak={results.get('peak_rolling20_win_rate', 0):.1%}, "
              f"time={elapsed:.0f}s")
        return results
    else:
        print(f"  ⚠ No results file for {algo_name} N={n}")
        return {"algorithm": algo_name, "n_agents": n, "error": "no results file"}


def plot_scaling(all_results: dict, out_dir: str):
    """Create comparative scaling plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for algo_name, algo_cfg in ALGORITHMS.items():
        results = all_results.get(algo_name, [])
        valid = [r for r in results if "error" not in r]
        if not valid:
            continue

        ns = [r["n_agents_config"] for r in valid]
        peak_wrs = [r.get("peak_rolling20_win_rate", 0) * 100 for r in valid]
        final_wrs = [r.get("final_win_rate", 0) * 100 for r in valid]
        wall_clocks = [r.get("wall_clock_seconds", 0) / 60 for r in valid]

        axes[0].plot(ns, peak_wrs, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.capitalize())
        axes[1].plot(ns, final_wrs, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.capitalize())
        axes[2].plot(ns, wall_clocks, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.capitalize())

    axes[0].set_xlabel("Agent Count (N)")
    axes[0].set_ylabel("Peak Rolling-20 Win Rate (%)")
    axes[0].set_title("Peak Win Rate vs Agent Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Agent Count (N)")
    axes[1].set_ylabel("Final Win Rate (%)")
    axes[1].set_title("Final Win Rate vs Agent Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Agent Count (N)")
    axes[2].set_ylabel("Training Time (min)")
    axes[2].set_title("Wall-Clock Time vs Agent Count")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("N-Scaling: Memeplex vs TarMAC vs MAPPO (200k steps, Terran)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "n_scaling_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScaling plot saved: {path}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    all_results = {name: [] for name in ALGORITHMS}

    t_total = time.time()

    # Run all algorithms at each N, grouped by N for fair comparison
    for n in N_VALUES:
        for algo_name, algo_cfg in ALGORITHMS.items():
            result = run_single(algo_name, algo_cfg, n)
            all_results[algo_name].append(result)

    elapsed_total = time.time() - t_total

    # Save combined summary
    summary = {
        "experiment": "N-scaling",
        "n_values": N_VALUES,
        "algorithms": list(ALGORITHMS.keys()),
        "total_steps_per_run": TOTAL_STEPS,
        "seed": SEED,
        "total_wall_clock": elapsed_total,
        "results": {k: v for k, v in all_results.items()},
    }
    summary_path = os.path.join(BASE_DIR, "n_scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Plot
    plot_scaling(all_results, BASE_DIR)

    # Print table
    print(f"\n{'='*80}")
    print(f"  N-SCALING RESULTS  ({elapsed_total/60:.1f} min total)")
    print(f"{'='*80}")
    print(f"{'Algo':>10} {'N':>3}  {'Peak WR':>10}  {'Final WR':>10}  {'Reward':>10}  {'Time (s)':>10}")
    print("-" * 80)
    for algo_name in ALGORITHMS:
        for r in all_results[algo_name]:
            if "error" in r:
                print(f"{algo_name:>10} {r.get('n_agents', '?'):>3}  {'ERROR':>10}")
                continue
            print(f"{algo_name:>10} {r.get('n_agents_config', '?'):>3}  "
                  f"{r.get('peak_rolling20_win_rate', 0)*100:>9.1f}%  "
                  f"{r.get('final_win_rate', 0)*100:>9.1f}%  "
                  f"{r.get('final_mean_reward', 0):>10.2f}  "
                  f"{r.get('wall_clock_seconds', 0):>10.0f}")


if __name__ == "__main__":
    main()
