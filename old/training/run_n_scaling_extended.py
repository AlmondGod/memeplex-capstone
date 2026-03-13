#!/usr/bin/env python3.9
"""
Extended N-Scaling Experiment
=============================
Runs MISSING configurations only:
  - MADDPG at N ∈ {3, 5, 8, 10, 15, 20}
  - Memeplex, TarMAC, MAPPO at N ∈ {15, 20}
  
Then merges with existing N-scaling results and produces
a combined scaling report + plots.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
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
        "color": "#2196F3", "marker": "o",
    },
    "tarmac": {
        "script": "run_smacv2_tarmac.py",
        "extra_args": [],
        "results_file": "smacv2_tarmac_results.json",
        "color": "#4CAF50", "marker": "s",
    },
    "mappo": {
        "script": "run_smacv2_mappo.py",
        "extra_args": [],
        "results_file": "smacv2_mappo_results.json",
        "color": "#FF9800", "marker": "D",
    },
    "maddpg": {
        "script": "run_smacv2_maddpg.py",
        "extra_args": [],
        "results_file": "smacv2_maddpg_results.json",
        "color": "#F44336", "marker": "^",
    },
}

# Runs to execute (only those NOT already completed)
RUNS_TO_DO = [
    ("maddpg", 3), ("maddpg", 5), ("maddpg", 8), ("maddpg", 10),
    ("maddpg", 15), ("maddpg", 20),
    ("memeplex", 15), ("memeplex", 20),
    ("tarmac", 15), ("tarmac", 20),
    ("mappo", 15), ("mappo", 20),
]
# ───────────────────────────────────────────────────────────────────────────


def run_single(algo_name: str, n: int) -> dict:
    algo_cfg = ALGORITHMS[algo_name]
    save_dir = os.path.join(BASE_DIR, f"{algo_name}_N{n}_seed{SEED}")
    
    # Skip if results already exist
    results_path = os.path.join(save_dir, algo_cfg["results_file"])
    if os.path.exists(results_path):
        print(f"  ⏭ {algo_name} N={n}: results already exist, skipping")
        with open(results_path) as f:
            r = json.load(f)
        r["n_agents_config"] = n
        return r

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
        "--save-interval", "200",  # save less often to avoid disk issues
    ] + algo_cfg["extra_args"]

    print(f"\n{'='*60}")
    print(f"  N-SCALING: {algo_name} @ {n}v{n}  (seed={SEED})")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env.setdefault("SC2PATH", "/Applications/StarCraft II")

    t0 = time.time()
    proc = subprocess.run(cmd, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                          capture_output=False)
    elapsed = time.time() - t0

    # Clean up intermediate step checkpoints to save disk, keep latest + results
    import glob
    for f in glob.glob(os.path.join(save_dir, "*_step_*.pt")):
        os.remove(f)

    if proc.returncode != 0:
        print(f"  ⚠ {algo_name} N={n} exited code {proc.returncode}")
        return {"algorithm": algo_name, "n_agents_config": n, "error": f"exit {proc.returncode}"}

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        results["actual_wall_clock"] = elapsed
        results["n_agents_config"] = n
        print(f"  ✓ {algo_name} N={n}: peak={results.get('peak_rolling20_win_rate',0):.1%}, time={elapsed:.0f}s")
        return results
    else:
        return {"algorithm": algo_name, "n_agents_config": n, "error": "no results"}


def load_existing_results() -> dict:
    """Load any existing results from previous N-scaling runs."""
    all_results = {name: {} for name in ALGORITHMS}  # algo -> {N: result}
    
    for algo_name in ALGORITHMS:
        for n in [3, 5, 8, 10, 15, 20]:
            save_dir = os.path.join(BASE_DIR, f"{algo_name}_N{n}_seed{SEED}")
            results_path = os.path.join(save_dir, ALGORITHMS[algo_name]["results_file"])
            if os.path.exists(results_path):
                with open(results_path) as f:
                    r = json.load(f)
                r["n_agents_config"] = n
                all_results[algo_name][n] = r
    return all_results


def plot_scaling(all_results: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for algo_name, algo_cfg in ALGORITHMS.items():
        results_by_n = all_results.get(algo_name, {})
        if not results_by_n:
            continue
        
        ns = sorted(results_by_n.keys())
        valid = [(n, results_by_n[n]) for n in ns if "error" not in results_by_n[n]]
        if not valid:
            continue

        ns_v = [v[0] for v in valid]
        peak_wrs = [v[1].get("peak_rolling20_win_rate", 0) * 100 for v in valid]
        final_wrs = [v[1].get("final_win_rate", 0) * 100 for v in valid]
        wall_clocks = [v[1].get("wall_clock_seconds", 0) / 60 for v in valid]

        axes[0].plot(ns_v, peak_wrs, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.upper())
        axes[1].plot(ns_v, final_wrs, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.upper())
        axes[2].plot(ns_v, wall_clocks, f'{algo_cfg["marker"]}-',
                     color=algo_cfg["color"], linewidth=2, markersize=8,
                     label=algo_name.upper())

    for i, (ylabel, title) in enumerate([
        ("Peak Rolling-20 Win Rate (%)", "Peak Win Rate vs Agent Count"),
        ("Final Win Rate (%)", "Final Win Rate vs Agent Count"),
        ("Training Time (min)", "Wall-Clock Time vs Agent Count"),
    ]):
        axes[i].set_xlabel("Agent Count (N)", fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("N-Scaling: Memeplex vs TarMAC vs MAPPO vs MADDPG (200k steps, Terran NvN)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "n_scaling_extended_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScaling plot saved: {path}")
    return path


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    t_total = time.time()
    for algo_name, n in RUNS_TO_DO:
        run_single(algo_name, n)

    elapsed = time.time() - t_total

    # Load ALL results (existing + new)
    all_results = load_existing_results()

    # Save combined summary
    summary = {
        "experiment": "N-scaling-extended",
        "algorithms": list(ALGORITHMS.keys()),
        "total_steps_per_run": TOTAL_STEPS,
        "seed": SEED,
        "total_wall_clock_new_runs": elapsed,
        "results": {k: {str(n): v for n, v in d.items()} for k, d in all_results.items()},
    }
    with open(os.path.join(BASE_DIR, "n_scaling_extended_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot
    plot_scaling(all_results, BASE_DIR)

    # Print table
    all_ns = sorted(set(n for d in all_results.values() for n in d))
    print(f"\n{'='*90}")
    print(f"  N-SCALING EXTENDED RESULTS  (new runs: {elapsed/60:.1f} min)")
    print(f"{'='*90}")
    print(f"{'Algo':>10} {'N':>3}  {'Peak WR':>10}  {'Final WR':>10}  {'Reward':>10}  {'Time (s)':>10}")
    print("-" * 90)
    for algo_name in ALGORITHMS:
        for n in all_ns:
            r = all_results[algo_name].get(n)
            if r is None:
                continue
            if "error" in r:
                print(f"{algo_name:>10} {n:>3}  {'ERROR':>10}")
                continue
            print(f"{algo_name:>10} {n:>3}  "
                  f"{r.get('peak_rolling20_win_rate',0)*100:>9.1f}%  "
                  f"{r.get('final_win_rate',0)*100:>9.1f}%  "
                  f"{r.get('final_mean_reward',0):>10.2f}  "
                  f"{r.get('wall_clock_seconds',0):>10.0f}")


if __name__ == "__main__":
    main()
