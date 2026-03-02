#!/usr/bin/env python3.9
import os
import subprocess
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")

PYTHON = "python3.9"
SCRIPT = "run_smacv2_memeplex.py"
N_AGENTS = 5
MEMES = 8
SEED = 42
TOTAL_STEPS = 200_000

# Base directory for ablations
BASE_DIR = "checkpoints/ablations"

ABLATIONS = [
    {
        "name": "No Infection",
        "dir": "ablate_no_infection",
        "args": ["--infection-interval", "9999999"]
    },
    {
        "name": "No Mutation",
        "dir": "ablate_no_mutation",
        "args": ["--mutation-sigma", "0.0"]
    },
    {
        "name": "No Immunity",
        "dir": "ablate_no_immunity",
        "args": ["--immunity-boost", "0.0"]
    },
    {
        "name": "Random Contagion",
        "dir": "ablate_random_contagion",
        "args": ["--ablate-attention"]
    },
    {
        "name": "Blind Comms",
        "dir": "ablate_blind_comms",
        "args": ["--ablate-meme-context"]
    }
]

def run_ablation(cfg):
    save_dir = os.path.join(BASE_DIR, cfg["dir"])
    results_path = os.path.join(save_dir, "smacv2_memeplex_results.json")

    if os.path.exists(results_path):
        print(f"Skipping {cfg['name']} (already completed).")
        with open(results_path) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        PYTHON, SCRIPT,
        "--mode", "train",
        "--total-steps", str(TOTAL_STEPS),
        "--n-units", str(N_AGENTS),
        "--n-enemies", str(N_AGENTS),
        "--n-memes", str(MEMES),
        "--seed", str(SEED),
        "--save-dir", save_dir,
        "--save-interval", "200",  # Save less often
        "--log-interval", "5",
    ] + cfg["args"]

    print(f"\n============================================================")
    print(f"  RUNNING ABLATION: {cfg['name']}")
    print(f"  Args: {cfg['args']}")
    print(f"============================================================\n")

    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)) or ".", capture_output=False)

    # Clean intermediate checkpoints
    import glob
    for f in glob.glob(os.path.join(save_dir, "*_step_*.pt")):
        try:
            os.remove(f)
        except OSError:
            pass

    if proc.returncode != 0:
        print(f"Error: {cfg['name']} failed with code {proc.returncode}")
        return {"error": f"Failed with code {proc.returncode}", "name": cfg["name"]}

    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
            data["name"] = cfg["name"]
            return data
    else:
        return {"error": "Missing results", "name": cfg["name"]}

def plot_ablations(results_dict, baseline_path):
    print("\nGenerating ablation plots...")
    
    # Load baseline
    baseline_data = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
            baseline_data["name"] = "Baseline (Memeplex M=8)"
    
    # Combine data
    all_runs = []
    if baseline_data:
        all_runs.append(baseline_data)
        
    for cfg in ABLATIONS:
        name = cfg["name"]
        if name in results_dict and "error" not in results_dict[name]:
            data = results_dict[name]
            data["name"] = name
            all_runs.append(data)
            
    if not all_runs:
        print("No successful runs to plot.")
        return
        
    # Plotting Peak WR, Final WR, and Diversity
    names = [r["name"] for r in all_runs]
    peak_wrs = [r.get("peak_rolling20_win_rate", 0) * 100 for r in all_runs]
    final_wrs = [r.get("final_win_rate", 0) * 100 for r in all_runs]
    diversities = [r.get("final_meme_diversity", 0) for r in all_runs]
    entropies = [r.get("final_meme_usage_entropy", 0) for r in all_runs]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Peak WR
    axes[0].barh(names, peak_wrs, color='skyblue')
    axes[0].set_title("Peak Rolling-20 Win Rate (%)")
    axes[0].set_xlim(0, max(peak_wrs) + 5)
    for i, v in enumerate(peak_wrs):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va='center')
        
    # Final WR
    axes[1].barh(names, final_wrs, color='lightgreen')
    axes[1].set_title("Final Win Rate (%)")
    axes[1].set_xlim(0, max(max(final_wrs)+5, 10))
    for i, v in enumerate(final_wrs):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va='center')
        
    # Diversity
    axes[2].barh(names, diversities, color='salmon')
    axes[2].set_title("Final Meme Diversity (0-1)")
    axes[2].set_xlim(0, 1.1)
    for i, v in enumerate(diversities):
        axes[2].text(v + 0.02, i, f"{v:.2f}", va='center')
        
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "ablation_results.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Saved ablation plots to {out_path}")
    
    # Generate summary JSON
    summary = {r["name"]: {
        "peak_wr": r.get("peak_rolling20_win_rate", 0),
        "final_wr": r.get("final_win_rate", 0),
        "mean_reward": r.get("final_mean_reward", 0),
        "diversity": r.get("final_meme_diversity", 0),
        "usage_entropy": r.get("final_meme_usage_entropy", 0),
        "infections": r.get("total_infections", 0)
    } for r in all_runs}
    
    with open(os.path.join(BASE_DIR, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    baseline_path = "checkpoints/scaling_M/M8_seed42/smacv2_memeplex_results.json"
    
    results = {}
    for cfg in ABLATIONS:
        data = run_ablation(cfg)
        results[cfg["name"]] = data
        
    plot_ablations(results, baseline_path)
    
if __name__ == "__main__":
    main()
