#!/usr/bin/env python3.9
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_results(base_dir):
    data = []
    if not os.path.exists(base_dir):
        return data
        
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            json_file = None
            for f in os.listdir(item_path):
                if f.endswith("results.json"):
                    json_file = os.path.join(item_path, f)
                    break
            if json_file:
                try:
                    with open(json_file) as f:
                        data.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    return data

def plot_m_scaling():
    print("Plotting M-scaling...")
    results = load_results("checkpoints/scaling_M")
    if not results:
        print("No M-scaling results found.")
        return
        
    valid_results = [r for r in results if r.get("algorithm") == "Memeplex"]
    valid_results.sort(key=lambda x: x.get("n_memes", 0))
    
    if not valid_results:
        print("No valid Memeplex results found in M-scaling data.")
        return
        
    m_values = [r["n_memes"] for r in valid_results]
    peak_wrs = [r.get("peak_rolling20_win_rate", 0) * 100 for r in valid_results]
    diversities = [r.get("final_meme_diversity", 0) for r in valid_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Memes (M)')
    ax1.set_ylabel('Peak Rolling-20 Win Rate (%)', color=color)
    ax1.plot(m_values, peak_wrs, marker='o', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(peak_wrs) + 5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Final Meme Diversity', color=color)
    ax2.plot(m_values, diversities, marker='s', color=color, linewidth=2, linestyle='--', markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.0, 1.05)
    
    plt.title('Memeplex M-Scaling: Win Rate & Diversity vs Meme Capacity')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("graphs/m_scaling_curve.png", dpi=150)
    plt.close()
    print("Saved graphs/m_scaling_curve.png")

def plot_n_scaling():
    print("Plotting N-scaling...")
    results = load_results("checkpoints/scaling_N")
    if not results:
        print("No N-scaling results found.")
        return
        
    algos = {}
    for r in results:
        algo = r.get("algorithm", "Unknown")
        # Handle naming discrepancies
        if algo == "TarMAC": algo = "TarMAC"
        elif algo == "MAPPO": algo = "MAPPO"
        elif algo == "MADDPG": algo = "MADDPG"
        elif algo == "Memeplex": algo = "Memeplex"
        else: continue
            
        if algo not in algos:
            algos[algo] = []
        algos[algo].append(r)
        
    if not algos:
        print("No valid algorithmic results found for N-scaling.")
        return
        
    plt.figure(figsize=(12, 7))
    markers = {'Memeplex': 'o', 'TarMAC': 's', 'MAPPO': '^', 'MADDPG': 'D'}
    colors = {'Memeplex': 'tab:blue', 'TarMAC': 'tab:green', 'MAPPO': 'tab:orange', 'MADDPG': 'tab:red'}
    
    for algo, data in algos.items():
        # Sort by agent count
        data.sort(key=lambda x: int(x.get("scenario", "0v0").split("v")[0]))
        
        n_values = []
        peak_wrs = []
        for d in data:
            scenario = d.get("scenario", "")
            if "v" in scenario:
                try:
                    n = int(scenario.split("v")[0])
                    n_values.append(n)
                    peak_wrs.append(d.get("peak_rolling20_win_rate", 0) * 100)
                except ValueError:
                    continue
                    
        if n_values:
            plt.plot(n_values, peak_wrs, marker=markers.get(algo, 'x'), color=colors.get(algo, 'k'), 
                     label=algo, linewidth=2.5 if algo == 'Memeplex' else 1.5, markersize=8)
                     
    plt.xlabel('Number of Agents (N)')
    plt.ylabel('Peak Rolling-20 Win Rate (%)')
    plt.title('Algorithm Scaling: Win Rate vs Agent Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("graphs/n_scaling_curve.png", dpi=150)
    plt.close()
    print("Saved graphs/n_scaling_curve.png")

if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    plot_m_scaling()
    plot_n_scaling()
