import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Terran 5v5 setup matches our environment
AGENT_LABELS = [
    "Marine 1",
    "Marine 2",
    "Marauder 1",
    "Marauder 2",
    "Medivac"
]

def load_and_verify(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return None
    data = np.load(path)
    if data.shape != (5, 5):
        print(f"Warning: Expected shape (5, 5), got {data.shape} in {path}")
        return None
    return data

def plot_heatmap(ax, data, title):
    if data is None:
        ax.text(0.5, 0.5, 'Data not available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
        return

    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1.0)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(AGENT_LABELS)), labels=AGENT_LABELS)
    ax.set_yticks(np.arange(len(AGENT_LABELS)), labels=AGENT_LABELS)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(AGENT_LABELS)):
        for j in range(len(AGENT_LABELS)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="w" if data[i, j] < 0.5 else "k")

    ax.set_title(title)
    ax.set_xlabel("Attending To (Key/Value Agent)")
    ax.set_ylabel("Agent (Querying Agent)")

def main():
    parser = argparse.ArgumentParser(description="Plot TarMAC attention heatmaps.")
    parser.add_argument("--pre", type=str, default="pre_eggroll_attention.npy", help="Pre-EGGROLL attention array")
    parser.add_argument("--post", type=str, default="post_eggroll_attention.npy", help="Post-EGGROLL attention array")
    parser.add_argument("--out", type=str, default="attention_comparison.png", help="Output image file")
    
    args = parser.parse_args()

    pre_data = load_and_verify(args.pre)
    post_data = load_and_verify(args.post)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    plot_heatmap(ax1, pre_data, "Pre-EGGROLL TarMAC Attention (Base Checkpoint)")
    plot_heatmap(ax2, post_data, "Post-EGGROLL TarMAC Attention (Fine-Tuned Checkpoint)")

    fig.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved attention heatmap to {args.out}")

if __name__ == '__main__':
    main()
