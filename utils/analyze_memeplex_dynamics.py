import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_smacv2_memeplex import MemeplexActorCritic, make_env

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")

AGENT_LABELS = [
    "Marine 1",
    "Marine 2",
    "Marauder 1",
    "Marauder 2",
    "Medivac"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scaling_M/M8_seed42/smacv2_memeplex_latest.pt")
    args = parser.parse_args()

    # Match training params
    n_agents = 5
    n_enemies = 5
    race = "terran"
    n_memes = 8
    meme_dim = 16
    hidden_dim = 128
    comm_dim = 16

    env = make_env(race, n_agents, n_enemies, render=False)

    env_info = env.get_env_info()
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    n_actions = env_info["n_actions"]

    device = torch.device("cpu")

    policy = MemeplexActorCritic(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        comm_dim=comm_dim,
        meme_dim=meme_dim,
        n_memes=n_memes,
    ).to(device)

    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    all_attentions = []
    all_meme_weights = []

    env.reset()
    terminated = False

    with torch.no_grad():
        while not terminated:
            obs = np.array(env.get_obs(), dtype=np.float32)
            obs_t = torch.tensor(obs, device=device)
            avail = np.array(env.get_avail_actions(), dtype=np.float32)

            # Get hidden encodings
            hidden = policy.encoder(obs_t)
            # Select memes
            active_memes, meme_weights = policy.select_memes(hidden)
            all_meme_weights.append(meme_weights.cpu().numpy())

            # Communicate (populates policy.last_attention)
            context = torch.zeros(n_agents, comm_dim, device=device)
            # 1 round of comms
            h_meme = torch.cat([hidden, active_memes], dim=-1)
            keys = policy.key_head(h_meme).unsqueeze(0)
            values = policy.value_head(h_meme).unsqueeze(0)
            queries = policy.query_head(hidden).unsqueeze(0)

            scores = torch.bmm(queries, keys.transpose(1, 2)) / policy.scale
            mask = torch.eye(n_agents, device=device, dtype=torch.bool).unsqueeze(0)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.nn.functional.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0).squeeze(0)
            all_attentions.append(attn.cpu().numpy())

            # Action selection
            context = torch.bmm(attn.unsqueeze(0), values).squeeze(0)
            logits = policy.actor(torch.cat([hidden, active_memes, context], dim=-1))
            logits = logits.masked_fill(torch.tensor(avail) == 0, -1e10)
            
            # Greedy action for eval
            actions = logits.argmax(dim=-1).cpu().numpy()
            
            reward, terminated, info = env.step(actions)

    print(f"Episode finished after {len(all_attentions)} steps. Reward: {reward}")

    # Plotting
    # 1. Attention Heatmap (mean over episode)
    mean_attn = np.mean(all_attentions, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot Mean Attention
    ax = axes[0]
    im = ax.imshow(mean_attn, cmap='viridis', aspect='auto', vmin=0, vmax=1.0)
    ax.set_xticks(np.arange(len(AGENT_LABELS)), labels=AGENT_LABELS)
    ax.set_yticks(np.arange(len(AGENT_LABELS)), labels=AGENT_LABELS)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(AGENT_LABELS)):
        for j in range(len(AGENT_LABELS)):
            ax.text(j, i, f"{mean_attn[i, j]:.2f}",
                    ha="center", va="center", color="w" if mean_attn[i, j] < 0.5 else "k")
    ax.set_title("Memeplex Mean Communication Attention")
    ax.set_xlabel("Attending To (Sender)")
    ax.set_ylabel("Agent (Receiver)")
    
    # Plot Meme Usage Over Time
    # all_meme_weights shape: (T, N, M)
    meme_weights_arr = np.array(all_meme_weights)
    T, N, M = meme_weights_arr.shape
    
    ax = axes[1]
    # We will plot the dominant meme per agent over time
    dominant_memes = meme_weights_arr.argmax(axis=2) # (T, N)
    
    # Create a nice color map for the M memes
    cmap = plt.get_cmap('tab10', M)
    
    # We can plot a line for each agent
    for n in range(N):
        # Add slight jitter so lines don't perfectly overlap
        y_jitter = n * 0.1
        ax.scatter(np.arange(T), dominant_memes[:, n] + y_jitter, 
                   label=AGENT_LABELS[n], s=20, alpha=0.7)
        
    ax.set_yticks(np.arange(M))
    ax.set_yticklabels([f"Meme {i}" for i in range(M)])
    ax.set_title("Dominant Meme Over Time (Per Agent)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Active Meme Slot")
    ax.legend()
    
    fig.tight_layout()
    out_path = "checkpoints/memeplex_dynamics_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
