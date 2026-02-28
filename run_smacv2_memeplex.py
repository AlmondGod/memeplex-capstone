"""
run_smacv2_memeplex.py — Meme Epidemiology on SMAC v2
======================================================

Memetic Epidemiology: A novel multi-agent RL algorithm where discrete
behavioural modules ("memes") spread between agents through communication,
mutate on transmission, and are subject to evolutionary selection pressure.

Backbone: TarMAC-style attention communication + PPO (CTDE).

Architecture per step (per agent i, n_agents total):

  1. Encode observation:    z_i = encoder(obs_i)                   [hidden_dim]
  2. Select meme:           w_i = softmax(selector(z_i))           [n_memes]
                            φ_i = Σ_m w_{i,m} · Φ_{i,m}           [meme_dim]
  3. Produce message:
       key_i   = W_k · [z_i ; φ_i]                                [comm_dim]
       value_i = W_v · [z_i ; φ_i]                                [comm_dim]
  4. Produce query:         query_i = W_q · z_i                    [comm_dim]
  5. Targeted aggregation (soft attention, j ≠ i):
       α_{i←j} = softmax_j( query_i · key_j / √d )
       context_i = Σ_{j≠i} α_{i←j} · value_j                     [comm_dim]
  6. Act:  logits_i = actor([z_i ‖ φ_i ‖ context_i])
  7. Value (centralised): V = critic(global_state)

Infection (every K steps):
  For high-attention pairs (α_{i←j} > threshold):
    - Sender's most active meme may "infect" receiver
    - Probability ∝ virality = fitness × novelty - immunity
    - On infection: meme mutates (+ Gaussian noise), replaces least-used slot
    - Receiver gains immune memory against that meme hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Mac: default SC2PATH to Blizzard app location if not set
if not os.environ.get("SC2PATH"):
    _mac_default = "/Applications/StarCraft II"
    if os.path.isdir(_mac_default):
        os.environ["SC2PATH"] = _mac_default

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm


# ===========================================================================
# Environment helpers  (identical to TarMAC/MAPPO scripts)
# ===========================================================================

RACE_MAP_NAMES = {
    "terran":  "10gen_terran",
    "protoss": "10gen_protoss",
    "zerg":    "10gen_zerg",
}

RACE_CONFIGS = {
    "terran": {
        "unit_types": ["marine", "marauder", "medivac"],
        "exception_unit_types": ["medivac"],
        "weights": [0.45, 0.45, 0.1],
    },
    "protoss": {
        "unit_types": ["stalker", "zealot", "colossus"],
        "exception_unit_types": ["colossus"],
        "weights": [0.45, 0.45, 0.1],
    },
    "zerg": {
        "unit_types": ["zergling", "baneling", "hydralisk"],
        "exception_unit_types": ["baneling"],
        "weights": [0.45, 0.1, 0.45],
    },
}


def build_distribution_config(race: str, n_units: int, n_enemies: int) -> dict:
    rc = RACE_CONFIGS[race]
    return {
        "n_units":   n_units,
        "n_enemies": n_enemies,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": rc["unit_types"],
            "exception_unit_types": rc["exception_unit_types"],
            "weights": rc["weights"],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": n_enemies,
            "map_x": 32,
            "map_y": 32,
        },
    }


def make_env(race: str, n_units: int, n_enemies: int, render: bool = False):
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    dist_config = build_distribution_config(race, n_units, n_enemies)
    window_x, window_y = (1280, 720) if render else (640, 480)
    env = StarCraftCapabilityEnvWrapper(
        capability_config=dist_config,
        map_name=RACE_MAP_NAMES[race],
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        window_size_x=window_x,
        window_size_y=window_y,
    )
    return env


# ===========================================================================
# Meme Bank  (vectorized: all N agents stored as (N, M) tensors)
# ===========================================================================

class MemeBankVec:
    """Vectorized meme tracker for ALL agents at once.
    
    Stores usage/fitness as (N, M) tensors for batched updates.
    Immune memory remains per-agent (dict-based, sparse).
    """

    def __init__(self, n_agents: int, n_memes: int, meme_dim: int, device: str = "cpu"):
        self.n_agents = n_agents
        self.n_memes = n_memes
        self.meme_dim = meme_dim
        self.device = device

        self.usage_ema = torch.zeros(n_agents, n_memes, device=device)
        self.fitness_ema = torch.zeros(n_agents, n_memes, device=device)
        self.immune_memory: List[Dict[str, float]] = [{} for _ in range(n_agents)]

    def update_usage(self, sel_weights: torch.Tensor, alpha: float = 0.05):
        """sel_weights: (N, M) softmax distribution per agent."""
        with torch.no_grad():
            self.usage_ema = (1 - alpha) * self.usage_ema + alpha * sel_weights.detach()

    def update_fitness(self, sel_weights: torch.Tensor, reward: float, alpha: float = 0.1):
        """Batch-update fitness EMA for all agents."""
        with torch.no_grad():
            self.fitness_ema = (
                (1 - alpha) * self.fitness_ema
                + alpha * sel_weights.detach() * reward
            )

    def decay_immunity(self, decay: float = 0.99):
        for agent_mem in self.immune_memory:
            expired = [h for h, v in agent_mem.items() if v * decay < 0.01]
            for h in expired:
                del agent_mem[h]
            for h in list(agent_mem.keys()):
                agent_mem[h] *= decay

    def to(self, device):
        self.device = device
        self.usage_ema = self.usage_ema.to(device)
        self.fitness_ema = self.fitness_ema.to(device)
        return self


# ===========================================================================
# Memeplex Actor-Critic Network
# ===========================================================================

class MemeplexActorCritic(nn.Module):
    """
    TarMAC + Meme Bank actor-critic network.

    Components
    ----------
    encoder     : obs_dim  → hidden_dim
    selector    : hidden_dim → n_memes      (attention over meme bank)
    meme_banks  : per-agent bank of n_memes × meme_dim vectors
    key_head    : hidden_dim + meme_dim → comm_dim
    value_head  : hidden_dim + meme_dim → comm_dim
    query_head  : hidden_dim → comm_dim
    actor       : hidden_dim + meme_dim + comm_dim → n_actions
    critic      : state_dim → 1
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 128,
        comm_dim: int = 16,
        meme_dim: int = 16,
        n_memes: int = 8,
        comm_rounds: int = 1,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.comm_dim    = comm_dim
        self.meme_dim    = meme_dim
        self.n_memes     = n_memes
        self.n_agents    = n_agents
        self.comm_rounds = comm_rounds
        self.scale       = math.sqrt(comm_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Meme selector: produces attention weights over meme bank
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_memes),
        )

        # Single stacked meme bank: (N, M, D) — one batched parameter
        self.meme_params = nn.Parameter(torch.randn(n_agents, n_memes, meme_dim) * 0.1)

        # Communication heads — messages include meme context
        self.key_head   = nn.Linear(hidden_dim + meme_dim, comm_dim)
        self.value_head = nn.Linear(hidden_dim + meme_dim, comm_dim)
        self.query_head = nn.Linear(hidden_dim, comm_dim)

        # Actor uses hidden + meme + aggregated context
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + meme_dim + comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Centralised critic (uses global state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

        # Vectorized meme bank tracker (non-differentiable metadata)
        self.meme_bank = MemeBankVec(n_agents, n_memes, meme_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def select_memes(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized meme selection for all agents via batched matmul.

        hidden : (N, hidden_dim) or (B, N, hidden_dim)
        Returns: (active_memes, selection_weights)
        """
        squeeze = hidden.dim() == 2
        if squeeze:
            hidden = hidden.unsqueeze(0)  # (1, N, D)

        B, N, _ = hidden.shape
        logits = self.selector(hidden)           # (B, N, n_memes)
        weights = F.softmax(logits, dim=-1)      # (B, N, n_memes)

        # meme_params: (N, M, D) — expand for batch dim
        banks = self.meme_params.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, M, D)
        w = weights.unsqueeze(-1)                # (B, N, M, 1)
        active_memes = (banks * w).sum(dim=2)    # (B, N, D)

        if squeeze:
            active_memes = active_memes.squeeze(0)
            weights = weights.squeeze(0)

        return active_memes, weights

    def _communicate(self, hidden: torch.Tensor, active_memes: torch.Tensor) -> torch.Tensor:
        """
        Perform one round of targeted communication with meme-enriched messages.
        """
        squeeze = hidden.dim() == 2
        if squeeze:
            hidden = hidden.unsqueeze(0)
            active_memes = active_memes.unsqueeze(0)

        B, N, _ = hidden.shape
        h_meme = torch.cat([hidden, active_memes], dim=-1)  # (B, N, hidden+meme)

        keys    = self.key_head(h_meme)     # (B, N, C)
        values  = self.value_head(h_meme)   # (B, N, C)
        queries = self.query_head(hidden)   # (B, N, C)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        mask = torch.eye(N, device=hidden.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        self.last_attention = attn.detach().clone()

        context = torch.bmm(attn, values)  # (B, N, C)

        if squeeze:
            context = context.squeeze(0)

        return context

    def communicate_and_encode(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (hidden, active_memes, context) after comm_rounds rounds.
        """
        hidden = self.encoder(obs)
        active_memes, sel_weights = self.select_memes(hidden)

        context = torch.zeros(
            *hidden.shape[:-1], self.comm_dim,
            device=obs.device, dtype=obs.dtype
        )
        for _ in range(self.comm_rounds):
            context = self._communicate(hidden, active_memes)

        return hidden, active_memes, context, sel_weights

    def get_actor_output(self, obs: torch.Tensor) -> torch.Tensor:
        hidden, active_memes, context, _ = self.communicate_and_encode(obs)
        return self.actor(torch.cat([hidden, active_memes, context], dim=-1))

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, active_memes, context, _ = self.communicate_and_encode(obs)
        logits = self.actor(torch.cat([hidden, active_memes, context], dim=-1))
        logits = logits.masked_fill(avail_actions == 0, -1e10)
        dist   = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        values    = self.get_value(state)
        return log_probs, entropy, values


# ===========================================================================
# Vectorized Infection Logic
# ===========================================================================

def run_vectorized_infection(
    meme_params: nn.Parameter,     # (N, M, D)
    meme_bank: MemeBankVec,
    attn: torch.Tensor,            # (N, N) attention matrix
    mutation_sigma: float = 0.1,
    immunity_boost: float = 1.0,
    virality_threshold: float = 0.0,
    attn_min: float = 0.1,
) -> int:
    """
    Vectorized infection across all agent pairs.
    Returns number of infections that occurred.
    """
    N, M, D = meme_params.shape
    device = meme_params.device
    infections = 0

    with torch.no_grad():
        # Sender's most active meme index per agent: (N,)
        sender_idxs = meme_bank.usage_ema.argmax(dim=1)       # (N,)
        # Gather sender memes: (N, D)
        sender_memes = meme_params.data[torch.arange(N, device=device), sender_idxs]  # (N, D)
        # Sender fitness for their most active meme: (N,)
        sender_fitness = meme_bank.fitness_ema[torch.arange(N, device=device), sender_idxs]

        # Receiver's least used meme index per agent: (N,)
        receiver_idxs = meme_bank.usage_ema.argmin(dim=1)     # (N,)

        # Novelty: for each (receiver, sender) pair, compute max cosine sim
        # sender_memes: (N, D) -> expand for each receiver's bank
        # meme_params: (N, M, D)
        # We want: for receiver i, sender j: cos_sim(sender_memes[j], meme_params[i])
        sender_norm = F.normalize(sender_memes, dim=-1)       # (N, D)
        bank_norm = F.normalize(meme_params.data, dim=-1)     # (N, M, D)

        # cos_sim[i, j, m] = dot(sender_norm[j], bank_norm[i, m])
        # = (bank_norm[i] @ sender_norm[j].T) for all m
        # bank_norm: (N, M, D), sender_norm: (N, D)
        # Result shape: (N_recv, N_send)
        cos_sim_max = torch.einsum('imd,jd->ij', bank_norm, sender_norm).max(dim=-1).values
        # Wait - that gives (N, N) but einsum 'imd,jd->ij' sums over d, giving (N_i, N_j)
        # but we want max over m. Let me fix:
        # cos_sim[i, j, m] via einsum 'imd,jd->ijm'
        cos_sim_all = torch.einsum('imd,jd->ijm', bank_norm, sender_norm)  # (N, N, M)
        cos_sim_max = cos_sim_all.max(dim=-1).values  # (N, N) — max sim for recv i from sender j
        novelty = 1.0 - cos_sim_max  # (N, N)

        # Fitness term: (N,) -> broadcast to (N, N)
        fitness_term = (sender_fitness.abs() + 0.1).unsqueeze(0).expand(N, -1)  # (N_recv, N_send)

        # Virality: (N, N)
        virality = fitness_term * novelty * attn

        # Subtract immunity (per-pair, sparse — need loop but only over active pairs)
        # Mask: self-attention=0, low attention=0
        mask = (attn > attn_min)
        mask.fill_diagonal_(False)

        # Get candidate pairs
        recv_ids, send_ids = torch.where(mask)

        if len(recv_ids) == 0:
            return 0

        # Process candidates
        for k in range(len(recv_ids)):
            i = recv_ids[k].item()
            j = send_ids[k].item()

            # Get immunity for this specific meme
            s_meme = sender_memes[j]
            s_hash = hashlib.md5(s_meme.cpu().numpy().tobytes()).hexdigest()[:8]
            immunity = meme_bank.immune_memory[i].get(s_hash, 0.0)

            v = virality[i, j].item() - immunity
            if v < virality_threshold:
                continue

            prob = torch.sigmoid(torch.tensor(v)).item()
            if random.random() > prob:
                continue

            # Infection occurs!
            mutated = s_meme + torch.randn_like(s_meme) * mutation_sigma
            replace_idx = receiver_idxs[i].item()
            meme_params.data[i, replace_idx] = mutated
            meme_bank.usage_ema[i, replace_idx] = 0.0
            meme_bank.fitness_ema[i, replace_idx] = 0.0
            meme_bank.immune_memory[i][s_hash] = (
                meme_bank.immune_memory[i].get(s_hash, 0.0) + immunity_boost
            )
            infections += 1

    return infections


# ===========================================================================
# Rollout Buffer (same structure as TarMAC)
# ===========================================================================

class RolloutBuffer:
    def __init__(self):
        self.obs:           List[np.ndarray] = []
        self.states:        List[np.ndarray] = []
        self.actions:       List[np.ndarray] = []
        self.log_probs:     List[np.ndarray] = []
        self.rewards:       List[float]      = []
        self.dones:         List[bool]       = []
        self.avail_actions: List[np.ndarray] = []
        self.values:        List[np.ndarray] = []

    def add(self, obs, state, actions, log_probs, reward, done, avail_actions, values):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.avail_actions.append(avail_actions)
        self.values.append(values)

    def compute_returns(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        T       = len(self.rewards)
        n_agents= self.obs[0].shape[0]
        advantages = np.zeros((T, n_agents), dtype=np.float32)
        last_gae   = np.zeros(n_agents, dtype=np.float32)
        values_arr = np.array(self.values)
        rewards_arr= np.array(self.rewards)
        dones_arr  = np.array(self.dones)

        for t in reversed(range(T)):
            next_val   = last_values if t == T - 1 else values_arr[t + 1]
            mask       = 1.0 - float(dones_arr[t])
            delta      = rewards_arr[t] + gamma * next_val * mask - values_arr[t]
            last_gae   = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr
        return returns, advantages

    def clear(self):
        self.__init__()


# ===========================================================================
# Memeplex Trainer
# ===========================================================================

class MemeplexTrainer:
    def __init__(
        self,
        env,
        device: str = "cpu",
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        update_epochs: int = 4,
        num_mini_batches: int = 4,
        hidden_dim: int = 128,
        comm_dim: int = 16,
        meme_dim: int = 16,
        n_memes: int = 8,
        comm_rounds: int = 1,
        # Infection hyperparams
        infection_interval: int = 50,
        mutation_sigma: float = 0.1,
        immunity_decay: float = 0.99,
        immunity_boost: float = 1.0,
        virality_threshold: float = 0.0,
        fitness_ema_alpha: float = 0.1,
        usage_ema_alpha: float = 0.05,
    ):
        self.env            = env
        self.device         = device
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_coef      = clip_coef
        self.ent_coef       = ent_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
        self.update_epochs  = update_epochs
        self.num_mini_batches = num_mini_batches

        # Infection hyperparams
        self.infection_interval  = infection_interval
        self.mutation_sigma      = mutation_sigma
        self.immunity_decay      = immunity_decay
        self.immunity_boost      = immunity_boost
        self.virality_threshold  = virality_threshold
        self.fitness_ema_alpha   = fitness_ema_alpha
        self.usage_ema_alpha     = usage_ema_alpha

        env_info   = env.get_env_info()
        obs_dim    = env_info["obs_shape"]
        state_dim  = env_info["state_shape"]
        n_actions  = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

        self.policy = MemeplexActorCritic(
            obs_dim    = obs_dim,
            state_dim  = state_dim,
            n_actions  = n_actions,
            n_agents   = self.n_agents,
            hidden_dim = hidden_dim,
            comm_dim   = comm_dim,
            meme_dim   = meme_dim,
            n_memes    = n_memes,
            comm_rounds= comm_rounds,
        ).to(device)

        # Move meme bank device references
        self.policy.meme_bank.to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

        # Infection tracking
        self._step_counter = 0
        self.infection_log: List[Dict] = []

    @torch.no_grad()
    def _step(self):
        """Collect one environment step. Returns (reward, done, info)."""
        obs_list   = self.env.get_obs()
        state      = np.array(self.env.get_state(), dtype=np.float32)
        obs_arr    = np.array(obs_list, dtype=np.float32)
        avail_arr  = np.array(
            [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)],
            dtype=np.float32)

        obs_t   = torch.tensor(obs_arr,   device=self.device)
        avail_t = torch.tensor(avail_arr, device=self.device)
        state_t = torch.tensor(state,     device=self.device)

        hidden, active_memes, context, sel_weights = self.policy.communicate_and_encode(obs_t)

        logits = self.policy.actor(torch.cat([hidden, active_memes, context], dim=-1))
        logits = logits.masked_fill(avail_t == 0, -1e10)
        dist   = Categorical(logits=logits)
        actions= dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.policy.get_value(state_t.unsqueeze(0)).squeeze(0)

        reward, terminated, info = self.env.step(actions.cpu().numpy().tolist())

        # Vectorized usage update for all agents at once
        self.policy.meme_bank.update_usage(sel_weights, self.usage_ema_alpha)

        self.buffer.add(
            obs         = obs_arr,
            state       = state,
            actions     = actions.cpu().numpy(),
            log_probs   = log_probs.cpu().numpy(),
            reward      = reward,
            done        = terminated,
            avail_actions = avail_arr,
            values      = values.cpu().numpy() * np.ones(self.n_agents, dtype=np.float32),
        )

        return reward, terminated, info, sel_weights

    def _run_infection_step(self):
        """Vectorized meme infection between agents based on attention weights."""
        if not hasattr(self.policy, 'last_attention'):
            return 0

        attn = self.policy.last_attention
        if attn.dim() == 3:
            attn = attn.squeeze(0)  # (N, N)

        infections = run_vectorized_infection(
            meme_params=self.policy.meme_params,
            meme_bank=self.policy.meme_bank,
            attn=attn,
            mutation_sigma=self.mutation_sigma,
            immunity_boost=self.immunity_boost,
            virality_threshold=self.virality_threshold,
        )

        self.policy.meme_bank.decay_immunity(self.immunity_decay)
        return infections

    def collect_rollout(self, n_steps: int) -> Tuple[float, float, int]:
        """
        Collect up to n_steps steps (may span multiple episodes).
        Returns (ep_reward, win_rate, total_infections).
        """
        self.buffer.clear()
        self.env.reset()
        ep_reward   = 0.0
        win_count   = 0
        episode_count = 0
        terminated  = False
        total_infections = 0

        for step in range(n_steps):
            reward, terminated, info, sel_weights = self._step()
            ep_reward += reward
            self._step_counter += 1

            # Vectorized fitness update for all agents
            self.policy.meme_bank.update_fitness(
                sel_weights, reward, self.fitness_ema_alpha
            )

            # Infection step
            if self._step_counter % self.infection_interval == 0:
                infections = self._run_infection_step()
                total_infections += infections

            if terminated:
                episode_count += 1
                if info.get("battle_won", False):
                    win_count += 1
                self.env.reset()
                terminated = False

        # Bootstrap value for last state
        with torch.no_grad():
            state    = np.array(self.env.get_state(), dtype=np.float32)
            state_t  = torch.tensor(state, device=self.device)
            last_val = self.policy.get_value(state_t.unsqueeze(0)).squeeze(0)
            last_val_np = last_val.cpu().numpy() * np.ones(self.n_agents, dtype=np.float32)

        self.buffer.compute_returns(last_val_np, self.gamma, self.gae_lambda)
        win_rate = win_count / max(episode_count, 1)
        return ep_reward, win_rate, total_infections

    def update(self) -> Dict[str, float]:
        """One PPO update over the collected rollout."""
        T          = len(self.buffer.obs)
        N          = self.n_agents
        obs        = torch.tensor(np.array(self.buffer.obs),          device=self.device)
        states     = torch.tensor(np.array(self.buffer.states),       device=self.device)
        actions    = torch.tensor(np.array(self.buffer.actions),      device=self.device)
        old_lp     = torch.tensor(np.array(self.buffer.log_probs),    device=self.device)
        avail      = torch.tensor(np.array(self.buffer.avail_actions),device=self.device)

        with torch.no_grad():
            last_state = states[-1]
            last_val   = self.policy.get_value(last_state.unsqueeze(0)).squeeze(0)
            last_val_np = last_val.cpu().numpy() * np.ones(N, dtype=np.float32)
        returns_np, adv_np = self.buffer.compute_returns(last_val_np, self.gamma, self.gae_lambda)

        returns    = torch.tensor(returns_np, device=self.device)
        advantages = torch.tensor(adv_np,     device=self.device)

        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.update_epochs):
            indices = torch.randperm(T)
            mb_size = max(1, T // self.num_mini_batches)

            for start in range(0, T, mb_size):
                mb_idx = indices[start:start + mb_size]
                mb_obs    = obs[mb_idx]
                mb_states = states[mb_idx]
                mb_acts   = actions[mb_idx]
                mb_old_lp = old_lp[mb_idx]
                mb_avail  = avail[mb_idx]
                mb_ret    = returns[mb_idx]
                mb_adv_2d = advantages[mb_idx]

                new_lp, entropy, values = self.policy.evaluate_actions(
                    mb_obs, mb_states, mb_acts, mb_avail
                )
                mb_ret_mean = mb_ret.mean(dim=-1)

                ratio  = torch.exp(new_lp - mb_old_lp)
                mb_adv_2d_norm = mb_adv_2d.reshape(-1)
                mb_adv_2d_norm = (mb_adv_2d_norm - mb_adv_2d_norm.mean()) / (mb_adv_2d_norm.std() + 1e-8)
                mb_adv_2d_norm = mb_adv_2d_norm.view(ratio.shape)

                pg_loss1 = -mb_adv_2d_norm * ratio
                pg_loss2 = -mb_adv_2d_norm * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = F.mse_loss(values, mb_ret_mean)
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(-ent_loss.item())

        return {
            "pg_loss":  float(np.mean(pg_losses)),
            "vf_loss":  float(np.mean(vf_losses)),
            "entropy":  float(np.mean(ent_losses)),
        }

    def get_meme_stats(self) -> Dict[str, float]:
        """Compute diagnostic meme statistics (vectorized)."""
        usage = self.policy.meme_bank.usage_ema.cpu().numpy()       # (N, M)
        fitness = self.policy.meme_bank.fitness_ema.cpu().numpy()   # (N, M)
        meme_vecs = self.policy.meme_params.data.cpu().numpy()     # (N, M, D)

        flat_memes = meme_vecs.reshape(-1, meme_vecs.shape[-1])  # (N*M, D)
        norms = np.linalg.norm(flat_memes, axis=-1, keepdims=True) + 1e-8
        normed = flat_memes / norms
        cos_sim_matrix = normed @ normed.T
        n = cos_sim_matrix.shape[0]
        diversity = 1.0 - (cos_sim_matrix.sum() - n) / (n * (n - 1))

        return {
            "meme_usage_entropy":  float(-np.sum(usage * np.log(usage + 1e-8)) / self.n_agents),
            "meme_mean_fitness":   float(fitness.mean()),
            "meme_max_fitness":    float(fitness.max()),
            "meme_diversity":      float(diversity),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meme_bank": {
                "usage_ema": self.policy.meme_bank.usage_ema.cpu(),
                "fitness_ema": self.policy.meme_bank.fitness_ema.cpu(),
                "immune_memory": [dict(m) for m in self.policy.meme_bank.immune_memory],
            },
        }
        torch.save(save_dict, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "meme_bank" in ckpt:
            bd = ckpt["meme_bank"]
            self.policy.meme_bank.usage_ema = bd["usage_ema"].to(self.device)
            self.policy.meme_bank.fitness_ema = bd["fitness_ema"].to(self.device)
            self.policy.meme_bank.immune_memory = [dict(m) for m in bd["immune_memory"]]
        print(f"Checkpoint loaded: {path}")


# ===========================================================================
# Plotting
# ===========================================================================

def save_plot(rewards, win_rates, infections, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(rewards);      axes[0].set_title("Episode Reward"); axes[0].set_xlabel("Update")
    axes[1].plot(win_rates);    axes[1].set_title("Win Rate");       axes[1].set_xlabel("Update")
    axes[2].plot(infections);   axes[2].set_title("Infections/Rollout"); axes[2].set_xlabel("Update")
    fig.tight_layout()
    path = os.path.join(save_dir, "smacv2_memeplex_training.png")
    fig.savefig(path);  plt.close(fig)
    print(f"Training plot saved: {path}")


# ===========================================================================
# Run modes
# ===========================================================================

def run_train(args):
    """Memeplex training loop with PPO + infection."""
    print(f"=== Memeplex Training: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"  n_memes={args.n_memes}  meme_dim={args.meme_dim}")
    print(f"  infection_interval={args.infection_interval}  mutation_sigma={args.mutation_sigma}")
    print(f"  comm_dim={args.comm_dim}  comm_rounds={args.comm_rounds}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"  device: {device}")

    env = make_env(args.race, args.n_units, args.n_enemies, render=False)
    trainer = MemeplexTrainer(
        env                = env,
        device             = device,
        lr                 = args.lr,
        gamma              = args.gamma,
        gae_lambda         = args.gae_lambda,
        clip_coef          = args.clip_coef,
        ent_coef           = args.ent_coef,
        vf_coef            = args.vf_coef,
        max_grad_norm      = args.max_grad_norm,
        update_epochs      = args.update_epochs,
        num_mini_batches   = args.num_mini_batches,
        hidden_dim         = args.hidden_dim,
        comm_dim           = args.comm_dim,
        meme_dim           = args.meme_dim,
        n_memes            = args.n_memes,
        comm_rounds        = args.comm_rounds,
        infection_interval = args.infection_interval,
        mutation_sigma     = args.mutation_sigma,
        immunity_decay     = args.immunity_decay,
        immunity_boost     = args.immunity_boost,
        virality_threshold = args.virality_threshold,
        fitness_ema_alpha  = args.fitness_ema_alpha,
        usage_ema_alpha    = args.usage_ema_alpha,
    )

    total_params = sum(p.numel() for p in trainer.policy.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,}")

    if args.load_path and os.path.exists(args.load_path):
        trainer.load(args.load_path)

    os.makedirs(args.save_dir, exist_ok=True)
    all_rewards, all_wins, all_infections, all_steps, all_times = [], [], [], [], []
    t_start = time.time()
    steps_done = 0

    pbar = tqdm(total=args.total_steps, desc="Memeplex")
    update_idx = 0

    while steps_done < args.total_steps:
        ep_reward, win_rate, infections = trainer.collect_rollout(args.rollout_steps)
        metrics = trainer.update()
        steps_done += args.rollout_steps
        update_idx += 1
        pbar.update(args.rollout_steps)

        meme_stats = trainer.get_meme_stats()

        all_rewards.append(ep_reward)
        all_wins.append(float(win_rate))
        all_infections.append(infections)
        all_steps.append(steps_done)
        all_times.append(time.time() - t_start)

        if update_idx % args.log_interval == 0:
            rolling_wr = np.mean(all_wins[-20:]) if all_wins else 0.0
            elapsed    = time.time() - t_start
            pbar.write(
                f"Step {steps_done:6d} | reward={ep_reward:7.2f} | "
                f"win_rate={rolling_wr:.2%} | "
                f"infections={infections} | "
                f"diversity={meme_stats['meme_diversity']:.3f} | "
                f"pg_loss={metrics['pg_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"time={elapsed:.0f}s"
            )

        if update_idx % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"smacv2_memeplex_step_{steps_done}.pt")
            trainer.save(path)

    pbar.close()
    trainer.save(os.path.join(args.save_dir, "smacv2_memeplex_latest.pt"))

    elapsed_total = time.time() - t_start
    save_plot(all_rewards, all_wins, all_infections, args.save_dir)

    final_meme_stats = trainer.get_meme_stats()

    results = {
        "algorithm": "Memeplex",
        "race": args.race,
        "scenario": f"{args.n_units}v{args.n_enemies}",
        "total_steps": steps_done,
        "wall_clock_seconds": elapsed_total,
        "rollout_steps": args.rollout_steps,
        "n_memes": args.n_memes,
        "meme_dim": args.meme_dim,
        "infection_interval": args.infection_interval,
        "mutation_sigma": args.mutation_sigma,
        "comm_dim": args.comm_dim,
        "comm_rounds": args.comm_rounds,
        "final_win_rate": float(np.mean(all_wins[-10:])) if all_wins else 0.0,
        "final_mean_reward": float(np.mean(all_rewards[-10:])) if all_rewards else 0.0,
        "peak_rolling20_win_rate": float(max(
            np.mean(all_wins[max(0, i-19):i+1])
            for i in range(len(all_wins))
        )) if all_wins else 0.0,
        "total_infections": sum(all_infections),
        "final_meme_diversity": final_meme_stats["meme_diversity"],
        "final_meme_usage_entropy": final_meme_stats["meme_usage_entropy"],
        "rewards": all_rewards,
        "win_rate_history": all_wins,
        "infection_history": all_infections,
        "steps_history": all_steps,
        "wall_clock_history": all_times,
    }
    with open(os.path.join(args.save_dir, "smacv2_memeplex_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete! {steps_done} steps in {elapsed_total:.1f}s")
    print(f"Final win rate (last 10 updates): {results['final_win_rate']:.1%}")
    print(f"Peak rolling-20 win rate: {results['peak_rolling20_win_rate']:.1%}")
    print(f"Total infections: {results['total_infections']}")
    print(f"Final meme diversity: {results['final_meme_diversity']:.3f}")
    env.close()


def run_eval(args):
    """Evaluate a saved Memeplex checkpoint."""
    if not args.load_path:
        print("Error: --load-path required for eval mode")
        return

    print(f"=== Memeplex Evaluation: {args.race} {args.n_units}v{args.n_enemies} ===")
    print(f"Loading: {args.load_path}")

    device  = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env     = make_env(args.race, args.n_units, args.n_enemies, render=getattr(args, "render", False))

    trainer = MemeplexTrainer(
        env        = env,
        device     = device,
        hidden_dim = args.hidden_dim,
        comm_dim   = args.comm_dim,
        meme_dim   = args.meme_dim,
        n_memes    = args.n_memes,
        comm_rounds= args.comm_rounds,
    )
    trainer.load(args.load_path)
    trainer.policy.eval()

    total_reward, wins = 0.0, 0

    for ep in range(args.eval_episodes):
        env.reset()
        terminated = False
        ep_reward  = 0.0

        while not terminated:
            obs_list = env.get_obs()
            obs_t    = torch.tensor(np.array(obs_list, dtype=np.float32), device=device)
            avail    = np.array([env.get_avail_agent_actions(i) for i in range(trainer.n_agents)],
                                dtype=np.float32)
            avail_t  = torch.tensor(avail, device=device)
            with torch.no_grad():
                hidden, active_memes, context, _ = trainer.policy.communicate_and_encode(obs_t)
                logits = trainer.policy.actor(torch.cat([hidden, active_memes, context], dim=-1))
                logits = logits.masked_fill(avail_t == 0, -1e10)
                actions = logits.argmax(dim=-1)

            reward, terminated, info = env.step(actions.cpu().numpy().tolist())
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


# ===========================================================================
# Argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memeplex (Meme Epidemiology) on SMACv2")

    # Mode
    p.add_argument("--mode", choices=["train", "eval"], default="train")

    # Environment
    p.add_argument("--race",     choices=["terran", "protoss", "zerg"], default="terran")
    p.add_argument("--n-units",  type=int, default=5)
    p.add_argument("--n-enemies",type=int, default=5)

    # Communication
    p.add_argument("--comm-dim",   type=int, default=16)
    p.add_argument("--comm-rounds",type=int, default=1)

    # Meme parameters
    p.add_argument("--n-memes",  type=int, default=8, help="Number of memes per agent")
    p.add_argument("--meme-dim", type=int, default=16, help="Dimensionality of meme vectors")

    # Infection parameters
    p.add_argument("--infection-interval", type=int, default=50,
                   help="Steps between infection attempts")
    p.add_argument("--mutation-sigma", type=float, default=0.1,
                   help="Std of Gaussian noise on meme transmission")
    p.add_argument("--immunity-decay", type=float, default=0.99,
                   help="Per-step decay of immune memory")
    p.add_argument("--immunity-boost", type=float, default=1.0,
                   help="Immunity gained after encountering a meme")
    p.add_argument("--virality-threshold", type=float, default=0.0,
                   help="Minimum virality score for infection")
    p.add_argument("--fitness-ema-alpha", type=float, default=0.1,
                   help="Smoothing factor for per-meme fitness tracking")
    p.add_argument("--usage-ema-alpha", type=float, default=0.05,
                   help="Smoothing factor for meme usage tracking")

    # PPO hyperparams
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae-lambda",      type=float, default=0.95)
    p.add_argument("--clip-coef",       type=float, default=0.2)
    p.add_argument("--ent-coef",        type=float, default=0.01)
    p.add_argument("--vf-coef",         type=float, default=0.5)
    p.add_argument("--max-grad-norm",   type=float, default=10.0)
    p.add_argument("--update-epochs",   type=int,   default=4)
    p.add_argument("--num-mini-batches",type=int,   default=4)

    # Training
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--total-steps",     type=int,   default=200_000)
    p.add_argument("--rollout-steps",   type=int,   default=200)

    # Logging / checkpointing
    p.add_argument("--save-dir",      type=str, default="checkpoints/smacv2_memeplex")
    p.add_argument("--load-path",     type=str, default="")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--log-interval",  type=int, default=5)
    p.add_argument("--save-interval", type=int, default=10)

    # Misc
    p.add_argument("--cpu",    action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed",   type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
