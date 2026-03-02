# Memeplex: Meme Epidemiology for Multi-Agent Reinforcement Learning

## Overview

**Memeplex** is a novel multi-agent reinforcement learning (MARL) algorithm that introduces **memetic epidemiology** — the idea that learned behavioral strategies ("memes") can spread between cooperating agents like cultural contagions. Instead of relying solely on gradient-based learning, agents evolve a shared strategy repertoire through communication-driven transmission, mutation, and immune selection.

The core thesis: **the communication channel IS the evolutionary pressure mechanism**. Agents don't just share information — they *infect* each other with strategies, creating emergent epidemic dynamics that balance exploration and exploitation.

---

## Theoretical Motivation & Design Choices

Memeplex was designed to fuse the rigorous, gradient-based sample efficiency of Deep Reinforcement Learning with the open-ended, divergent search capabilities of Evolutionary Algorithms. It draws inspiration from several fields:

### 1. The Bottleneck of Centralised MARL (Inspired by MAPPO & MADDPG)
Standard MARL algorithms like MAPPO and MADDPG treat agents as entirely separate computational entities during execution, only sharing information through a centralised critic during training.
* **The Problem:** When an agent discovers a novel, highly-effective strategy (e.g., a specific kiting maneuver), there is no mechanism for other agents to *learn* that exact strategy other than slowly stumbling upon it themselves via environmental reward gradients. 
* **Memeplex Solution:** Treat strategies not just as neural network weights, but as discrete, transmittable "memes" (vectors in $\mathbb{R}^d$) that can be directly copied from one agent's brain to another's memory bank.

### 2. The Power of Gated Communication (Inspired by TarMAC)
TarMAC introduced the idea that agents should use attention mechanisms to decide *who* to listen to, rather than broadcasting uniformly to everyone.
* **The Problem:** In TarMAC, communication is just a transient exchange of state observations. It helps agents coordinate *this* timestep, but it doesn't fundamentally change the receiving agent's underlying behavior policy permanently.
* **Memeplex Solution:** Repurpose the TarMAC communication channel as an **evolutionary transmission vector**. Attention ($\alpha_{i \leftarrow j}$) doesn't just gate immediate coordination; it gates the probability of long-term strategic infection. If you pay high attention to a peer, you are structurally more likely to adopt their underlying strategy.

### 3. Cultural Evolution & Quality-Diversity (Inspired by EGGROLL)
Algorithms like EGGROLL rely entirely on evolutionary pseudo-gradients, maintaining massive populations of agents and selecting over them.
* **The Problem:** Pure evolutionary algorithms are notoriously sample-inefficient and struggle with the dense, continuous control required in environments like SMACv2. Conversely, pure RL is sample efficient but struggles with exploration and often collapses into local optima.
* **Memeplex Solution:** Create a "population within a population." The agents themselves learn how to map states to actions via highly sample-efficient PPO, but the *strategic modules* (memes) they condition those actions upon form a parallel evolutionary ecosystem. 

### Why the Specific Components?
* **Meme Context in Messages:** Ablations proved that if agents communicate blindly, infection accuracy degrades. By embedding the active meme $\phi_i$ into the TarMAC Key/Value heads, we ensure agents route their attention based on *what strategy* their peers are currently executing.
* **Immune Memory:** In early iterations, agents would simply swap their best-performing memes back and forth infinitely, destroying diversity. Cryptographic immune memory (hash-based prevention of immediate re-infection) prevents this biological "echo chamber," forcing the population to constantly explore new variations.
* **Mutation Annealing:** Perfect copying (zero mutation) stabilizes convergence but harms mean reward by collapsing exploration. High mutation drives robust exploration but prevents deep exploitation. Therefore, Memeplex employs linear annealing of the mutation rate to capture both benefits.

---

## Algorithm Design

### Formal Definition

Each agent $i$ maintains:
- A **meme bank** $\Phi_i = \{\phi_{i,1}, \dots, \phi_{i,M}\}$ — $M$ learnable strategy vectors in $\mathbb{R}^{d}$
- **Usage tracking**: EMA of selection weights per meme slot
- **Fitness tracking**: EMA of reward-weighted selection per meme slot
- **Immune memory**: sparse hash map recording previously encountered memes

### Architecture (per timestep, per agent)

```
1. Encode observation:     z_i = encoder(obs_i)                    [hidden_dim]
2. Select meme (soft):     w_i = softmax(selector(z_i))            [n_memes]
                           φ_i = Σ_m w_{i,m} · Φ_{i,m}            [meme_dim]
3. Produce message:
     key_i   = W_k · [z_i ; φ_i]                                  [comm_dim]
     value_i = W_v · [z_i ; φ_i]                                  [comm_dim]
4. Produce query:          query_i = W_q · z_i                     [comm_dim]
5. Targeted aggregation (TarMAC-style, soft attention):
     α_{i←j} = softmax_j( query_i · key_j / √d )
     context_i = Σ_{j≠i} α_{i←j} · value_j                       [comm_dim]
6. Act:  logits_i = actor([z_i ‖ φ_i ‖ context_i])
7. Value (centralised):  V = critic(global_state)
```

Messages embed meme context (`[z_i ; φ_i]`), meaning that **what an agent communicates is shaped by which strategy it's currently using**. This couples the communication and evolution dynamics.

### Epidemiology Mechanics (every K steps)

```
For each pair (receiver_i, sender_j) where α_{i←j} > threshold:
  1. Sender selects most-active meme: φ_j* = argmax(usage_ema_j)
  2. Compute novelty:  n = 1 - max_m cos_sim(φ_j*, Φ_{i,m})
  3. Compute virality:  v = (|fitness(φ_j*)| + ε) × novelty × α_{i←j}
  4. Subtract immunity:  v' = v - immunity_i(hash(φ_j*))
  5. Infection probability:  p = σ(v')
  6. If Bernoulli(p) = 1 (infection occurs):
     a. Mutate:  φ_new = φ_j* + ε,  ε ~ N(0, σ²I)
     b. Replace least-used meme in Φ_i with φ_new
     c. Update immune memory:  immunity_i(hash(φ_j*)) += δ
     d. Reset usage/fitness for the replaced slot
```

### Key Properties

| Property | Mechanism |
|----------|-----------|
| **Exploration** | Meme mutation on transmission introduces novel strategies |
| **Exploitation** | High-fitness memes spread faster (virality ∝ fitness) |
| **Self-regulation** | Immune memory prevents re-infection by similar memes |
| **Communication coupling** | Attention weights from TarMAC gate infection probability |
| **No extra loss** | Epidemic dynamics are non-differentiable; meme vectors still receive PPO gradients through the selector |

---

## Training

Memeplex uses **PPO (Proximal Policy Optimization)** with centralised training, decentralised execution (CTDE):

- **Encoder, selector, actor, communication heads** — all trained via PPO gradients
- **Meme vectors** ($\Phi_i$) — trained via PPO (they're `nn.Parameter` in the computation graph through the soft selector) AND evolved via epidemic dynamics (infection replaces memes non-differentiably)
- **Critic** — centralised, uses global state

This creates a dual optimization: gradient descent optimizes meme *usage*, while epidemic dynamics optimizes meme *content* through population-level selection.

---

## Comparison Algorithms

| Algorithm | Communication | Optimization | Key Feature |
|-----------|--------------|-------------|-------------|
| **MAPPO** | None (independent actors, shared critic) | PPO | Baseline |
| **TarMAC** | Targeted attention (K-Q-V) | PPO | Communication mechanism |
| **MADDPG** | None | DDPG + centralised critic | Off-policy, continuous actions |
| **EGGROLL** | None | Evolution (pseudo-gradients) | Gradient-free |
| **Memeplex** | TarMAC + meme context | PPO + epidemic evolution | Communication-driven strategy evolution |

---

## Metrics of Comparison

### Primary Performance Metrics

#### 1. Peak Rolling-20 Win Rate (%)
The maximum win rate achieved over any window of 20 consecutive evaluation episodes during training. This captures the **best performance** the algorithm achieves, filtering out late-training instability.

- **Why this metric**: MARL training is noisy. Final win rate can be misleading due to late-stage oscillations. Peak rolling-20 captures whether the algorithm *ever* learned good coordination.
- **Computed as**: `max over all t of: mean(win_rate[t-19:t+1])`

#### 2. Final Win Rate (%)
The win rate at the end of training (last 10 evaluation windows). Captures **stability** — whether the algorithm maintains its performance.

- **Why this metric**: An algorithm that peaks high but crashes shows instability.
- **Note**: In Memeplex, epidemic dynamics can cause exploration-exploitation oscillations that lower final WR even when the algorithm has learned strong strategies.

#### 3. Final Mean Reward
The average reward at the end of training. Unlike win rate (binary), reward captures **partial success** (e.g., damage dealt, units survived).

### Scaling Metrics

#### 4. Peak Win Rate vs Agent Count (N-Scaling)
How does each algorithm's peak performance degrade as the team grows from 3 to 20 agents? The **scaling exponent** (slope on log-log plot) indicates how well coordination scales.

- **Memeplex thesis**: Epidemic dynamics should help MORE with larger populations (more agents = richer meme ecology).
- **Measured at**: N ∈ {3, 5, 8, 10, 15, 20} agents per team.

#### 5. Peak Win Rate vs Meme Count (M-Scaling)
How does Memeplex performance scale with meme bank capacity?

- **Analogous to**: "number of parameters" in LLM scaling laws.
- **Measured at**: M ∈ {2, 4, 8, 16, 32} memes per agent.

### Efficiency Metrics

#### 6. Wall-Clock Time (seconds)
Total training time for a fixed 200k-step budget.

- **Why this metric**: Algorithms that take 10× longer per step aren't practical regardless of final performance.
- **After vectorization**: Memeplex achieves ~100 it/s, competitive with TarMAC (~91 it/s) and MAPPO (~95 it/s).

#### 7. Throughput (iterations/second)
Environment steps processed per second.

### Memeplex-Specific Diagnostics

#### 8. Meme Diversity
Average pairwise cosine distance across all memes in all agents' banks:
```
diversity = 1 - (Σ_{i≠j} cos_sim(φ_i, φ_j)) / (N*M * (N*M - 1))
```
- **High diversity** (>0.9): Memes represent genuinely different strategies
- **Low diversity** (<0.5): Memes have collapsed to similar vectors (bad)

#### 9. Infection Rate
Number of successful meme infections per rollout.
- **Self-regulating**: ~14-16 infections per rollout across all M values
- **Too high**: Memes change too fast for PPO to optimize
- **Too low**: No evolutionary pressure

#### 10. Meme Usage Entropy
Entropy of the meme selection distribution, averaged across agents:
```
H = -Σ_m usage_ema_m × log(usage_ema_m)
```
- **High entropy**: Agent uniformly uses all memes (exploration)
- **Low entropy**: Agent has converged to a few dominant memes (exploitation)

#### 11. Meme Fitness
EMA of reward weighted by meme selection:
```
fitness_m(t) = (1-α) × fitness_m(t-1) + α × w_m × reward
```
Tracks which memes are associated with high-reward episodes.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Environment | SMACv2 (StarCraft Multi-Agent Challenge v2) |
| Race | Terran |
| Default team size | 5v5 |
| Training steps | 200,000 (fixed budget) |
| Optimizer | Adam (lr=5e-4, ε=1e-5) |
| PPO epochs | 4 per rollout |
| PPO clip | 0.2 |
| Rollout length | 400 steps |
| Discount (γ) | 0.99 |
| GAE (λ) | 0.95 |
| Entropy coefficient | 0.01 |
| Seed | 42 (single seed for scaling experiments) |

### Memeplex-Specific Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_memes` (M) | 8 | Memes per agent |
| `meme_dim` | 16 | Dimensionality of meme vectors |
| `infection_interval` | 50 | Steps between infection attempts |
| `mutation_sigma` | 0.1 | Std dev of Gaussian mutation on transmission |
| `immunity_boost` | 1.0 | Immunity score added on infection |
| `immunity_decay` | 0.99 | Per-step decay of immune memory |
| `virality_threshold` | 0.0 | Minimum virality for infection attempt |
| `comm_dim` | 16 | Communication embedding dimension |
| `comm_rounds` | 1 | TarMAC communication rounds |

---

## Implementation Details

### Vectorization

Three optimizations reduced Memeplex's wall-clock time by **4.1×**:

1. **Single `nn.Parameter(N, M, D)`** — consolidated from N separate `nn.ParameterList` entries, enabling batched optimizer updates
2. **Batched meme selection** — replaced per-agent Python loop with `(banks * weights.unsqueeze(-1)).sum(dim=2)` for a single batched operation
3. **Vectorized infection** — `torch.einsum('imd,jd->ijm', ...)` computes all pairwise novelty scores in one operation; tensor masking identifies candidate infection pairs

### File Structure

| File | Description |
|------|-------------|
| `run_smacv2_memeplex.py` | Full Memeplex implementation (algorithm + trainer) |
| `run_smacv2_tarmac.py` | TarMAC baseline |
| `run_smacv2_mappo.py` | MAPPO baseline |
| `run_smacv2_maddpg.py` | MADDPG baseline |
| `run_smacv2_eggroll.py` | EGGROLL baseline |
| `run_scaling_experiment.py` | M-scaling experiment runner |
| `run_n_scaling_experiment.py` | N-scaling experiment runner |
| `run_n_scaling_extended.py` | Extended N-scaling (higher N + MADDPG) |
| `SMACV2_COMPARISON_REPORT.md` | Results and analysis |
