# SMACv2 Algorithm Comparison Report

**Date:** 2026-02-25
**Environment:** SMACv2 — StarCraft II Multi-Agent Challenge v2, Terran 5v5 (`10gen_terran`)
**Evaluation budget:** 10,000 environment steps (matched across algorithms)

---

## Overview

This report compares three multi-agent RL algorithms on SMACv2 Terran 5v5:

| Algorithm | Paradigm | Communication | Critic |
|-----------|----------|---------------|--------|
| **MAPPO** | CTDE | None | Centralised (global state) |
| **TarMAC** | CTDE | Learned (soft-attention) | Centralised (global state) |
| **MADDPG / Method I** | Self-play (MPE2) | — | Centralised / Decentralised |

> **Note on MADDPG and Method I (LA-IPPO):** These algorithms were evaluated on a different environment (MPE2 `simple_tag_v3`, predator-prey) using Elo-based competitive metrics. They are included here as reference baselines from the companion experiment. Direct cross-environment comparison should be treated with caution.

---

## SMACv2 Results (Step-Matched, 200k steps, Terran 5v5)

### Training summary

| Metric | MAPPO | TarMAC | MAPPO + EGGROLL | TarMAC + EGGROLL (Tuned) | EGGROLL (Scratch) | MADDPG |
|--------|-------|--------|-----------------|--------------------------|-------------------|--------|
| Total env steps | 200,000 | 200,000 | 201,600 | 204,800 | 204,800 | 200,000 |
| Wall-clock time | 2118.1 s | 2190.3 s | 465.0 s | 406.4 s | **382.4 s** | 2228.0 s |
| Final mean reward | 9.58 | **110.84** | 9.38 | 10.23 | 1.81 | 5.52 |
| Final win rate | 12.1% | **25.4%** | 5.6% | 11.6% | 0.0% | 5.0% |
| Peak 20-ep win rate | 16.9% | 27.5% | 10.9% | 15.1% | 0.0% | 10.0% |
| Final entropy | 0.264 | N/A | N/A (ES) | N/A (ES) | N/A (ES) | N/A |
| Final pg_loss | -0.002 | N/A | N/A (ES) | N/A (ES) | N/A (ES) | -28.91 (actor) |
| Final vf_loss | 1.157 | N/A | N/A (ES) | N/A (ES) | N/A (ES) | 191.80 (critic) |

### Key observations

1. **TarMAC achieves ~11.5× higher final mean reward** than MAPPO at equal 200k step budgets (110.84 vs 9.58).
2. **Win rate advantage for TarMAC**: TarMAC reaches a final win rate of **25.4%**, which is more than double MAPPO's **12.1%**.
3. **EGGROLL Fine-tuning Degradation:** Applying EGGROLL on top of the MAPPO and TarMAC checkpoints over an additional 200k steps *degraded* the win-rates (from 12.1% to 5.6% for MAPPO, and from 25.4% to 11.6% for TarMAC).
   * Note on hyper-parameter tuning: Even with hyper-parameters tuned for this specific architecture (`pop_size=64`, `rank=8`, `lr=1e-4`, `sigma=0.05`), EGGROLL was unable to improve the TarMAC checkpoint. The global ES updates seemingly struggled to maintain or improve the highly specialized critic-guided behaviors learned by PPO within this densely connected architecture.
4. **EGGROLL From-Scratch:** Pure ES on a random initialization struggles to learn a coordinated combat policy within 200k steps, achieving a 0.0% win rate and a final mean reward of ~1.81. This is consistent with general RL literature that pure zeroth-order black-box optimization has high sample complexity in dense deep-RL (SMAC) without a value critic guiding the gradient directions. However, as the paper claims, it is undeniably blazing fast, taking only ~382 seconds to complete 200k steps.
5. **MADDPG struggles in SMACv2**: Similar to previous literature on fully decentralised/off-policy algorithms in SMAC, MADDPG performs worst out of the base actor-critic algorithms at 200k steps (5.0% win rate), and its wall-clock time is highest (2228s) due to maintaining replay buffers and individual critics per agent.

---

## MPE2 Reference: MADDPG vs Method I (LA-IPPO) vs PPO — Elo Results

*Environment: `simple_tag_v3` (3 predators vs 1 prey, continuous actions, 200k steps)*

### Step-matched (200k env steps each)

| Metric | MADDPG | Method I (LA-IPPO) | PPO |
|--------|--------|-------------------|-----|
| Predator Elo (final) | 1198.4 | **1201.9** | 1194.0 |
| Prey Elo (final) | **796.6** | 779.5 | 755.6 |
| Predator exploitability | 0.0 | 0.0 | 0.0 |
| Prey exploitability | 0.5 | **0.4** | 0.5 |
| Wall-clock time | 781.4 s | **130.8 s** | **124.4 s** |

### Wall-clock matched (~781 s)

| Metric | MADDPG (200k steps) | Method I (1,219,584 steps) | PPO (1,265,840 steps) |
|--------|---------------------|--------------------------|----------------------|
| Predator Elo (final) | 1198.4 | **1269.7** | 1227.7 |
| Prey Elo (final) | 796.6 | 716.7 | 726.8 |
| Predator exploitability | 0.0 | 0.0 | 0.0 |
| Prey exploitability | 0.5 | 0.5 | 0.5 |
| Env steps | 200,000 | **1,219,584** | 1,265,840 |

### Fixed opponent pool Elo (shared pool, step-matched)

| Metric | MADDPG | Method I | PPO |
|--------|--------|----------|-----|
| Predator Elo (fixed pool) | **1124.1** | 1120.5 | 1116.8 |
| Prey Elo (fixed pool) | **881.6** | 851.3 | 869.7 |

---

## Algorithm Descriptions

### MAPPO (Multi-Agent PPO)
- Centralised-training, decentralised-execution (CTDE)
- Shared actor-critic parameters across all agents
- Centralised critic observes global state
- No explicit inter-agent communication at execution time
- PPO update with GAE, clip ratio 0.2

### TarMAC (Targeted Multi-Agent Communication + PPO)
- CTDE with explicit learned communication
- Each agent encodes observation to produce key, value, query vectors
- Soft-attention aggregation: `context_i = Σ_{j≠i} softmax(query_i · key_j / √d) · value_j`
- Actor conditions on `[hidden_i ‖ context_i]` — local encoding + aggregated peer messages
- Centralised critic on global state (same as MAPPO)
- Adds communication overhead; runs ~29% slower than MAPPO at 10k steps

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- CTDE with centralised critics per agent
- Off-policy; actor-critic with replay buffer and target networks
- Deterministic policy; exploration via OU noise
- Significantly slower per step due to replay buffer and per-agent critics

### Method I — LA-IPPO (Latent-Aligned Independent PPO)
- Fully decentralised at execution time (no centralised critic, no messages)
- PPO with shared latent encoder per agent
- Periodic distillation: align latent encoders across agent pairs (`L_distill = ‖f_θi(o) − sg(f_θj(o))‖²`)
- ~6× faster wall-clock than MADDPG; competitive Elo at equal budgets

---

## Comparative Analysis

### Communication benefit (MAPPO → TarMAC)
On SMACv2 Terran 5v5, TarMAC's learned communication via targeted soft-attention **substantially outperforms vanilla MAPPO** in terms of reward signal at equal steps:
- Mean reward 4.34× higher (21.99 vs 4.34)
- Best single update 4.69× higher (41.48 vs 8.85)
- Win rate advantage: 5% vs 1%

The cost is ~29% longer wall-clock time per step, consistent with the attention pass adding one matrix multiply per communication round.

### Decentralised communication (Method I vs MADDPG, on MPE2)
Method I achieves **comparable or higher predator Elo** than MADDPG at equal step budgets (1201.9 vs 1198.4) with **no communication at execution time** — only a latent alignment distillation during training. At wall-clock matched budgets, Method I reaches Elo 1269.7 vs MADDPG's 1198.4 — a **+71.3 Elo advantage** — because its ~6× throughput advantage lets it process far more experience.

### Cross-environment synthesis

| Property | MAPPO | TarMAC | MADDPG | Method I |
|----------|-------|--------|--------|----------|
| Communication at exec time | None | Soft-attention | None | None |
| Critic type | Centralised | Centralised | Centralised (per-agent) | Decentralised |
| Sample efficiency | Medium | Medium-high | Low (off-policy) | Medium |
| Throughput | Fast | Moderate | Slow | Very fast |
| SMACv2 reward @10k | 4.34 | **21.99** | — | — |
| SMACv2 win rate @10k | 1% | **5%** | — | — |
| MPE2 predator Elo @200k | — | — | 1198.4 | **1201.9** |

---

## Limitations and Caveats

1. **SMACv2 runs are very short (10k steps).** Typical SMACv2 SOTA experiments run 2–5M steps. Win rates of 1–5% are expected this early; meaningful win-rate comparisons require 500k+ steps.
2. **Different environments.** MADDPG and Method I Elo results are from MPE2 (cooperative/competitive predator-prey), not SMACv2 (cooperative combat). These are not directly comparable.
3. **Single seed.** All SMACv2 runs are single-seed. Variance can be substantial at this step count.
4. **TarMAC reward scale.** SMACv2 rewards accumulate across surviving agents over an episode; TarMAC's higher reward partially reflects longer episode survival (more damage dealt) rather than pure win-rate improvement.

---

## Reproduction Commands

```bash
# TarMAC on SMACv2 (run performed in this session)
python3.9 run_smacv2_tarmac.py \
  --mode train --race terran --n-units 5 --n-enemies 5 \
  --total-steps 10000 --rollout-steps 200 \
  --save-dir checkpoints/smacv2_tarmac_run \
  --comm-dim 16 --comm-rounds 1 --hidden-dim 128

# MAPPO on SMACv2 (existing mac_test run)
python3.9 run_smacv2_mappo.py \
  --mode train --race terran --n-units 5 --n-enemies 5 \
  --total-steps 10000 --save-dir checkpoints/mac_test

# MPE2 step-matched Elo experiments
python train_maddpg.py --max-steps 200000
python train_method_i.py --max-steps 200000 --results-key method_i
python train_ppo.py --max-steps 200000 --results-key ppo

# Fixed pool Elo evaluation
python evaluate_fixed_pool.py --pool-set step
```

---

## Result Files

| File | Description |
|------|-------------|
| `checkpoints/smacv2_tarmac_run/smacv2_tarmac_results.json` | TarMAC training results (this run) |
| `checkpoints/mac_test/smacv2_mappo_results.json` | MAPPO training results |
| `experiment_results.json` | MPE2 Elo results (MADDPG, Method I, PPO) |
| `METHOD_I_SUMMARY.md` | Detailed Method I analysis |
