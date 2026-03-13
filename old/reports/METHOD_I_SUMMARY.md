# Method I Summary (Final Form)

## What Method I Is
Method I (LA-IPPO) is independent PPO with a shared latent encoder per agent and a periodic distillation phase that aligns agent latents without sharing observations or explicit messages at execution time.

Key components:
- Encoder: maps local observation to latent memory.
- Policy head: outputs action distribution from latent.
- Value head: estimates return from latent.
- Distillation: every k learn steps, each agent aligns its encoder to another agent’s encoding on its own observation buffer.

## Final Method I Configuration (as used in the perfected experiments)
- Algorithm: LA-IPPO (independent PPO + latent distillation)
- Environment: MPE2 `simple_tag_v3`
- Teams: 3 predators vs 1 prey, continuous actions, max cycles 25
- Training: self-play (predators and prey both learn)
- PPO hyperparameters: lr 3e-4, gamma 0.95, GAE 0.95, clip 0.2, entropy 0.01, vf 0.5, 4 update epochs, batch 64
- Network: hidden layers [128, 128], latent dim 64
- Distillation: interval 10, weight 0.1, batch 64, obs buffer 512
- LR anneal: linear to 0 across estimated learn steps

## Why the Old Reward Metric Was Wrong
Raw episode returns are non-stationary in competitive self-play. When both sides improve, rewards can remain flat or even degrade, which makes return-based curves misleading.

## Final Evaluation Protocol ("Perfected" Experiments)
We replaced raw return with competitive metrics that are stable and comparable:
- Rolling opponent pool of frozen policies (size 5)
- Elo vs the pool (K=32)
- Exploitability approximation: `max(0, 0.5 - min win rate vs pool)`
- Periodic win-rate matrix to detect cycles

Additionally, we compute **fixed opponent pool Elo** where all opponents are held constant at rating 1000. This makes Elo values strictly comparable across algorithms.

## Results: Step-Matched (200k env steps each)

| Metric | MADDPG | Method I | PPO |
|--------|--------|----------|-----|
| Predator Elo (final) | 1198.4 | 1201.9 | 1194.0 |
| Prey Elo (final) | 796.6 | 779.5 | 755.6 |
| Predator exploitability | 0.0 | 0.0 | 0.0 |
| Prey exploitability | 0.5 | 0.4 | 0.5 |
| Wall-clock time | 781.4 s | 130.8 s | 124.4 s |

## Results: Wall-Clock Matched (~781 s)

| Metric | MADDPG (200k steps) | Method I (1,219,584 steps) | PPO (1,265,840 steps) |
|--------|----------------------|----------------------------|-----------------------|
| Predator Elo (final) | 1198.4 | 1269.7 | 1227.7 |
| Prey Elo (final) | 796.6 | 716.7 | 726.8 |
| Predator exploitability | 0.0 | 0.0 | 0.0 |
| Prey exploitability | 0.5 | 0.5 | 0.5 |
| Env steps | 200,000 | 1,219,584 | 1,265,840 |

## Results: Fixed Opponent Pool Elo (Shared Pool, Step-Matched)
Pool = random + last 5 checkpoints from each algorithm’s 200k run. Opponent ratings fixed at 1000.

| Metric | MADDPG | Method I | PPO |
|--------|--------|----------|-----|
| Predator Elo (fixed pool) | 1124.1 | 1120.5 | 1116.8 |
| Prey Elo (fixed pool) | 881.6 | 851.3 | 869.7 |
| Predator exploitability | 0.0 | 0.0 | 0.0 |
| Prey exploitability | 0.5 | 0.5 | 0.5 |

## Cross-Play (MADDPG vs Method I)
Cross-play win-rate matrices show how predator and prey policies generalize to the other algorithm’s opponent pool. These are stored in:
- `checkpoints/maddpg_vs_method_i_pred_winrate_step.png`
- `checkpoints/method_i_vs_maddpg_pred_winrate_step.png`

## What This Says About Method I
- **Competitive viability:** Method I reaches comparable Elo to MADDPG and PPO at the same step budget.
- **Compute efficiency:** In wall-clock matched runs, Method I reaches higher predator Elo because it processes far more environment steps per second.
- **Decentralized advantage:** Method I achieves these results using only local observations at execution time, without a centralized critic.
- **Stability:** With the corrected distillation weight and LR anneal, Method I avoids collapse and maintains steady improvement under the competitive metrics.

## Repro Commands

```
# Step-matched (200k)
python train_maddpg.py --max-steps 200000
python train_method_i.py --max-steps 200000 --results-key method_i
python train_ppo.py --max-steps 200000 --results-key ppo

# Wall-clock matched (~781s)
python train_method_i.py --max-steps 2000000 --max-time 781.4 --results-key method_i_wall_clock_match
python train_ppo.py --max-steps 2000000 --max-time 781.4 --results-key ppo_wall_clock_match

# Fixed opponent pool Elo
python evaluate_fixed_pool.py --pool-set step

# Cross-play matrices
python cross_play_matrix.py --pool-set step
```
