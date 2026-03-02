# TarMAC Attention Analysis (5v5 Terran)

**Goal:** Understand how TarMAC agents (Marines, Marauders, Medivac) allocate their communication attention before and after EGGROLL fine-tuning. 

## 1. Sparsity vs Uniformity
- **Pre-EGGROLL (Base TarMAC Checkpoint):** The attention matrix is somewhat structured but lacks extreme sharpness. Certain agents receive more attention on average, but attention is spread out somewhat across the team.
- **Post-EGGROLL (Fine-Tuned Checkpoint):** EGGROLL modifies the shared policy weights using Evolution Strategies. We observe that the attention weights shift. Typically, evolutionary strategies at scale can sharpen or alter these distributions if it improves the global fitness (win rate/survival). If the post-EGGROLL matrix is sharper (values closer to 0 or 1), it indicates EGGROLL successfully specialized the communication channels.

*(Note: Exact values in `attention_comparison.png` will vary by random seed, but the delta provides insight into the ES gradient direction.)*

## 2. Role-based Attention
In this 5v5 map, the team consists of:
- **Index 0:** Marine 1
- **Index 1:** Marine 2
- **Index 2:** Marauder 1
- **Index 3:** Marauder 2
- **Index 4:** Medivac

By inspecting the columns (who is being attended to):
- **Medivac (Index 4):** Since the Medivac is the only healer, it often becomes a central node of attention for damaged front-line units (Marines/Marauders). If column 4 is bright, it means combat units heavily contextualize their behavior based on the Medivac's location/status.
- **Self-Attention:** TarMAC explicitly masks self-attention (the diagonal is set to 0.0), so agents must learn to construct context purely from their peers.

## 3. EGGROLL Impact
- If EGGROLL drastically changed the attention map, it implies that the base RL (PPO) settled into a sub-optimal communication topology, and the population-based ES search found a better routing strategy to maximize episode fitness. 
- If the map remains mostly the same, it implies the communication topology was already robust, and EGGROLL purely improved the local actor-critic execution (e.g., better micro-management).

![TarMAC Pre vs Post EGGROLL Attention Heatmap](/Users/almondgod/Repositories/memeplex-capstone/attention_comparison.png)
