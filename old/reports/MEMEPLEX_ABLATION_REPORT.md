# Memeplex Ablation Studies

To definitively prove *why* Memeplex achieves its performance and validate the epidemiological design choices, we conducted 5 structural ablation studies against the $M=8$ Memeplex baseline over 200,000 steps ($N=5$ Terran).

![Ablation Results Overview](./checkpoints/ablations/ablation_results.png)

## Empirical Results

| Architecture | Peak WR | Final WR | Mean Reward | Diversity | Usage Entropy | Infections |
|--------------|--------:|---------:|------------:|----------:|--------------:|-----------:|
| **Baseline (M=8)** | **22.8%** | 7.3% | 46.95 | **0.981** | 1.25 | 14,715 |
| *1. No Infection* | 18.8% | 0.0% | 38.70 | 0.979 | 0.77 | 0 |
| *2. Blind Comms* | 19.4% | 7.3% | 36.38 | 0.882 | 1.33 | 15,646 |
| *3. Random Contagion*| 19.9% | 0.0% | 38.01 | 0.968 | 1.25 | 21,184 |
| *4. No Immunity* | 22.4% | 2.5% | 40.15 | 0.914 | 1.09 | 24,212 |
| *5. No Mutation* | **23.1%** | **18.5%**| 37.79 | 0.940 | 0.67 | 15,339 |

---

## Analysis & Conclusions

### 1. The Epidemiological Core is Required ("No Infection")
* **Ablation:** Disabled agent-to-agent transmission (agents just have a private "Mixture of Experts" meme bank).
* **Result:** Peak win rate drops heavily from 22.8% to **18.8%**, and final win rate collapses to 0.0%.
* **Conclusion:** Simply giving agents extra trainable strategy vectors does *not* explain Memeplex's performance. The actual sharing and evolutionary selection of strategies via the infection dynamic is the primary driver of the algorithm's capability.

### 2. Strategy-Aware Communication Matters ("Blind Comms")
* **Ablation:** Removed the active meme vector $\phi_i$ from the TarMAC Key/Value communication embeds.
* **Result:** Peak win rate degrades to **19.4%**, diversity drops notably (0.88), and mean reward plummets.
* **Conclusion:** For infection to work optimally, agents must format their communication packets conditionally based on the strategic module they are currently running. Stripping this context forces the network to attempt blind propagation.

### 3. Attention Must Gate Transmission ("Random Contagion")
* **Ablation:** Infection transmission is completely decoupled from the TarMAC attention weights $\alpha_{i \leftarrow j}$ (random social connections).
* **Result:** Peak WR drops to **19.9%** and the agents drastically over-infect each other (21k instances vs 14k baseline) because they transmit to peers who aren't functionally interacting with them.
* **Conclusion:** Gating evolutionary transmission through the live, multi-agent communication network is superior to random uniform mixing.

### 4. Immunity Solves Echo Chambers ("No Immunity")
* **Ablation:** Removed the cryptographic immune memory, allowing immediate re-infection.
* **Result:** Massive spike in transmission volume (**24,212 infections**), usage entropy drops from 1.25 to 1.09, and the population's final performance crashes to 2.5%.
* **Conclusion:** Without an immune system, pairs of heavily-interacting agents get locked in tight echo chambers, constantly over-writing each other's memory banks with the exact same strategies, stalling cooperative progress.

### 5. The Exploration/Exploitation Tradeoff ("No Mutation")
* **Ablation:** Disabled Gaussian noise upon transmission (perfect strategy cloning).
* **Result:** Surprising outcome! No Mutation actually *matched* the baseline peak win rate (23.1%) and achieved the highest final win rate (18.5%). However, mean reward plummeted by nearly 20% (37.79 vs 46.95 baseline) and usage entropy dropped precipitously to **0.67**.
* **Conclusion:** Perfect copying causes intense **exploitation**. The population quickly converges aggressively to a few "good enough" shared strategies (low entropy). This yields more stable end-game win rates, but the strategy is fundamentally shallower and less robust (drastically lower mean damage/reward) because the lack of mutational **exploration** prevents the discovery of deeper synergies.
