# Literature Review: Emergent Temporal Credit Assignment in Asynchronous Multi-Agent Reinforcement Learning with Heterogeneous Horizons

## Research Area Overview

This research investigates whether multi-agent RL systems with heterogeneous decision horizons spontaneously develop hierarchical temporal credit assignment strategies. The hypothesis posits that agents with longer horizons will learn to implicitly model and compensate for shorter-horizon agents' delayed feedback, and that this emergence can be quantified through mutual information between agent value functions across timescales.

The literature spans three intersecting subfields: (1) credit assignment in cooperative MARL, (2) temporal abstraction and hierarchical RL, and (3) asynchronous multi-agent decision-making.

---

## Key Papers

### 1. Agent-Temporal Credit Assignment (TAR2) — Kapoor et al. (RLC 2024)
- **arXiv**: 2412.14779
- **Key Contribution**: First principled framework for **joint** agent-temporal credit decomposition. Decomposes episodic returns into per-agent, per-timestep rewards using learned temporal weights w_t and agent weights w'_{i,t}.
- **Methodology**: Uses alternating temporal and agent attention modules built on AREL. Temporal weights sum to 1 over time; agent weights sum to 1 over agents at each timestep.
- **Theoretical Result**: Proven equivalent to potential-based reward shaping (preserves Nash equilibria). Policy gradient direction is preserved. Variance of advantage estimates grows linearly with agent count without proper decomposition.
- **Environments**: SMACLite (5m_vs_6m), Alice & Bob, Google Football
- **Key Finding**: TAR2 + simple IPPO/MAPPO matches or beats dedicated MARL methods, suggesting credit assignment can be solved separately from the coordination algorithm.
- **Code**: Not publicly available
- **Relevance**: **Highest**. Directly addresses the agent-temporal credit intersection. However, does NOT address heterogeneous horizons or asynchronous execution.

### 2. STAS: Spatial-Temporal Return Decomposition — Chen et al. (AAAI 2024)
- **arXiv**: 2304.07520
- **Key Contribution**: Dual transformer architecture for simultaneous spatial (agent) and temporal credit decomposition from episodic rewards.
- **Methodology**: (1) Causal transformer for temporal decomposition (each timestep attends only to past); (2) Shapley attention for spatial decomposition using random coalition masks to approximate marginal contributions. STAS-ML variant interleaves both modules across stacked layers.
- **Training**: Simple reconstruction loss — sum of all decomposed Shapley values should equal episodic return.
- **Environments**: Alice & Bob (grid-world with sequential cooperation), MPE (3, 6, 15 agents) — all with episodic-only rewards
- **Key Results**: Robust to agent scaling (3→15 agents) while baselines degrade. Pearson correlation between credit and distance-to-target confirms meaningful decomposition.
- **Code**: https://github.com/zowiezhang/STAS
- **Relevance**: **Very high**. Provides a concrete architecture for joint spatial-temporal decomposition. Could be extended to heterogeneous horizons.

### 3. LICA: Learning Implicit Credit Assignment — Zhou et al. (NeurIPS 2020)
- **arXiv**: 2007.02529
- **Key Contribution**: Implicit credit assignment via hypernetwork-based mixing critic that introduces multiplicative state-action associations into policy gradients.
- **Methodology**: Centralized critic formulated as a hypernetwork mapping state → weights that mix individual action vectors into joint Q-value. Adaptive entropy regularization rescales entropy gradients inversely proportional to current policy entropy.
- **Environments**: MPE (Predator-Prey, Cooperative Navigation), SMAC
- **Key Results**: Outperforms COMA, MADDPG, SQDDPG. Mixing critic (CMix) provides richer credit assignment than MLP critic via multiplicative state embedding in gradients.
- **Code**: Available within PyMARL2 framework
- **Relevance**: **High**. Important baseline for implicit credit assignment approaches.

### 4. RUDDER: Return Decomposition for Delayed Rewards — Arjona-Medina et al. (NeurIPS 2019)
- **arXiv**: 1806.07857
- **Key Contribution**: Reward redistribution to create return-equivalent MDPs with zero expected future rewards, simplifying Q-value estimation to computing means.
- **Methodology**: (1) Reward redistribution creating return-equivalent Sequence-Markov Decision Processes (SDPs); (2) Return decomposition via contribution analysis using LSTMs to predict returns and attribute credit to state-action pairs.
- **Theoretical Result**: Optimal reward redistribution makes expected future rewards zero. Return-equivalent SDPs preserve optimal policies.
- **Environments**: Atari games (on top of PPO), artificial delayed reward tasks
- **Key Results**: Significantly faster than MC, exponentially faster than MCTS and TD(λ) for delayed rewards.
- **Code**: https://github.com/ml-jku/rudder
- **Relevance**: **High**. Foundational work on temporal credit assignment. Single-agent but the theory of return-equivalent processes could extend to multi-agent settings.

### 5. QMIX: Monotonic Value Function Factorisation — Rashid et al. (ICML 2018)
- **arXiv**: 1803.11485
- **Key Contribution**: Foundational value decomposition method for cooperative MARL. Learns a monotonic mixing network conditioned on global state.
- **Methodology**: Individual agent Q-values mixed via hypernetwork with non-negative weights, ensuring monotonicity (∂Q_tot/∂Q_a ≥ 0). Enables greedy decentralized execution.
- **Environments**: SMAC
- **Relevance**: **Essential baseline**. The dominant value decomposition method in cooperative MARL.

### 6. COMA: Counterfactual Multi-Agent Policy Gradients — Foerster et al. (AAAI 2018)
- **arXiv**: 1705.08926
- **Key Contribution**: Counterfactual baseline for multi-agent credit assignment. Each agent's advantage is computed by marginalizing over its actions while holding other agents' actions fixed.
- **Environments**: SMAC, StarCraft unit micromanagement
- **Relevance**: **Important baseline** for explicit credit assignment.

### 7. MAPPO: Multi-Agent PPO — Yu et al. (NeurIPS 2022)
- **arXiv**: 2103.01955
- **Key Contribution**: Shows that properly tuned PPO with parameter sharing achieves SOTA on cooperative MARL benchmarks, challenging the need for complex value decomposition.
- **Environments**: SMAC, MPE, Google Football, Hanabi
- **Relevance**: **Essential baseline**. Simple but strong baseline for cooperative MARL.

### 8. Option-Critic Architecture — Bacon et al. (AAAI 2017)
- **arXiv**: 1609.05140
- **Key Contribution**: End-to-end learning of options (temporally extended actions) and their termination conditions within the options framework.
- **Relevance**: **Moderate**. Provides theoretical foundation for temporal abstraction. Could be combined with multi-agent methods for hierarchical temporal credit assignment.

### 9. FeUdal Networks — Vezhnevets et al. (ICML 2017)
- **arXiv**: 1703.01161
- **Key Contribution**: Hierarchical RL with a Manager (sets goals at lower temporal resolution) and a Worker (executes primitive actions). Manager operates at a coarser timescale.
- **Relevance**: **Moderate-High**. Directly demonstrates heterogeneous decision horizons in a hierarchical setting. Conceptually similar to what might emerge in our multi-agent setting.

### 10. MAVEN: Multi-Agent Variational Exploration — Mahajan et al. (NeurIPS 2019)
- **arXiv**: 1910.07483
- **Key Contribution**: Addresses committed exploration in value decomposition by conditioning on a shared latent variable that controls joint exploration.
- **Relevance**: **Moderate**. Exploration in MARL is important when studying emergent behaviors.

### 11. RODE: Role-Based Decomposition — Wang et al. (ICLR 2021)
- **arXiv**: 2011.09189
- **Key Contribution**: Decomposes joint action space by learning agent roles that restrict available actions, enabling more efficient exploration.
- **Relevance**: **Moderate**. Role decomposition relates to heterogeneous agent specialization.

### 12. QPLEX: Duplex Dueling Value Decomposition — Wang et al. (ICLR 2021)
- **arXiv**: 2006.04222
- **Key Contribution**: More expressive value decomposition than QMIX by using duplex dueling architecture, achieving full expressiveness while maintaining decentralized execution.
- **Relevance**: **Moderate**. Important value decomposition baseline.

### 13. Weighted QMIX — Rashid et al. (NeurIPS 2020)
- **arXiv**: 2011.09533
- **Key Contribution**: Relaxes QMIX's monotonicity constraint by weighting the projection error, allowing better approximation of the true Q-function.
- **Relevance**: **Moderate**. Addresses limitations of QMIX's structural constraints.

### 14. CIA: Contrastive Identity-Aware Learning — Li et al. (AAAI 2022)
- **arXiv**: 2202.02673
- **Key Contribution**: Uses contrastive learning to learn identity-aware agent representations for better credit assignment in value decomposition.
- **Relevance**: **Moderate**. Agent identity differentiation is relevant to heterogeneous settings.

### 15. MACCA: Offline MARL with Causal Credit Assignment — Wang et al. (2023)
- **arXiv**: 2312.03644
- **Key Contribution**: Causal reasoning for credit assignment in offline MARL settings.
- **Relevance**: **Low-Moderate**. Causal approach to credit assignment could inform temporal analysis.

### 16. Hindsight Credit Assignment — Harutyunyan et al. (NeurIPS 2019)
- **arXiv**: 1912.02503
- **Key Contribution**: Uses future information to improve temporal credit assignment via hindsight-conditioned policy gradients.
- **Relevance**: **Moderate**. Temporal credit with hindsight relates to our delayed feedback setting.

---

## Common Methodologies

1. **Value Decomposition**: QMIX, VDN, QPLEX, Weighted QMIX — decompose joint Q-value into individual agent utilities. Used in Papers 5, 12, 13.
2. **Counterfactual Baselines**: COMA, SQDDPG — assess agent contributions via counterfactual reasoning. Used in Papers 6, related to Paper 2 (Shapley attention).
3. **Reward Redistribution/Decomposition**: RUDDER, TAR2, STAS — redistribute episodic rewards to per-step signals. Used in Papers 1, 2, 4.
4. **Hypernetwork-Based Mixing**: QMIX, LICA — use state-conditioned weight generation for mixing. Used in Papers 3, 5.
5. **Temporal Abstraction**: Options, FeUdal Networks — multi-scale decision-making. Used in Papers 8, 9.
6. **Macro-Action MARL**: MacDec-POMDP framework — agents execute temporally extended actions of different durations. Used in MacroMARL.

## Standard Baselines

- **QMIX** — primary value decomposition baseline
- **VDN** — additive value decomposition (simpler baseline)
- **MAPPO** — strong policy gradient baseline
- **COMA** — counterfactual credit assignment baseline
- **Independent PPO/Q-learning** — no credit assignment baseline

## Evaluation Metrics

- **Win rate** (SMAC): percentage of episodes won
- **Episode return**: cumulative reward per episode
- **Convergence speed**: training steps to reach threshold performance
- **Credit assignment quality**: Pearson correlation between assigned credit and ground-truth contributions (STAS)
- **Variance of policy gradients**: theoretical metric for credit assignment quality (TAR2)

## Datasets/Environments in the Literature

- **SMAC/SMACv2** (StarCraft): Used by QMIX, COMA, MAPPO, LICA, QPLEX, WQMIX, RODE, MAVEN
- **MPE** (Multi-Agent Particle): Used by LICA, STAS, MADDPG
- **Alice & Bob** (grid-world): Used by STAS, TAR2
- **SMACLite**: Used by TAR2
- **MacDec-POMDP environments** (Box Pushing, Warehouse): Used by MacroMARL, ToMacVF

---

## Gaps and Opportunities

1. **No work addresses emergent hierarchical credit assignment**: Existing methods design hierarchies explicitly (FeUdal) or assume synchronous agents (QMIX, STAS). No one has studied whether hierarchy *emerges* from heterogeneous horizons.

2. **Temporal and agent credit assignment are treated separately**: TAR2 and STAS combine them but in fixed architectures. The interaction between timescale heterogeneity and credit assignment is unexplored.

3. **Mutual information between value functions across timescales is unstudied**: No existing work quantifies cross-timescale information sharing between agent value functions as a measure of emergent coordination.

4. **Asynchronous MARL credit assignment is nascent**: MacroMARL provides the environment framework but lacks sophisticated credit assignment methods. Combining TAR2/STAS-style decomposition with MacDec-POMDP settings is an open direction.

5. **Scalability of joint decomposition**: Both TAR2 and STAS have been tested on relatively small agent counts (≤15). Behavior with heterogeneous horizons at scale is unknown.

---

## Recommendations for Our Experiment

### Recommended Environments
1. **MacroMARL / MacDec-POMDP environments** — Native support for heterogeneous decision horizons (different macro-action durations). Most directly aligned with the hypothesis.
2. **Modified MPE via PettingZoo** — Easily customizable for different agent decision frequencies. Fast iteration. Use AEC API for asynchronous stepping.
3. **SMAC/SMACv2** — For community-standard baseline comparisons.

### Recommended Baselines
1. **QMIX** — Standard value decomposition
2. **MAPPO** — Strong policy gradient baseline
3. **TAR2** — Agent-temporal credit (if reimplementable)
4. **STAS** — Spatial-temporal decomposition (code available)
5. **Independent PPO** — No credit assignment control

### Recommended Metrics
1. **Task performance** (return, win rate) — Does heterogeneous-horizon MARL learn effectively?
2. **Mutual information** between agent value functions across timescales — Core hypothesis metric
3. **Emergent hierarchy score** — Measure whether longer-horizon agents develop coordinator-like behavior
4. **Credit assignment accuracy** — Pearson correlation with ground-truth contributions
5. **Convergence speed** — Does emergent hierarchy accelerate learning?

### Methodological Considerations
- Start with MPE for fast prototyping of heterogeneous-horizon mechanics
- Use MacroMARL for realistic asynchronous evaluation
- Implement mutual information estimation between value functions using MINE or similar neural estimators
- Consider both fixed heterogeneous horizons (controlled experiment) and learned horizons (emergent experiment)
- Use RUDDER's return-equivalent SDP theory as theoretical foundation for reward redistribution across timescales
