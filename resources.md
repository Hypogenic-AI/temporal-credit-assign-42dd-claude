# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on "Emergent Temporal Credit Assignment in Asynchronous Multi-Agent Reinforcement Learning with Heterogeneous Horizons."

---

## Papers
Total papers downloaded: 22

| # | Title | Authors | Year | File | Key Relevance |
|---|-------|---------|------|------|---------------|
| 1 | Agent-Temporal Credit Assignment (TAR2) | Kapoor et al. | 2024 | `2412.14779_agent_temporal_credit_assignment.pdf` | Joint agent-temporal decomposition |
| 2 | STAS: Spatial-Temporal Return Decomposition | Chen et al. | 2024 | `2304.07520_stas_spatial_temporal_return.pdf` | Dual transformer for spatial-temporal credit |
| 3 | LICA: Learning Implicit Credit Assignment | Zhou et al. | 2020 | `2007.02529_learning_implicit_credit_assignment.pdf` | Hypernetwork mixing critic |
| 4 | RUDDER: Return Decomposition | Arjona-Medina et al. | 2019 | `1806.07857_rudder_return_decomposition.pdf` | Temporal credit via reward redistribution |
| 5 | QMIX: Monotonic Value Factorisation | Rashid et al. | 2018 | `1803.11485_qmix_monotonic_value_factorisation.pdf` | Foundational value decomposition |
| 6 | COMA: Counterfactual Multi-Agent | Foerster et al. | 2018 | `1705.08926_coma_counterfactual_multi_agent.pdf` | Counterfactual credit assignment |
| 7 | MAPPO | Yu et al. | 2022 | `2103.01955_mappo_multi_agent_ppo.pdf` | Strong PPO baseline for MARL |
| 8 | MAVEN: Multi-Agent Variational Exploration | Mahajan et al. | 2019 | `1910.07483_maven_multi_agent_variational.pdf` | Exploration in value decomposition |
| 9 | CIA: Contrastive Identity-Aware | Li et al. | 2022 | `2202.02673_cia_contrastive_identity_aware.pdf` | Identity-aware credit assignment |
| 10 | RODE: Role-Based Decomposition | Wang et al. | 2021 | `2011.09189_rode_role_decomposition_marl.pdf` | Role decomposition for MARL |
| 11 | QPLEX: Duplex Dueling Value Decomposition | Wang et al. | 2021 | `2006.04222_qplex_duplex_dueling_value_decomp.pdf` | Expressive value decomposition |
| 12 | Weighted QMIX | Rashid et al. | 2020 | `2011.09533_weighted_qmix.pdf` | Relaxed monotonicity for QMIX |
| 13 | Option-Critic Architecture | Bacon et al. | 2017 | `1609.05140_option_critic_architecture.pdf` | Temporal abstraction framework |
| 14 | FeUdal Networks | Vezhnevets et al. | 2017 | `1703.01161_feudal_networks_hierarchical_rl.pdf` | Hierarchical RL with multi-timescale |
| 15 | Hindsight Credit Assignment | Harutyunyan et al. | 2019 | `1912.02503_hindsight_credit_assignment.pdf` | Temporal credit with future info |
| 16 | Asynchronous Multi-Agent RL | Xiao et al. | 2022 | `2209.10113_asynchronous_multi_agent_rl.pdf` | Asynchronous actor-critic for MARL |
| 17 | MACCA: Offline MARL Causal Credit | Wang et al. | 2023 | `2312.03644_macca_offline_causal_credit.pdf` | Causal credit assignment |
| 18 | Temporal Credit in MARL | Various | 2023 | `2312.09858_temporal_credit_marl.pdf` | Temporal credit assignment |
| 19 | Partial Reward Decoupling | Various | 2024 | `2408.04295_partial_reward_decoupling_credit.pdf` | Reward decoupling for credit |
| 20 | Counterfactual Multi-Agent Credit | Various | 2020 | `2003.13903_counterfactual_multi_agent_credit.pdf` | Counterfactual credit methods |
| 21 | SMAC Benchmark | Samvelyan et al. | 2019 | `1906.04118_starcraft_smac_benchmark.pdf` | SMAC environment paper |
| 22 | Understanding Value Decomposition | Various | 2020 | `2011.12895_qmix_is_not_enough.pdf` | Analysis of QMIX limitations |

---

## Environments (Datasets)

For MARL research, "datasets" are RL environments rather than static data files.

| Name | Source | Type | Agents | Async Support | Location/Install |
|------|--------|------|--------|---------------|------------------|
| PettingZoo MPE | `pip install pettingzoo pygame` | Cooperative 2D particle | 3-6 | Via AEC API | Installed in .venv |
| MacroMARL (MacDec-POMDP) | GitHub: yuchen-x/MacroMARL | Cooperative async macro-actions | 2-4 | **Native** | `code/MacroMARL/` |
| SMAC/SMACv2 | GitHub: oxwhirl/smac | Cooperative StarCraft combat | 2-27 | No | Requires SC2 engine |
| JaxMARL | `pip install jaxmarl` | JAX-accelerated MARL envs | Various | No | Not installed (optional) |
| Overcooked-AI | GitHub: HumanCompatibleAI/overcooked_ai | Cooperative cooking | 2 | No (hierarchical subtasks) | Not cloned (optional) |

### Primary Environments for This Research

1. **MacroMARL / MacDec-POMDP** — Most directly relevant. Supports agents with different macro-action durations (heterogeneous horizons). Includes Box Pushing and Warehouse environments.
2. **Modified MPE via PettingZoo** — Fast prototyping. Can be customized for different decision frequencies via the AEC API.
3. **SMAC** — Community standard benchmark for reporting results.

### Environment Verification

PettingZoo MPE was tested and verified working:
```
simple_spread agents: ['agent_0', 'agent_1', 'agent_2']
Action spaces: Discrete(5) per agent
Observation spaces: (18,) per agent
```

---

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Key Info |
|------|-----|---------|----------|----------|
| PyMARL | github.com/oxwhirl/pymarl | MARL framework (QMIX, COMA, VDN) | `code/pymarl/` | Original WhiRL framework for SMAC |
| PyMARL2 | github.com/hijkzzz/pymarl2 | Extended MARL (QPLEX, WQMIX, LICA, RODE) | `code/pymarl2/` | 705 stars, fine-tuned for SMAC |
| EPyMARL | github.com/uoe-agents/epymarl | Extended PyMARL (MAPPO, IPPO, MADDPG) | `code/epymarl/` | 661 stars, PettingZoo+SMACv2 support |
| RUDDER | github.com/ml-jku/rudder | Return decomposition for delayed rewards | `code/rudder/` | Original RUDDER implementation |
| MacroMARL | github.com/yuchen-x/MacroMARL | Macro-action MARL for async agents | `code/MacroMARL/` | MacDec-POMDP environments + algorithms |

### Additional Relevant Repos (Not Cloned)

| Name | URL | Purpose |
|------|-----|---------|
| STAS | github.com/zowiezhang/STAS | Spatial-temporal return decomposition |
| SMAC | github.com/oxwhirl/smac | StarCraft Multi-Agent Challenge |
| SMACv2 | github.com/oxwhirl/smacv2 | Updated SMAC benchmark |
| JaxMARL | github.com/FLAIROx/JaxMARL | JAX-accelerated MARL |
| VMAS | github.com/proroklab/VectorizedMultiAgentSimulator | Vectorized multi-agent simulator |
| BenchMARL | github.com/facebookresearch/BenchMARL | Meta's MARL benchmarking library |

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for three targeted queries covering temporal credit assignment, heterogeneous horizons, and hierarchical MARL.
2. Identified 312 unique papers, filtered to top 25 by relevance scoring.
3. Downloaded 22 papers from arXiv (some Semantic Scholar IDs had wrong arXiv mappings).
4. Deep-read 5 key papers using PDF chunking; skimmed abstracts of all others.

### Selection Criteria
- Papers directly addressing credit assignment in cooperative MARL (highest priority)
- Papers on temporal abstraction and hierarchical RL
- Papers on asynchronous multi-agent decision-making
- Foundational value decomposition methods used as baselines
- Environment/benchmark papers

### Challenges Encountered
- Semantic Scholar API rate limiting (429 errors) required slower requests
- Some arXiv IDs from Semantic Scholar mapped to wrong papers (2407.00718, 2501.02905, 2210.05839 were medical/weather/NLP papers instead of MARL)
- The actual ToMacVF and async credit assignment MARL papers may have different arXiv IDs than initially assumed

### Gaps and Workarounds
- ToMacVF paper could not be downloaded (wrong arXiv ID); the MacroMARL repo contains related algorithms
- Multi-level credit assignment paper had wrong content; the concept is covered by TAR2 and STAS
- SMAC requires StarCraft II engine installation (not feasible in this environment); documented for experiment runner

---

## Recommendations for Experiment Design

### 1. Primary Environments
- **MacroMARL Box Pushing / Warehouse** — Test heterogeneous decision horizons with native macro-action support
- **Custom MPE with heterogeneous stepping** — Modify PettingZoo MPE to have agents with different decision frequencies (e.g., Agent 1 acts every step, Agent 2 every 3 steps, Agent 3 every 5 steps)

### 2. Baseline Methods
- **QMIX** (PyMARL2) — Standard value decomposition
- **MAPPO** (EPyMARL) — Strong policy gradient baseline
- **Independent PPO** — No credit assignment control
- **STAS** (available code) — Spatial-temporal decomposition comparison

### 3. Evaluation Metrics
- Task performance (return, success rate)
- Mutual information between agent value functions at different timescales (core hypothesis metric)
- Emergent hierarchy indicators (e.g., longer-horizon agents' policies conditioned on shorter-horizon agent states)
- Credit assignment accuracy where ground truth is available

### 4. Code to Adapt/Reuse
- **EPyMARL** — Best starting point due to PettingZoo integration and MAPPO/IPPO implementations
- **MacroMARL** — Use directly for MacDec-POMDP experiments
- **PyMARL2** — For QMIX/QPLEX baseline comparisons on SMAC
- **RUDDER** — Adapt return decomposition theory for multi-agent temporal credit
