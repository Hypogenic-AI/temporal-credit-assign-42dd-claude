# Cloned Repositories

## 1. PyMARL — Original MARL Framework
- **URL**: https://github.com/oxwhirl/pymarl
- **Location**: `code/pymarl/`
- **Purpose**: Base framework for value decomposition methods (QMIX, VDN, COMA) on SMAC
- **Key files**: `src/runners/`, `src/learners/`, `src/modules/mixers/`
- **Dependencies**: PyTorch, StarCraft II
- **Notes**: The foundation that PyMARL2 and EPyMARL extend

## 2. PyMARL2 — Extended MARL with More Algorithms
- **URL**: https://github.com/hijkzzz/pymarl2
- **Location**: `code/pymarl2/`
- **Purpose**: Includes QPLEX, Weighted QMIX, LICA, RODE, and more with fine-tuned hyperparameters
- **Key files**: `src/learners/`, `src/modules/mixers/`, `src/config/`
- **Stars**: 705
- **Notes**: Achieves 100% win rate on most SMAC scenarios with fine-tuned QMIX

## 3. EPyMARL — Extended PyMARL with Policy Gradient Methods
- **URL**: https://github.com/uoe-agents/epymarl
- **Location**: `code/epymarl/`
- **Purpose**: Adds MAPPO, IPPO, MADDPG, MAA2C to PyMARL. Supports PettingZoo, VMAS, SMACv2.
- **Key files**: `src/runners/`, `src/learners/`, `src/config/`
- **Stars**: 661
- **Notes**: **Best starting point for our experiments** — native PettingZoo integration allows easy custom environment support

## 4. RUDDER — Return Decomposition
- **URL**: https://github.com/ml-jku/rudder
- **Location**: `code/rudder/`
- **Purpose**: Original implementation of RUDDER for temporal credit assignment via return decomposition
- **Key files**: Atari game implementations, LSTM-based return prediction
- **Notes**: Single-agent; theory could be extended to multi-agent temporal credit

## 5. MacroMARL — Macro-Action Multi-Agent RL
- **URL**: https://github.com/yuchen-x/MacroMARL
- **Location**: `code/MacroMARL/`
- **Purpose**: MacDec-POMDP framework for asynchronous multi-agent RL with macro-actions of different durations
- **Key files**: Environment definitions, MacDec-DDRQN, Mac-IAC, Mac-CAC algorithms
- **Notes**: **Most directly relevant** — provides native heterogeneous decision horizon support

## Additional Repos (Not Cloned — Available Online)

### Credit Assignment

| Name | URL | Purpose |
|------|-----|---------|
| STAS | github.com/zowiezhang/STAS | Spatial-temporal return decomposition (AAAI 2024) — **clone for baseline** |
| LICA | github.com/mzho7212/LICA | Implicit credit assignment (NeurIPS 2020) |
| MLCA | github.com/YuxuanXie/MLCA | Multi-level credit assignment for cooperative MARL |
| QPLEX | github.com/wjh720/QPLEX | Duplex dueling value decomposition |
| Weighted QMIX | github.com/oxwhirl/wqmix | Relaxed monotonicity QMIX |

### Frameworks and Environments

| Name | URL | Purpose |
|------|-----|---------|
| SMAC | github.com/oxwhirl/smac | StarCraft environment (install via pip when SC2 available) |
| SMACv2 | github.com/oxwhirl/smacv2 | Updated SMAC benchmark with randomized units |
| JaxMARL | github.com/FLAIROx/JaxMARL | JAX-accelerated MARL (12,500x faster, includes SMAX) |
| MARLlib | github.com/Replicable-MARL/MARLlib | Comprehensive MARL library on Ray/RLlib |
| Mava | github.com/instadeepai/Mava | JAX-based MARL with MAT and Sable |
| MacDec-via-Cen | github.com/yuchen-x/MacDec-via-Cen | Companion to MacroMARL for centralized Q-net |

### Hierarchical RL

| Name | URL | Purpose |
|------|-----|---------|
| Hierarchical MARL | github.com/011235813/hierarchical-marl | Hierarchical cooperative MARL with skill discovery |
| TJU-DRL-LAB | github.com/TJU-DRL-LAB/Multiagent-RL | Collection of MARL code (scalability, credit, exploration) |
