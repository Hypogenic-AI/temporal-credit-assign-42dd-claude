# Emergent Temporal Credit Assignment in Async Multi-Agent RL

## Overview
This project investigates whether multi-agent RL systems with heterogeneous decision horizons (agents acting at different frequencies: 1x, 2x, 4x) spontaneously develop hierarchical temporal credit assignment strategies. We use three custom cooperative gridworld environments, Independent PPO training, and mutual information analysis between agent value functions.

## Key Findings

- **No emergent hierarchy**: Agents with different decision frequencies do not spontaneously develop hierarchical credit assignment. Mutual information between value functions shows no directional asymmetry (slow→fast = fast→slow).
- **Task-dependent heterogeneity effects**: Heterogeneous agents outperform homogeneous by 45% in complementary-role tasks (foraging, p<0.0001, d=6.56) but underperform by 17% in synchronization tasks (rendezvous, p=0.001, d=-4.83).
- **Temporal features matter selectively**: Temporal context features significantly improve heterogeneous coordination only in synchronization-heavy tasks (rendezvous: +7.6%, p=0.042).
- **Value coupling is feature-driven, not heterogeneity-driven**: Removing temporal features reduces cross-agent MI by 40-50%, while heterogeneity itself has minimal effect on MI levels.

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch numpy matplotlib scipy scikit-learn

# Run main experiments (~35 min)
export USER=researcher
python src/run_experiments.py

# Run detailed MI analysis (~25 min)
python src/deep_mi_analysis.py
```

## File Structure

| File | Description |
|---|---|
| `REPORT.md` | Full research report with results and analysis |
| `src/environments.py` | Three cooperative gridworld environments |
| `src/agents.py` | Independent PPO implementation |
| `src/analysis.py` | MI, TD-depth, and influence metrics |
| `src/run_experiments.py` | Main experiment runner (45 runs) |
| `src/deep_mi_analysis.py` | Deep MI analysis across conditions |
| `results/` | Metrics JSON files and 17 visualization plots |

## Statistical Summary

| Env | Hetero vs Homo | Hetero vs Ablation |
|---|---|---|
| Relay | p=0.004, d=-2.94 (homo wins) | p=0.287, n.s. |
| Foraging | p<0.0001, d=+6.56 (hetero wins) | p=0.780, n.s. |
| Rendezvous | p=0.001, d=-4.83 (homo wins) | p=0.042, d=+1.53 (temporal helps) |

See `REPORT.md` for full details.
