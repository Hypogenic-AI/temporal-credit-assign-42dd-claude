# Research Plan: Emergent Temporal Credit Assignment in Async MARL

## Motivation & Novelty Assessment

### Why This Research Matters
Real-world multi-agent systems (robotics, supply chains, autonomous vehicles) involve agents operating at different timescales. Understanding how credit assignment emerges under temporal heterogeneity is essential for designing scalable MARL systems that don't require hand-crafted coordination hierarchies.

### Gap in Existing Work
Per the literature review: TAR2 and STAS address joint agent-temporal credit but assume synchronous agents with uniform horizons. MacroMARL handles asynchronous macro-actions but lacks credit assignment analysis. No work studies whether hierarchical credit assignment *emerges* from heterogeneous decision frequencies, or quantifies this via mutual information between value functions.

### Our Novel Contribution
We test whether agents with heterogeneous decision frequencies spontaneously develop hierarchical temporal credit assignment, measured by mutual information between value functions across timescales. This bridges the gap between temporal abstraction (FeUdal Networks, Options) and multi-agent credit assignment (QMIX, STAS).

### Experiment Justification
- **Exp 1 (Baseline comparison)**: Compare homogeneous vs heterogeneous horizon agents on cooperative tasks to establish that heterogeneity affects learning dynamics
- **Exp 2 (MI analysis)**: Compute mutual information between agent value functions at different time lags to detect emergent temporal dependencies
- **Exp 3 (Ablation)**: Remove temporal context features to confirm coordination strategies are learned, not hard-coded

## Research Question
Do MARL agents with heterogeneous decision horizons spontaneously develop hierarchical temporal credit assignment strategies, and can this be quantified via mutual information between value functions?

## Hypothesis Decomposition
1. **H1**: Heterogeneous-horizon agents achieve comparable or better team performance than homogeneous agents on cooperative tasks requiring temporal coordination
2. **H2**: Mutual information I(V_i(t), V_j(t+τ)) between slow and fast agents' value functions is significantly higher than between same-speed agents
3. **H3**: Slow agents develop value functions that implicitly model fast agents' future contributions (higher TD-error propagation depth)
4. **H4**: Removing temporal context features degrades coordination, confirming learned (not hard-coded) strategies

## Methodology

### Approach
Custom gridworld cooperative environments with 3 agents at different action frequencies (1x, 2x, 4x). Train using Independent PPO with temporal context features. Analyze value function mutual information and TD-error propagation patterns.

### Environments
1. **Cooperative Relay**: Agents must pass an item across the grid in sequence. Different frequencies create natural temporal coordination challenges.
2. **Multi-Pace Foraging**: Agents collect resources that appear at different rates, requiring coordination between fast scouts and slow collectors.
3. **Synchronized Rendezvous**: All agents must reach a common location, but move at different speeds.

### Baselines
- Homogeneous agents (all same frequency)
- No temporal context features
- Random policy baseline

### Evaluation Metrics
- Cumulative team reward
- I(V_i(t), V_j(t+τ)) mutual information
- TD-error propagation depth
- Coordination efficiency (synchronized actions / total)

### Statistical Analysis
- 5 random seeds per condition
- Welch's t-test for performance comparison
- Bootstrap confidence intervals for MI estimates

## Timeline (1 hour total)
- Planning: 5 min (done)
- Implementation: 25 min
- Experiments: 20 min
- Analysis & Documentation: 10 min

## Success Criteria
- Clear performance difference between homogeneous and heterogeneous conditions
- Significant MI signal between slow and fast agents
- Ablation confirms learned coordination
