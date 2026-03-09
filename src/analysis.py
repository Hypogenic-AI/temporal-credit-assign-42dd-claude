"""
Analysis tools for measuring emergent temporal credit assignment:
- Mutual information between agent value functions across time lags
- TD-error propagation depth
- Coordination efficiency metrics
"""

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple
import json


def compute_mutual_information(values_i: np.ndarray, values_j: np.ndarray,
                                max_lag: int = 20, n_bins: int = 20) -> Dict[int, float]:
    """
    Compute mutual information I(V_i(t), V_j(t+τ)) for various time lags τ.

    Uses histogram-based MI estimation for speed and reliability.
    """
    mi_by_lag = {}
    min_len = min(len(values_i), len(values_j))

    for lag in range(0, max_lag + 1):
        if lag >= min_len:
            break
        v_i = values_i[:min_len - lag]
        v_j = values_j[lag:min_len]

        if len(v_i) < 50:
            break

        # Histogram-based MI
        # Discretize into bins
        v_i_d = np.digitize(v_i, np.linspace(v_i.min() - 1e-8, v_i.max() + 1e-8, n_bins))
        v_j_d = np.digitize(v_j, np.linspace(v_j.min() - 1e-8, v_j.max() + 1e-8, n_bins))

        # Joint and marginal distributions
        joint_hist = np.zeros((n_bins + 1, n_bins + 1))
        for a, b in zip(v_i_d, v_j_d):
            joint_hist[a, b] += 1
        joint_hist /= joint_hist.sum()

        p_i = joint_hist.sum(axis=1)
        p_j = joint_hist.sum(axis=0)

        mi = 0.0
        for a in range(n_bins + 1):
            for b in range(n_bins + 1):
                if joint_hist[a, b] > 0 and p_i[a] > 0 and p_j[b] > 0:
                    mi += joint_hist[a, b] * np.log2(joint_hist[a, b] / (p_i[a] * p_j[b]))

        mi_by_lag[lag] = max(mi, 0.0)  # MI is non-negative

    return mi_by_lag


def compute_mi_matrix(value_logs: Dict[int, np.ndarray], max_lag: int = 20
                      ) -> Dict[str, Dict[int, float]]:
    """
    Compute MI between all pairs of agents across time lags.
    Returns dict keyed by "i->j" with MI values per lag.
    """
    n_agents = len(value_logs)
    mi_matrix = {}

    for i in range(n_agents):
        for j in range(n_agents):
            key = f"{i}->{j}"
            mi_matrix[key] = compute_mutual_information(
                value_logs[i], value_logs[j], max_lag=max_lag
            )

    return mi_matrix


def compute_td_error_depth(value_logs: Dict[int, np.ndarray], rewards: List[float],
                           gamma: float = 0.99) -> Dict[int, float]:
    """
    Measure effective horizon of TD-error propagation for each agent.
    Computed as the autocorrelation decay length of TD errors.
    """
    depths = {}
    rewards_arr = np.array(rewards)

    for i, values in value_logs.items():
        if len(values) < 10:
            depths[i] = 0.0
            continue

        # Compute TD errors: δ_t = r_t + γ*V(t+1) - V(t)
        T = min(len(values) - 1, len(rewards_arr))
        td_errors = np.zeros(T)
        for t in range(T):
            next_v = values[t + 1] if t + 1 < len(values) else 0
            r = rewards_arr[t] if t < len(rewards_arr) else 0
            td_errors[t] = r + gamma * next_v - values[t]

        if len(td_errors) < 10 or np.std(td_errors) < 1e-8:
            depths[i] = 0.0
            continue

        # Autocorrelation decay
        td_centered = td_errors - td_errors.mean()
        autocorr = np.correlate(td_centered, td_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # Find decay to 1/e
        threshold = 1.0 / np.e
        depth = 0
        for k in range(len(autocorr)):
            if autocorr[k] < threshold:
                depth = k
                break
        else:
            depth = len(autocorr)

        depths[i] = float(depth)

    return depths


def compute_coordination_efficiency(positions_log: List[np.ndarray],
                                    action_frequencies: List[int],
                                    threshold: float = 2.0) -> float:
    """
    Coordination efficiency: fraction of timesteps where agents are within
    threshold distance, weighted by their action frequencies.
    """
    if not positions_log:
        return 0.0

    n_agents = len(positions_log[0])
    synchronized = 0
    total = 0

    for positions in positions_log:
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    synchronized += 1
                total += 1

    return synchronized / max(total, 1)


def compute_agent_influence(value_logs: Dict[int, np.ndarray],
                            window: int = 10) -> Dict[str, float]:
    """
    Measure how much each agent's value function changes predict changes
    in other agents' value functions (Granger-like causality proxy).
    """
    n_agents = len(value_logs)
    influences = {}

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue

            v_i = value_logs[i]
            v_j = value_logs[j]
            min_len = min(len(v_i), len(v_j))

            if min_len < window + 5:
                influences[f"{i}->{j}"] = 0.0
                continue

            # Use correlation between V_i changes and subsequent V_j changes
            dv_i = np.diff(v_i[:min_len])
            dv_j = np.diff(v_j[:min_len])

            # Lagged correlation (i's changes predict j's future changes)
            corrs = []
            for lag in range(1, window + 1):
                if lag >= len(dv_i):
                    break
                c, _ = stats.pearsonr(dv_i[:len(dv_i) - lag], dv_j[lag:len(dv_j)])
                if not np.isnan(c):
                    corrs.append(abs(c))

            influences[f"{i}->{j}"] = float(np.mean(corrs)) if corrs else 0.0

    return influences


def run_statistical_tests(results_homo: List[float], results_hetero: List[float],
                          metric_name: str = "reward") -> Dict:
    """Run statistical comparison between homogeneous and heterogeneous conditions."""
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(results_hetero, results_homo, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(results_hetero, ddof=1) ** 2 + np.std(results_homo, ddof=1) ** 2) / 2
    )
    cohens_d = (np.mean(results_hetero) - np.mean(results_homo)) / (pooled_std + 1e-8)

    # Bootstrap CI for difference
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_bootstrap):
        h_sample = rng.choice(results_hetero, len(results_hetero), replace=True)
        o_sample = rng.choice(results_homo, len(results_homo), replace=True)
        diffs.append(np.mean(h_sample) - np.mean(o_sample))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return {
        'metric': metric_name,
        'homo_mean': float(np.mean(results_homo)),
        'homo_std': float(np.std(results_homo)),
        'hetero_mean': float(np.mean(results_hetero)),
        'hetero_std': float(np.std(results_hetero)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'ci_95': [float(ci_low), float(ci_high)],
        'significant': p_value < 0.05
    }
