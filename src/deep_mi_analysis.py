"""
Deep mutual information analysis comparing heterogeneous vs homogeneous conditions.
Focuses on cross-timescale MI patterns that indicate emergent credit assignment.
"""

import os
import sys
import json
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from environments import make_env
from agents import IndependentPPO
from analysis import compute_mutual_information, compute_mi_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_value_logs(env_name, heterogeneous, use_temporal_obs, seed, n_train=300, n_eval=50):
    """Train and collect value function logs for MI analysis."""
    set_seed(seed)
    env = make_env(env_name, heterogeneous=heterogeneous, seed=seed)
    obs = env.reset()
    obs_size = len(obs[0])

    ppo = IndependentPPO(
        n_agents=env.n_agents, obs_size=obs_size, n_actions=env.n_actions,
        lr=3e-4, gamma=0.99, epochs=4, batch_size=64, device='cpu'
    )

    # Train
    for ep in range(n_train):
        obs = env.reset()
        for step in range(env.max_steps):
            actions, log_probs, values = ppo.get_actions(obs)
            next_obs, reward, done, info = env.step(actions, use_temporal_obs=use_temporal_obs)
            ppo.store_transition(obs, actions, log_probs, reward, values, done)
            obs = next_obs
            if done:
                break
        if (ep + 1) % 5 == 0:
            ppo.update()

    # Collect evaluation value logs
    ppo.clear_value_logs()
    for ep in range(n_eval):
        env_eval = make_env(env_name, heterogeneous=heterogeneous, seed=seed + 5000 + ep)
        obs = env_eval.reset()
        for step in range(env_eval.max_steps):
            actions, log_probs, values = ppo.get_actions(obs)
            next_obs, reward, done, info = env_eval.step(actions, use_temporal_obs=use_temporal_obs)
            obs = next_obs
            if done:
                break

    return ppo.get_value_logs()


def compute_cross_timescale_mi_summary(mi_matrix, n_agents=3):
    """Compute summary stats about cross-agent MI patterns."""
    # Self MI (autocorrelation of value function)
    self_mi = {}
    cross_mi = {}

    for pair, lags in mi_matrix.items():
        i, j = int(pair[0]), int(pair[-1])
        if i == j:
            self_mi[i] = lags
        else:
            cross_mi[pair] = lags

    # Key metrics:
    # 1. Total cross-agent MI at lag 0 (instantaneous coupling)
    total_cross_mi_lag0 = np.mean([v.get(0, 0) for v in cross_mi.values()])

    # 2. MI decay rate (how quickly MI drops with lag)
    decay_rates = []
    for pair, lags in cross_mi.items():
        if 0 in lags and 5 in lags and lags[0] > 0:
            decay_rates.append(lags[5] / lags[0])
    avg_decay_rate = np.mean(decay_rates) if decay_rates else 0

    # 3. Asymmetric MI: is MI(fast->slow) different from MI(slow->fast)?
    # In heterogeneous settings, we expect slow agents to have more info about fast agents
    forward_mi = {}  # fast -> slow direction
    backward_mi = {}  # slow -> fast direction
    for pair, lags in cross_mi.items():
        i, j = int(pair[0]), int(pair[-1])
        if i < j:  # i is faster (lower index = higher frequency)
            forward_mi[pair] = lags.get(0, 0)
        else:
            backward_mi[pair] = lags.get(0, 0)

    avg_forward = np.mean(list(forward_mi.values())) if forward_mi else 0
    avg_backward = np.mean(list(backward_mi.values())) if backward_mi else 0

    # 4. Peak lag for cross-agent MI (at which lag is MI highest for cross-pairs?)
    peak_lags = []
    for pair, lags in cross_mi.items():
        if lags:
            peak_lag = max(lags, key=lags.get)
            peak_lags.append(peak_lag)
    avg_peak_lag = np.mean(peak_lags) if peak_lags else 0

    return {
        'total_cross_mi_lag0': float(total_cross_mi_lag0),
        'avg_decay_rate': float(avg_decay_rate),
        'avg_forward_mi': float(avg_forward),
        'avg_backward_mi': float(avg_backward),
        'mi_asymmetry': float(avg_backward - avg_forward),
        'avg_peak_lag': float(avg_peak_lag),
    }


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    plots_dir = os.path.join(results_dir, 'plots')

    print("Deep MI Analysis")
    print("=" * 60)

    SEEDS = [42, 123, 456]
    ENV_NAMES = ['relay', 'foraging', 'rendezvous']

    all_mi_results = {}

    for env_name in ENV_NAMES:
        print(f"\n--- {env_name} ---")
        env_mi = {'heterogeneous': [], 'homogeneous': [], 'ablation': []}

        for seed in SEEDS:
            print(f"  Seed {seed}:")

            # Heterogeneous
            vl = collect_value_logs(env_name, True, True, seed)
            mi = compute_mi_matrix(vl, max_lag=20)
            summary = compute_cross_timescale_mi_summary(mi)
            env_mi['heterogeneous'].append(summary)
            print(f"    Hetero: cross_MI_lag0={summary['total_cross_mi_lag0']:.3f}, "
                  f"asymmetry={summary['mi_asymmetry']:.4f}")

            # Homogeneous
            vl = collect_value_logs(env_name, False, True, seed)
            mi = compute_mi_matrix(vl, max_lag=20)
            summary = compute_cross_timescale_mi_summary(mi)
            env_mi['homogeneous'].append(summary)
            print(f"    Homo:   cross_MI_lag0={summary['total_cross_mi_lag0']:.3f}, "
                  f"asymmetry={summary['mi_asymmetry']:.4f}")

            # Ablation
            vl = collect_value_logs(env_name, True, False, seed)
            mi = compute_mi_matrix(vl, max_lag=20)
            summary = compute_cross_timescale_mi_summary(mi)
            env_mi['ablation'].append(summary)
            print(f"    Ablat:  cross_MI_lag0={summary['total_cross_mi_lag0']:.3f}, "
                  f"asymmetry={summary['mi_asymmetry']:.4f}")

        all_mi_results[env_name] = env_mi

    # Generate comparative MI plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, env_name in enumerate(ENV_NAMES):
        ax = axes[idx]
        data = all_mi_results[env_name]

        metrics = ['total_cross_mi_lag0', 'avg_decay_rate', 'mi_asymmetry']
        labels = ['Cross MI\n(lag=0)', 'MI Decay\nRate', 'MI\nAsymmetry']
        x = np.arange(len(metrics))
        width = 0.25

        for ci, (cond, color) in enumerate([('heterogeneous', 'steelblue'),
                                              ('homogeneous', 'orange'),
                                              ('ablation', 'green')]):
            means = [np.mean([d[m] for d in data[cond]]) for m in metrics]
            stds = [np.std([d[m] for d in data[cond]]) for m in metrics]
            ax.bar(x + ci * width, means, width, yerr=stds, label=cond,
                   color=color, alpha=0.8, capsize=3)

        ax.set_xlabel('Metric')
        ax.set_title(env_name.capitalize())
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Timescale Mutual Information Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mi_comparison_summary.png'), dpi=150)
    plt.close()

    # Save detailed MI results
    serializable = {}
    for env_name, data in all_mi_results.items():
        serializable[env_name] = {}
        for cond, summaries in data.items():
            avg = {k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]}
            std = {f"{k}_std": float(np.std([s[k] for s in summaries])) for k in summaries[0]}
            serializable[env_name][cond] = {**avg, **std}

    with open(os.path.join(results_dir, 'mi_analysis.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    print("\n\nSummary of MI Analysis:")
    print("=" * 60)
    for env_name in ENV_NAMES:
        print(f"\n{env_name.upper()}:")
        for cond in ['heterogeneous', 'homogeneous', 'ablation']:
            d = serializable[env_name][cond]
            print(f"  {cond:20s}: MI_lag0={d['total_cross_mi_lag0']:.3f}, "
                  f"decay={d['avg_decay_rate']:.3f}, "
                  f"asymmetry={d['mi_asymmetry']:.4f}")

    print(f"\nPlots saved to {plots_dir}/mi_comparison_summary.png")
    print(f"Data saved to {results_dir}/mi_analysis.json")


if __name__ == '__main__':
    main()
