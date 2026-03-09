"""
Main experiment runner for Emergent Temporal Credit Assignment study.

Experiments:
1. Performance comparison: homogeneous vs heterogeneous agent horizons
2. Mutual information analysis between agent value functions
3. Ablation: with vs without temporal context features
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from environments import make_env
from agents import IndependentPPO
from analysis import (
    compute_mi_matrix, compute_td_error_depth,
    compute_coordination_efficiency, compute_agent_influence,
    run_statistical_tests
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_episode(env, ppo, use_temporal_obs=True):
    """Run one training episode, return reward and info."""
    obs = env.reset()
    total_reward = 0
    positions_log = []
    rewards_log = []

    for step in range(env.max_steps):
        actions, log_probs, values = ppo.get_actions(obs)
        next_obs, reward, done, info = env.step(actions, use_temporal_obs=use_temporal_obs)
        ppo.store_transition(obs, actions, log_probs, reward, values, done)
        total_reward += reward
        rewards_log.append(reward)
        if 'positions' in info:
            positions_log.append(info['positions'])

        obs = next_obs
        if done:
            break

    return total_reward, rewards_log, positions_log, info


def run_condition(env_name, heterogeneous, use_temporal_obs, n_episodes, seed,
                  label=""):
    """Run a full training condition and return metrics."""
    set_seed(seed)
    env = make_env(env_name, heterogeneous=heterogeneous, seed=seed)

    # Determine obs size from environment
    obs = env.reset()
    obs_size = len(obs[0])

    ppo = IndependentPPO(
        n_agents=env.n_agents,
        obs_size=obs_size,
        n_actions=env.n_actions,
        lr=3e-4,
        gamma=0.99,
        epochs=4,
        batch_size=64,
        device='cpu'  # Gridworld is too light for GPU overhead
    )

    episode_rewards = []
    episode_infos = []
    all_positions = []
    all_rewards = []
    update_interval = 5  # Update PPO every N episodes

    for ep in range(n_episodes):
        reward, rewards_log, positions_log, info = train_episode(
            env, ppo, use_temporal_obs=use_temporal_obs
        )
        episode_rewards.append(reward)
        episode_infos.append(info)
        all_positions.extend(positions_log)
        all_rewards.extend(rewards_log)

        if (ep + 1) % update_interval == 0:
            loss = ppo.update()

        if (ep + 1) % 50 == 0:
            mean_r = np.mean(episode_rewards[-50:])
            print(f"  [{label}] Episode {ep+1}/{n_episodes}, "
                  f"Mean Reward (last 50): {mean_r:.3f}")

    # Collect value logs from final episodes for MI analysis
    ppo.clear_value_logs()
    eval_rewards = []
    for ep in range(20):  # 20 eval episodes
        env_eval = make_env(env_name, heterogeneous=heterogeneous, seed=seed + 1000 + ep)
        r, rl, pl, info = train_episode(env_eval, ppo, use_temporal_obs=use_temporal_obs)
        eval_rewards.append(r)

    value_logs = ppo.get_value_logs()

    # Compute metrics
    mi_matrix = compute_mi_matrix(value_logs, max_lag=20)
    td_depths = compute_td_error_depth(value_logs, all_rewards[-200:])
    coord_eff = compute_coordination_efficiency(
        all_positions[-200:] if all_positions else [],
        env.action_frequencies
    )
    influences = compute_agent_influence(value_logs)

    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'mi_matrix': mi_matrix,
        'td_depths': td_depths,
        'coordination_efficiency': coord_eff,
        'influences': influences,
        'value_logs': {str(k): v.tolist() for k, v in value_logs.items()},
        'final_infos': [str(info) for info in episode_infos[-5:]],
    }


def plot_learning_curves(results, env_name, save_dir):
    """Plot learning curves comparing conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for condition, data in results.items():
        rewards_by_seed = data['all_episode_rewards']
        # Average across seeds, smooth with window
        all_rewards = np.array(rewards_by_seed)
        mean = all_rewards.mean(axis=0)
        std = all_rewards.std(axis=0)

        # Smooth
        window = 10
        if len(mean) > window:
            mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
            x = np.arange(len(mean_smooth))
            ax.plot(x, mean_smooth, label=condition)
            ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.2)
        else:
            ax.plot(mean, label=condition)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title(f'Learning Curves - {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'learning_curves_{env_name}.png'), dpi=150)
    plt.close()


def plot_mi_heatmap(mi_matrix, env_name, condition, save_dir):
    """Plot MI matrix as heatmap across agents and time lags."""
    # Organize data
    pairs = sorted(mi_matrix.keys())
    max_lag = max(max(v.keys()) for v in mi_matrix.values() if v)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]

    for idx, (pair, ax) in enumerate(zip(pairs, axes)):
        lags = sorted(mi_matrix[pair].keys())
        mi_values = [mi_matrix[pair][l] for l in lags]
        ax.bar(lags, mi_values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Time Lag τ')
        ax.set_ylabel('MI (bits)')
        ax.set_title(f'{pair}')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Mutual Information - {env_name} ({condition})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mi_{env_name}_{condition}.png'), dpi=150)
    plt.close()


def plot_td_depths(all_depths, env_name, save_dir):
    """Plot TD-error propagation depth comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = list(all_depths.keys())
    n_agents = len(list(all_depths.values())[0])
    x = np.arange(n_agents)
    width = 0.25

    for i, cond in enumerate(conditions):
        depths = all_depths[cond]
        agents = sorted(depths.keys())
        vals = [depths[a] for a in agents]
        ax.bar(x + i * width, vals, width, label=cond)

    ax.set_xlabel('Agent (0=fast, 1=medium, 2=slow)')
    ax.set_ylabel('TD-Error Propagation Depth')
    ax.set_title(f'TD-Error Depth by Agent Type - {env_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Agent {a}\n({["1x","2x","4x"][a]})' for a in range(n_agents)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'td_depths_{env_name}.png'), dpi=150)
    plt.close()


def plot_influence_comparison(all_influences, env_name, save_dir):
    """Plot agent influence scores comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = list(all_influences.keys())
    pairs = sorted(list(all_influences.values())[0].keys())
    x = np.arange(len(pairs))
    width = 0.35

    for i, cond in enumerate(conditions):
        vals = [all_influences[cond].get(p, 0) for p in pairs]
        ax.bar(x + i * width, vals, width, label=cond)

    ax.set_xlabel('Agent Pair (i→j)')
    ax.set_ylabel('Influence Score')
    ax.set_title(f'Agent Influence Scores - {env_name}')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(pairs, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'influences_{env_name}.png'), dpi=150)
    plt.close()


def main():
    start_time = time.time()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("Emergent Temporal Credit Assignment Experiments")
    print("=" * 70)
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    print(f"Device: CPU (gridworld environments)")
    print()

    # Configuration
    SEEDS = [42, 123, 456, 789, 1024]
    N_EPISODES = 300
    ENV_NAMES = ['relay', 'foraging', 'rendezvous']

    all_results = {}

    for env_name in ENV_NAMES:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        env_results = {}

        # === Condition 1: Heterogeneous (with temporal features) ===
        print(f"\n--- Condition: Heterogeneous (with temporal context) ---")
        hetero_rewards_all = []
        hetero_mi_all = []
        hetero_td_all = []
        hetero_influence_all = []
        hetero_coord_all = []

        for seed in SEEDS:
            print(f"  Seed {seed}:")
            result = run_condition(
                env_name, heterogeneous=True, use_temporal_obs=True,
                n_episodes=N_EPISODES, seed=seed, label=f"hetero-{seed}"
            )
            hetero_rewards_all.append(result['episode_rewards'])
            hetero_mi_all.append(result['mi_matrix'])
            hetero_td_all.append(result['td_depths'])
            hetero_influence_all.append(result['influences'])
            hetero_coord_all.append(result['coordination_efficiency'])

        env_results['heterogeneous'] = {
            'all_episode_rewards': hetero_rewards_all,
            'eval_rewards': [r[-1] for r in hetero_rewards_all],  # last episode
            'mi_matrices': hetero_mi_all,
            'td_depths': hetero_td_all,
            'influences': hetero_influence_all,
            'coordination': hetero_coord_all,
        }

        # === Condition 2: Homogeneous (with temporal features) ===
        print(f"\n--- Condition: Homogeneous (all same frequency) ---")
        homo_rewards_all = []
        homo_mi_all = []
        homo_td_all = []
        homo_influence_all = []
        homo_coord_all = []

        for seed in SEEDS:
            print(f"  Seed {seed}:")
            result = run_condition(
                env_name, heterogeneous=False, use_temporal_obs=True,
                n_episodes=N_EPISODES, seed=seed, label=f"homo-{seed}"
            )
            homo_rewards_all.append(result['episode_rewards'])
            homo_mi_all.append(result['mi_matrix'])
            homo_td_all.append(result['td_depths'])
            homo_influence_all.append(result['influences'])
            homo_coord_all.append(result['coordination_efficiency'])

        env_results['homogeneous'] = {
            'all_episode_rewards': homo_rewards_all,
            'eval_rewards': [r[-1] for r in homo_rewards_all],
            'mi_matrices': homo_mi_all,
            'td_depths': homo_td_all,
            'influences': homo_influence_all,
            'coordination': homo_coord_all,
        }

        # === Condition 3: Heterogeneous WITHOUT temporal features (ablation) ===
        print(f"\n--- Condition: Heterogeneous (NO temporal context - ablation) ---")
        ablation_rewards_all = []
        ablation_mi_all = []
        ablation_td_all = []
        ablation_coord_all = []

        for seed in SEEDS:
            print(f"  Seed {seed}:")
            result = run_condition(
                env_name, heterogeneous=True, use_temporal_obs=False,
                n_episodes=N_EPISODES, seed=seed, label=f"ablation-{seed}"
            )
            ablation_rewards_all.append(result['episode_rewards'])
            ablation_mi_all.append(result['mi_matrix'])
            ablation_td_all.append(result['td_depths'])
            ablation_coord_all.append(result['coordination_efficiency'])

        env_results['ablation_no_temporal'] = {
            'all_episode_rewards': ablation_rewards_all,
            'eval_rewards': [r[-1] for r in ablation_rewards_all],
            'mi_matrices': ablation_mi_all,
            'td_depths': ablation_td_all,
            'coordination': ablation_coord_all,
        }

        # === Generate Plots ===
        print(f"\nGenerating plots for {env_name}...")

        # Learning curves
        plot_learning_curves(env_results, env_name, plots_dir)

        # MI heatmaps (use first seed for visualization)
        plot_mi_heatmap(hetero_mi_all[0], env_name, 'heterogeneous', plots_dir)
        plot_mi_heatmap(homo_mi_all[0], env_name, 'homogeneous', plots_dir)

        # TD depths (average across seeds)
        avg_td_hetero = {}
        avg_td_homo = {}
        for agent_id in range(3):
            avg_td_hetero[agent_id] = np.mean([d[agent_id] for d in hetero_td_all])
            avg_td_homo[agent_id] = np.mean([d[agent_id] for d in homo_td_all])
        plot_td_depths({'heterogeneous': avg_td_hetero, 'homogeneous': avg_td_homo},
                       env_name, plots_dir)

        # Influence comparison (average across seeds)
        avg_inf_hetero = {}
        avg_inf_homo = {}
        all_pairs = set()
        for inf_dict in hetero_influence_all:
            all_pairs.update(inf_dict.keys())
        for pair in all_pairs:
            avg_inf_hetero[pair] = np.mean([d.get(pair, 0) for d in hetero_influence_all])
            avg_inf_homo[pair] = np.mean([d.get(pair, 0) for d in homo_influence_all])
        plot_influence_comparison(
            {'heterogeneous': avg_inf_hetero, 'homogeneous': avg_inf_homo},
            env_name, plots_dir
        )

        # === Statistical Tests ===
        print(f"\nStatistical analysis for {env_name}:")

        # Compare final 50 episodes mean reward
        hetero_final = [np.mean(r[-50:]) for r in hetero_rewards_all]
        homo_final = [np.mean(r[-50:]) for r in homo_rewards_all]
        ablation_final = [np.mean(r[-50:]) for r in ablation_rewards_all]

        stats_hetero_vs_homo = run_statistical_tests(
            homo_final, hetero_final, 'reward_hetero_vs_homo'
        )
        stats_hetero_vs_ablation = run_statistical_tests(
            ablation_final, hetero_final, 'reward_hetero_vs_ablation'
        )

        print(f"  Hetero vs Homo: p={stats_hetero_vs_homo['p_value']:.4f}, "
              f"d={stats_hetero_vs_homo['cohens_d']:.3f}")
        print(f"  Hetero vs Ablation: p={stats_hetero_vs_ablation['p_value']:.4f}, "
              f"d={stats_hetero_vs_ablation['cohens_d']:.3f}")

        env_results['stats'] = {
            'hetero_vs_homo': stats_hetero_vs_homo,
            'hetero_vs_ablation': stats_hetero_vs_ablation,
        }

        # === MI Summary Statistics ===
        # Average MI at lag 0 and lag 5 across agent pairs
        mi_summary = {}
        for cond_name, mi_list in [('hetero', hetero_mi_all),
                                     ('homo', homo_mi_all),
                                     ('ablation', ablation_mi_all)]:
            cross_type_mi = []  # MI between agents of different frequencies
            same_type_mi = []   # MI between agents of same frequency

            for mi_mat in mi_list:
                for pair, lags in mi_mat.items():
                    i, j = int(pair[0]), int(pair[-1])
                    if i == j:
                        continue
                    lag0_mi = lags.get(0, 0)
                    lag5_mi = lags.get(5, 0)

                    if cond_name == 'hetero':
                        # In heterogeneous, all agents are different type
                        cross_type_mi.append((lag0_mi, lag5_mi))
                    else:
                        same_type_mi.append((lag0_mi, lag5_mi))
                        cross_type_mi.append((lag0_mi, lag5_mi))

            mi_summary[cond_name] = {
                'cross_type_mi_lag0': float(np.mean([m[0] for m in cross_type_mi])) if cross_type_mi else 0,
                'cross_type_mi_lag5': float(np.mean([m[1] for m in cross_type_mi])) if cross_type_mi else 0,
            }

        env_results['mi_summary'] = mi_summary

        # Coordination efficiency
        env_results['coord_summary'] = {
            'heterogeneous': float(np.mean(hetero_coord_all)),
            'homogeneous': float(np.mean(homo_coord_all)),
            'ablation': float(np.mean(ablation_coord_all)),
        }

        all_results[env_name] = env_results

    # === Save All Results ===
    # Convert numpy arrays for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    # Save summary metrics (not full reward curves to keep file small)
    summary = {}
    for env_name in ENV_NAMES:
        env_data = all_results[env_name]
        summary[env_name] = {
            'stats': env_data.get('stats', {}),
            'mi_summary': env_data.get('mi_summary', {}),
            'coord_summary': env_data.get('coord_summary', {}),
            'td_depths': {
                'heterogeneous': {str(k): float(np.mean([d[k] for d in env_data['heterogeneous']['td_depths']]))
                                  for k in range(3)},
                'homogeneous': {str(k): float(np.mean([d[k] for d in env_data['homogeneous']['td_depths']]))
                                for k in range(3)},
            },
            'final_rewards': {
                'heterogeneous': {
                    'mean': float(np.mean([np.mean(r[-50:]) for r in env_data['heterogeneous']['all_episode_rewards']])),
                    'std': float(np.std([np.mean(r[-50:]) for r in env_data['heterogeneous']['all_episode_rewards']])),
                },
                'homogeneous': {
                    'mean': float(np.mean([np.mean(r[-50:]) for r in env_data['homogeneous']['all_episode_rewards']])),
                    'std': float(np.std([np.mean(r[-50:]) for r in env_data['homogeneous']['all_episode_rewards']])),
                },
                'ablation': {
                    'mean': float(np.mean([np.mean(r[-50:]) for r in env_data['ablation_no_temporal']['all_episode_rewards']])),
                    'std': float(np.std([np.mean(r[-50:]) for r in env_data['ablation_no_temporal']['all_episode_rewards']])),
                },
            },
        }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(make_serializable(summary), f, indent=2)

    # Save full learning curves for plotting
    curves = {}
    for env_name in ENV_NAMES:
        curves[env_name] = {}
        for cond in ['heterogeneous', 'homogeneous', 'ablation_no_temporal']:
            curves[env_name][cond] = make_serializable(
                all_results[env_name][cond]['all_episode_rewards']
            )

    with open(os.path.join(results_dir, 'learning_curves.json'), 'w') as f:
        json.dump(curves, f)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"All experiments completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {results_dir}/")
    print(f"Plots saved to {plots_dir}/")
    print(f"{'='*70}")

    return summary


if __name__ == '__main__':
    summary = main()
