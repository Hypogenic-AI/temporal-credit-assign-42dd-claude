"""
Custom cooperative gridworld environments with heterogeneous agent decision frequencies.

Three environments test different aspects of temporal coordination:
1. CooperativeRelay - Sequential item passing requiring temporal coordination
2. MultiPaceForaging - Resource collection with different agent speeds
3. SynchronizedRendezvous - Meeting point coordination with speed differences
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class HeterogeneousGridworld:
    """
    Base class for gridworld environments where agents have different action frequencies.

    Agents act at multiples of a base frequency:
    - Agent 0: acts every step (1x, "fast")
    - Agent 1: acts every 2 steps (2x, "medium")
    - Agent 2: acts every 4 steps (4x, "slow")

    When an agent doesn't act, it repeats its last action (or stays still).
    """

    def __init__(self, grid_size=8, n_agents=3, action_frequencies=None,
                 feedback_delays=None, max_steps=200, seed=42):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.action_frequencies = action_frequencies or [1, 2, 4]
        self.feedback_delays = feedback_delays or [0, 0, 0]
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.n_actions = 5
        self.action_deltas = {
            0: (0, 0),   # stay
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1),   # right
        }

        # State tracking
        self.positions = None
        self.step_count = 0
        self.last_actions = [0] * n_agents
        self.reward_buffers = [[] for _ in range(n_agents)]  # For delayed feedback

    def _get_obs_size(self):
        """Size of observation vector per agent."""
        # Own position (2) + other agent positions (2*(n-1)) + temporal features (3)
        # temporal features: step_count_normalized, agent_frequency, time_since_last_action
        return 2 + 2 * (self.n_agents - 1) + 3

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        obs = []
        # Own position (normalized)
        obs.extend(self.positions[agent_id] / self.grid_size)
        # Other agents' positions
        for j in range(self.n_agents):
            if j != agent_id:
                obs.extend(self.positions[j] / self.grid_size)
        # Temporal context features
        obs.append(self.step_count / self.max_steps)  # normalized timestep
        obs.append(self.action_frequencies[agent_id] / max(self.action_frequencies))  # normalized frequency
        obs.append((self.step_count % self.action_frequencies[agent_id]) / self.action_frequencies[agent_id])
        return np.array(obs, dtype=np.float32)

    def _get_obs_no_temporal(self, agent_id: int) -> np.ndarray:
        """Get observation WITHOUT temporal context features (for ablation)."""
        obs = []
        obs.extend(self.positions[agent_id] / self.grid_size)
        for j in range(self.n_agents):
            if j != agent_id:
                obs.extend(self.positions[j] / self.grid_size)
        # Replace temporal features with zeros
        obs.extend([0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def _move_agent(self, agent_id: int, action: int):
        """Move agent according to action, clipping to grid boundaries."""
        dy, dx = self.action_deltas[action]
        new_y = np.clip(self.positions[agent_id][0] + dy, 0, self.grid_size - 1)
        new_x = np.clip(self.positions[agent_id][1] + dx, 0, self.grid_size - 1)
        self.positions[agent_id] = np.array([new_y, new_x], dtype=np.float32)

    def can_act(self, agent_id: int) -> bool:
        """Check if agent can act at current timestep."""
        return self.step_count % self.action_frequencies[agent_id] == 0

    def reset(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, float, bool, dict]:
        raise NotImplementedError


class CooperativeRelay(HeterogeneousGridworld):
    """
    Cooperative relay task: agents must sequentially visit waypoints in order.
    Agent 0 (fast) visits waypoint 0, then agent 1 (medium) visits waypoint 1, etc.
    Team reward based on number of waypoints completed within time limit.
    Tests: temporal coordination across different decision frequencies.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_waypoints = self.n_agents
        self.waypoints = None
        self.waypoint_reached = None
        self.current_waypoint = 0

    def reset(self):
        self.step_count = 0
        self.last_actions = [0] * self.n_agents
        # Random starting positions
        self.positions = np.array([
            self.rng.randint(0, self.grid_size, 2).astype(np.float32)
            for _ in range(self.n_agents)
        ])
        # Random waypoint positions
        self.waypoints = np.array([
            self.rng.randint(1, self.grid_size - 1, 2).astype(np.float32)
            for _ in range(self.n_waypoints)
        ])
        self.waypoint_reached = [False] * self.n_waypoints
        self.current_waypoint = 0

        return {i: self._get_obs(i) for i in range(self.n_agents)}

    def step(self, actions, use_temporal_obs=True):
        self.step_count += 1
        reward = -0.01  # small time penalty

        # Move agents that can act
        for i in range(self.n_agents):
            if self.can_act(i):
                action = actions.get(i, 0)
                self.last_actions[i] = action
                self._move_agent(i, action)
            else:
                # Repeat last action or stay
                pass

        # Check waypoint completion (sequential)
        if self.current_waypoint < self.n_waypoints:
            agent_id = self.current_waypoint  # Agent i is responsible for waypoint i
            dist = np.linalg.norm(self.positions[agent_id] - self.waypoints[self.current_waypoint])
            if dist < 1.5:
                self.waypoint_reached[self.current_waypoint] = True
                reward += 1.0
                self.current_waypoint += 1

        # Bonus for completing all waypoints
        done = self.step_count >= self.max_steps
        if self.current_waypoint >= self.n_waypoints:
            reward += 5.0
            done = True

        obs_fn = self._get_obs if use_temporal_obs else self._get_obs_no_temporal
        obs = {i: obs_fn(i) for i in range(self.n_agents)}
        info = {
            'waypoints_completed': self.current_waypoint,
            'positions': self.positions.copy()
        }

        return obs, reward, done, info


class MultiPaceForaging(HeterogeneousGridworld):
    """
    Multi-pace foraging: resources spawn at fixed locations. Fast agents can scout,
    slow agents can collect (more reward per collection). Requires coordination.
    Tests: emergent role specialization across timescales.
    """

    def __init__(self, n_resources=5, **kwargs):
        super().__init__(**kwargs)
        self.n_resources = n_resources
        self.resources = None
        self.resource_active = None
        self.resource_values = None
        self.total_collected = 0

    def _get_obs_size(self):
        base = super()._get_obs_size()
        return base + self.n_resources * 3  # resource positions + active flags

    def _get_obs(self, agent_id):
        base_obs = super()._get_obs(agent_id)
        resource_obs = []
        for r in range(self.n_resources):
            resource_obs.extend(self.resources[r] / self.grid_size)
            resource_obs.append(float(self.resource_active[r]))
        return np.concatenate([base_obs, np.array(resource_obs, dtype=np.float32)])

    def _get_obs_no_temporal(self, agent_id):
        base_obs = super()._get_obs_no_temporal(agent_id)
        resource_obs = []
        for r in range(self.n_resources):
            resource_obs.extend(self.resources[r] / self.grid_size)
            resource_obs.append(float(self.resource_active[r]))
        return np.concatenate([base_obs, np.array(resource_obs, dtype=np.float32)])

    def reset(self):
        self.step_count = 0
        self.last_actions = [0] * self.n_agents
        self.total_collected = 0

        self.positions = np.array([
            self.rng.randint(0, self.grid_size, 2).astype(np.float32)
            for _ in range(self.n_agents)
        ])
        self.resources = np.array([
            self.rng.randint(0, self.grid_size, 2).astype(np.float32)
            for _ in range(self.n_resources)
        ])
        self.resource_active = [True] * self.n_resources
        # Slow agents get more reward per collection (incentivizes coordination)
        self.resource_values = np.ones(self.n_resources, dtype=np.float32)

        return {i: self._get_obs(i) for i in range(self.n_agents)}

    def step(self, actions, use_temporal_obs=True):
        self.step_count += 1
        reward = -0.01

        for i in range(self.n_agents):
            if self.can_act(i):
                action = actions.get(i, 0)
                self.last_actions[i] = action
                self._move_agent(i, action)

        # Check resource collection
        for r in range(self.n_resources):
            if not self.resource_active[r]:
                continue
            for i in range(self.n_agents):
                dist = np.linalg.norm(self.positions[i] - self.resources[r])
                if dist < 1.5:
                    # Slower agents get more reward (incentivizes role specialization)
                    collection_bonus = self.action_frequencies[i] * 0.5
                    reward += collection_bonus
                    self.resource_active[r] = False
                    self.total_collected += 1
                    break

        # Respawn resources occasionally
        if self.step_count % 20 == 0:
            for r in range(self.n_resources):
                if not self.resource_active[r]:
                    self.resources[r] = self.rng.randint(0, self.grid_size, 2).astype(np.float32)
                    self.resource_active[r] = True

        done = self.step_count >= self.max_steps
        obs_fn = self._get_obs if use_temporal_obs else self._get_obs_no_temporal
        obs = {i: obs_fn(i) for i in range(self.n_agents)}
        info = {'total_collected': self.total_collected}

        return obs, reward, done, info


class SynchronizedRendezvous(HeterogeneousGridworld):
    """
    All agents must converge to a rendezvous point. Reward based on how close
    all agents are to each other at the end. Tests temporal coordination when
    agents move at different effective speeds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = None

    def reset(self):
        self.step_count = 0
        self.last_actions = [0] * self.n_agents

        # Spread agents across the grid
        self.positions = np.array([
            self.rng.randint(0, self.grid_size, 2).astype(np.float32)
            for _ in range(self.n_agents)
        ])
        # Target is center of grid
        self.target = np.array([self.grid_size / 2, self.grid_size / 2], dtype=np.float32)

        return {i: self._get_obs(i) for i in range(self.n_agents)}

    def step(self, actions, use_temporal_obs=True):
        self.step_count += 1

        for i in range(self.n_agents):
            if self.can_act(i):
                action = actions.get(i, 0)
                self.last_actions[i] = action
                self._move_agent(i, action)

        # Reward: negative mean pairwise distance between agents + distance to target
        total_dist = 0
        count = 0
        for i in range(self.n_agents):
            total_dist += np.linalg.norm(self.positions[i] - self.target)
            for j in range(i + 1, self.n_agents):
                total_dist += np.linalg.norm(self.positions[i] - self.positions[j])
                count += 1

        reward = -0.01 * total_dist / (self.n_agents + count)

        # Bonus if all agents within 2 cells of each other
        max_pair_dist = max(
            np.linalg.norm(self.positions[i] - self.positions[j])
            for i in range(self.n_agents) for j in range(i + 1, self.n_agents)
        )
        if max_pair_dist < 2.0:
            reward += 1.0

        done = self.step_count >= self.max_steps
        obs_fn = self._get_obs if use_temporal_obs else self._get_obs_no_temporal
        obs = {i: obs_fn(i) for i in range(self.n_agents)}
        info = {'max_pair_dist': max_pair_dist, 'mean_dist_to_target': total_dist / self.n_agents}

        return obs, reward, done, info


def make_env(env_name, heterogeneous=True, seed=42, **kwargs):
    """Factory function to create environments."""
    freqs = [1, 2, 4] if heterogeneous else [1, 1, 1]

    if env_name == 'relay':
        return CooperativeRelay(action_frequencies=freqs, seed=seed, **kwargs)
    elif env_name == 'foraging':
        return MultiPaceForaging(action_frequencies=freqs, seed=seed, **kwargs)
    elif env_name == 'rendezvous':
        return SynchronizedRendezvous(action_frequencies=freqs, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
