"""
Independent PPO agents for heterogeneous-horizon MARL experiments.

Each agent has its own actor-critic network. Value functions are logged
for post-hoc mutual information analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class ActorCritic(nn.Module):
    """Simple actor-critic network for a single agent."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        return action, log_prob, value.squeeze(-1)


class PPOBuffer:
    """Rollout buffer for PPO."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns."""
        T = len(self.rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1].detach()
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t].detach()
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t].detach()

        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        return obs, actions, old_log_probs, returns, advantages, values

    def clear(self):
        self.__init__()


class IndependentPPO:
    """
    Independent PPO trainer for multiple agents.
    Each agent has its own actor-critic but shares the team reward.
    """

    def __init__(self, n_agents: int, obs_size: int, n_actions: int,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4, batch_size: int = 64,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 device: str = 'cpu'):
        self.n_agents = n_agents
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

        self.models = [
            ActorCritic(obs_size, n_actions).to(device)
            for _ in range(n_agents)
        ]
        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr)
            for m in self.models
        ]
        self.buffers = [PPOBuffer() for _ in range(n_agents)]

        # Value function logging for MI analysis
        self.value_logs = {i: [] for i in range(n_agents)}

    def get_actions(self, observations: Dict[int, np.ndarray],
                    deterministic: bool = False) -> Tuple[Dict, Dict, Dict]:
        actions = {}
        log_probs = {}
        values = {}

        for i in range(self.n_agents):
            obs = torch.tensor(observations[i], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                a, lp, v = self.models[i].get_action(obs.unsqueeze(0), deterministic)
            actions[i] = a.item()
            log_probs[i] = lp.squeeze()
            values[i] = v.squeeze()
            self.value_logs[i].append(v.item())

        return actions, log_probs, values

    def store_transition(self, observations, actions, log_probs, reward, values, done):
        for i in range(self.n_agents):
            obs_t = torch.tensor(observations[i], dtype=torch.float32).to(self.device)
            act_t = torch.tensor(actions[i], dtype=torch.long)
            self.buffers[i].store(obs_t, act_t, log_probs[i], reward, values[i], done)

    def update(self):
        """Run PPO update for all agents."""
        total_losses = []

        for i in range(self.n_agents):
            if len(self.buffers[i].rewards) == 0:
                continue

            obs, actions, old_log_probs, returns, advantages, old_values = \
                self.buffers[i].get(self.gamma, self.lam)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs = obs.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)

            for _ in range(self.epochs):
                # Mini-batch updates
                indices = torch.randperm(len(obs))
                for start in range(0, len(obs), self.batch_size):
                    end = min(start + self.batch_size, len(obs))
                    idx = indices[start:end]

                    logits, values = self.models[i](obs[idx])
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - old_log_probs[idx].detach())
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(-1), returns[idx])

                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    self.optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.models[i].parameters(), 0.5)
                    self.optimizers[i].step()

                    total_losses.append(loss.item())

            self.buffers[i].clear()

        return np.mean(total_losses) if total_losses else 0.0

    def get_value_logs(self):
        """Return value function logs for MI analysis."""
        return {i: np.array(v) for i, v in self.value_logs.items()}

    def clear_value_logs(self):
        self.value_logs = {i: [] for i in range(self.n_agents)}
