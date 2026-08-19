"""Contextual-bandit and masked (constrained) PPO IPS baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ips.actions import IpsAction
from ips.dataset import IpsEpisode
from ips.dataset_environment import DatasetBackedIpsEnv


class LinUcbBandit:
    """Per-action linear UCB learner; intentionally ignores delayed credit."""

    def __init__(self, state_dim: int = 7, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.A = np.stack([np.eye(state_dim) for _ in IpsAction])
        self.b = np.zeros((len(IpsAction), state_dim))

    def choose(self, state: np.ndarray, mask: np.ndarray) -> IpsAction:
        scores = np.full(len(IpsAction), -np.inf)
        for action in np.flatnonzero(mask):
            inverse = np.linalg.inv(self.A[action])
            theta = inverse @ self.b[action]
            scores[action] = theta @ state + self.alpha * np.sqrt(state @ inverse @ state)
        return IpsAction(int(np.argmax(scores)))

    def update(self, state: np.ndarray, action: IpsAction, reward: float) -> None:
        index = int(action)
        self.A[index] += np.outer(state, state)
        self.b[index] += reward * state


def train_bandit(
    episodes: tuple[IpsEpisode, ...],
    *,
    training_episodes: int = 300,
    seed: int = 42,
) -> LinUcbBandit:
    if not episodes or training_episodes < 1:
        raise ValueError("episodes and training_episodes must be non-empty/positive")
    rng = np.random.default_rng(seed)
    bandit = LinUcbBandit()
    for index in range(training_episodes):
        episode = episodes[int(rng.integers(len(episodes)))]
        env = DatasetBackedIpsEnv(episode, seed=seed + index)
        observation, info = env.reset(seed=seed + index)
        while True:
            state = observation.as_array()
            action = bandit.choose(state, np.asarray(info["action_mask"]))
            result = env.step(action)
            executed = IpsAction[str(result.info["executed_action"])]
            bandit.update(state, executed, result.reward)
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            if result.terminated or result.truncated:
                break
    return bandit


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 7, action_dim: int = 7, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("every action mask must permit at least one action")
    return Categorical(logits=logits.masked_fill(~masks, -1e9))


@dataclass(frozen=True)
class PpoConfig:
    training_episodes: int = 300
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.20
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    constrained: bool = False
    disruption_limit: float = 0.15
    lagrange_learning_rate: float = 0.05


def _discounted(values: list[float], dones: list[bool], gamma: float) -> torch.Tensor:
    output = []
    running = 0.0
    for reward, done in zip(reversed(values), reversed(dones)):
        running = reward + gamma * running * (not done)
        output.append(running)
    return torch.tensor(list(reversed(output)), dtype=torch.float32)


def train_ppo(
    episodes: tuple[IpsEpisode, ...],
    *,
    config: PpoConfig | None = None,
    seed: int = 42,
) -> tuple[ActorCritic, pd.DataFrame]:
    """Train masked PPO; constrained mode learns a disruption multiplier."""
    import pandas as pd

    config = config or PpoConfig()
    if not episodes or config.training_episodes < 1:
        raise ValueError("episodes and training_episodes must be non-empty/positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic(hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lagrange = 0.0
    history = []
    disruptive = {IpsAction.RATE_LIMIT, IpsAction.DROP_FLOW, IpsAction.TEMP_BLOCK_SOURCE,
                  IpsAction.BLOCK_DESTINATION_PORT, IpsAction.ISOLATE_HOST}
    for episode_index in range(config.training_episodes):
        episode = episodes[int(rng.integers(len(episodes)))]
        env = DatasetBackedIpsEnv(episode, seed=seed + episode_index)
        observation, info = env.reset(seed=seed + episode_index)
        states=[]; masks=[]; actions=[]; old_logs=[]; rewards=[]; costs=[]; dones=[]
        while True:
            state = torch.as_tensor(observation.as_array(), dtype=torch.float32)
            mask = torch.as_tensor(info["action_mask"], dtype=torch.bool)
            with torch.no_grad():
                logits, _ = model(state.unsqueeze(0))
                distribution = masked_distribution(logits, mask.unsqueeze(0))
                action = distribution.sample()[0]
                log_prob = distribution.log_prob(action.unsqueeze(0))[0]
            result = env.step(int(action))
            executed = IpsAction[str(result.info["executed_action"])]
            cost = float(
                executed in disruptive
                and (not episode.contains_attack or observation.critical_service == 1.0)
            )
            done = result.terminated or result.truncated
            states.append(state); masks.append(mask); actions.append(action)
            old_logs.append(log_prob); rewards.append(result.reward); costs.append(cost); dones.append(done)
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            if done: break

        state_tensor = torch.stack(states)
        mask_tensor = torch.stack(masks)
        action_tensor = torch.stack(actions)
        old_log_tensor = torch.stack(old_logs)
        returns = _discounted(rewards, dones, config.gamma)
        cost_returns = _discounted(costs, dones, config.gamma)
        with torch.no_grad():
            _, initial_values = model(state_tensor)
            advantages = returns - initial_values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        last_loss = 0.0
        for _ in range(config.update_epochs):
            logits, values = model(state_tensor)
            distribution = masked_distribution(logits, mask_tensor)
            log_probs = distribution.log_prob(action_tensor)
            ratio = torch.exp(log_probs - old_log_tensor)
            clipped = torch.clamp(ratio, 1-config.clip_ratio, 1+config.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            if config.constrained:
                policy_loss = policy_loss + lagrange * (ratio * cost_returns).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            last_loss = float(loss.detach())
        mean_cost = float(np.mean(costs))
        if config.constrained:
            lagrange = max(0.0, lagrange + config.lagrange_learning_rate * (mean_cost-config.disruption_limit))
        history.append({"episode":episode_index+1, "return":sum(rewards), "disruption_cost":mean_cost,
                        "lagrange":lagrange, "loss":last_loss})
    return model, pd.DataFrame(history)


def actor_policy(model: ActorCritic):
    def policy(state: np.ndarray, mask: np.ndarray) -> IpsAction:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            safe = logits[0].masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -torch.inf)
            return IpsAction(int(torch.argmax(safe)))
    return policy
