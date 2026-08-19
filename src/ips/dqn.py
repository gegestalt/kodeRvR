"""Reusable DQN components for the adaptive IPS.

The full training loop belongs in ``train_dqn.py``. This module keeps the model,
replay memory, epsilon schedule, and action masking independently testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ips.actions import IpsAction


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    next_action_mask: np.ndarray


@dataclass(frozen=True)
class DqnConfig:
    state_dim: int = 7
    action_dim: int = len(IpsAction)
    hidden_dim: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    target_update_steps: int = 500
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000


class QNetwork(nn.Module):
    """Small MLP mapping an IPS observation to one Q-value per action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class ReplayBuffer:
    """Bounded replay memory with seeded sampling."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size < 1 or batch_size > len(self._items):
            raise ValueError("batch_size must be positive and no larger than the buffer")
        indices = self._rng.choice(len(self._items), size=batch_size, replace=False)
        return [self._items[int(i)] for i in indices]


def epsilon_at_step(step: int, config: DqnConfig) -> float:
    """Linear exploration schedule with a fixed lower bound."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if config.epsilon_decay_steps < 1:
        raise ValueError("epsilon_decay_steps must be positive")
    progress = min(step / config.epsilon_decay_steps, 1.0)
    return float(config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start))


def select_action(
    network: QNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> IpsAction:
    """Epsilon-greedy action selection that can never bypass safety masking."""
    valid = np.flatnonzero(action_mask)
    if not len(valid):
        raise ValueError("action_mask must allow at least one action")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return IpsAction(int(rng.choice(valid)))

    network.eval()
    with torch.no_grad():
        q_values = network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0]
        safe_q = q_values.clone()
        safe_q[~torch.as_tensor(action_mask, dtype=torch.bool)] = -torch.inf
        return IpsAction(int(torch.argmax(safe_q).item()))
