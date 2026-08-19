"""Methods adopted from prior RL/IPS literature with explicit provenance.

References:
- Schaul et al. (2016), Prioritized Experience Replay.
- Hammar & Stadler (2021), intrusion prevention as optimal stopping.
- Jiang & Li (2016), doubly robust/off-policy evaluation foundations.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn

from ips.actions import IpsAction
from ips.dataset import EpisodeSplits
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.dataset_experiment import DatasetPolicyMetrics, evaluate_on_episodes
from ips.dqn import DqnConfig, QNetwork, Transition, epsilon_at_step, select_action


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with normalized importance weights."""

    def __init__(self, capacity: int, *, alpha: float = 0.6, seed: int = 42) -> None:
        if capacity < 1 or not 0 <= alpha <= 1:
            raise ValueError("capacity must be positive and alpha in [0, 1]")
        self.capacity = capacity
        self.alpha = alpha
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition, *, priority: float | None = None) -> None:
        default = max(self._priorities, default=1.0)
        value = default if priority is None else float(priority)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("priority must be finite and positive")
        self._items.append(transition)
        self._priorities.append(value)

    def sample(self, batch_size: int, *, beta: float = 0.4) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if batch_size < 1 or batch_size > len(self) or not 0 <= beta <= 1:
            raise ValueError("invalid batch_size or beta")
        priorities = np.asarray(self._priorities, dtype=float) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = self._rng.choice(len(self), size=batch_size, replace=False, p=probabilities)
        weights = (len(self) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        items = list(self._items)
        return [items[int(index)] for index in indices], indices.astype(int), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        values = list(self._priorities)
        for index, priority in zip(indices, priorities):
            if not np.isfinite(priority) or priority <= 0:
                raise ValueError("priority must be finite and positive")
            values[int(index)] = float(priority)
        self._priorities = deque(values, maxlen=self.capacity)


class RepeatedEvidenceGate:
    """Per-identity detect/forget hysteresis for noisy detector decisions.

    Inspired by Akamai's production description of requiring repeated positive
    evidence and expiring state after a configurable quiet period.  This class
    is generic: callers choose the identity key and must avoid using labels.
    """

    def __init__(self, *, positives_to_activate: int = 2, negatives_to_forget: int = 3) -> None:
        if positives_to_activate < 1 or negatives_to_forget < 1:
            raise ValueError("hysteresis thresholds must be positive")
        self.positives_to_activate = positives_to_activate
        self.negatives_to_forget = negatives_to_forget
        self._positive_runs: defaultdict[str, int] = defaultdict(int)
        self._negative_runs: defaultdict[str, int] = defaultdict(int)
        self._active: defaultdict[str, bool] = defaultdict(bool)

    def update(self, identity: str, positive: bool) -> bool:
        """Update one identity and return whether its evidence state is active."""
        if positive:
            self._positive_runs[identity] += 1
            self._negative_runs[identity] = 0
            if self._positive_runs[identity] >= self.positives_to_activate:
                self._active[identity] = True
        else:
            self._positive_runs[identity] = 0
            self._negative_runs[identity] += 1
            if self._negative_runs[identity] >= self.negatives_to_forget:
                self._active[identity] = False
        return self._active[identity]


def shared_ip_guardrail(
    proposed: IpsAction,
    mask: np.ndarray,
    *,
    shared_ip_probability: float,
    threshold: float = 0.70,
) -> IpsAction:
    """Avoid source-wide collateral damage when an address is probably shared.

    A source block or host/port-wide action is downgraded to the strongest valid
    flow-local action.  This is a safety postprocessor, not a learned label and
    not evidence that a traffic source is actually behind NAT.
    """
    if not 0 <= shared_ip_probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probabilities and threshold must be in [0, 1]")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(IpsAction),):
        raise ValueError(f"mask must have shape ({len(IpsAction)},)")
    if not mask[int(proposed)]:
        return IpsAction.MONITOR
    broad_actions = {
        IpsAction.TEMP_BLOCK_SOURCE,
        IpsAction.BLOCK_DESTINATION_PORT,
        IpsAction.ISOLATE_HOST,
    }
    if shared_ip_probability < threshold or proposed not in broad_actions:
        return proposed
    for fallback in (IpsAction.DROP_FLOW, IpsAction.RATE_LIMIT, IpsAction.MONITOR, IpsAction.ALLOW):
        if mask[int(fallback)]:
            return fallback
    return IpsAction.ALLOW


def optimal_stopping_policy(
    state: np.ndarray, mask: np.ndarray, *, threshold: float = 0.70
) -> IpsAction:
    """Threshold baseline inspired by intrusion-prevention stopping games."""
    threat, _, stage, compromise, critical, _, _ = np.asarray(state, dtype=float)
    risk = 0.45 * threat + 0.25 * stage + 0.25 * compromise - 0.10 * critical
    if risk < threshold:
        return IpsAction.MONITOR if mask[IpsAction.MONITOR] else IpsAction.ALLOW
    for action in (
        IpsAction.ISOLATE_HOST,
        IpsAction.BLOCK_DESTINATION_PORT,
        IpsAction.TEMP_BLOCK_SOURCE,
        IpsAction.DROP_FLOW,
        IpsAction.RATE_LIMIT,
    ):
        if mask[int(action)]:
            return action
    return IpsAction.MONITOR


def weighted_importance_sampling(
    returns: np.ndarray, target_probabilities: np.ndarray, behavior_probabilities: np.ndarray
) -> float:
    """Self-normalized OPE estimate; valid only with logged behavior propensities."""
    returns = np.asarray(returns, dtype=float)
    target = np.asarray(target_probabilities, dtype=float)
    behavior = np.asarray(behavior_probabilities, dtype=float)
    if not (returns.shape == target.shape == behavior.shape) or not returns.size:
        raise ValueError("OPE arrays must have equal non-zero shape")
    if np.any(behavior <= 0) or np.any(target < 0):
        raise ValueError("behavior probabilities must be positive and target non-negative")
    weights = target / behavior
    if weights.sum() <= 0:
        raise ValueError("importance weights sum to zero")
    return float(np.sum(weights * returns) / np.sum(weights))


def effective_sample_size(weights: np.ndarray) -> float:
    """Importance-weight diagnostic; low ESS warns that OPE is unreliable."""
    values = np.asarray(weights, dtype=float)
    if not values.size or np.any(values < 0) or values.sum() == 0:
        raise ValueError("weights must be non-negative with positive sum")
    return float(values.sum() ** 2 / np.square(values).sum())


def optimize_prioritized_batch(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    config: DqnConfig,
    *,
    beta: float,
) -> float:
    """Masked Double-DQN update weighted by proportional replay importance."""
    transitions, indices, weights = replay.sample(config.batch_size, beta=beta)
    states = torch.as_tensor(np.stack([item.state for item in transitions]), dtype=torch.float32)
    actions = torch.as_tensor([item.action for item in transitions], dtype=torch.long)
    rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([item.next_state for item in transitions]), dtype=torch.float32)
    terminated = torch.as_tensor([item.terminated for item in transitions], dtype=torch.bool)
    masks = torch.as_tensor(np.stack([item.next_action_mask for item in transitions]), dtype=torch.bool)
    current = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = online(next_states).masked_fill(~masks, -torch.inf)
        selected = online_next.argmax(dim=1)
        bootstrap = target(next_states).gather(1, selected.unsqueeze(1)).squeeze(1)
        bootstrap = bootstrap.masked_fill(terminated, 0.0)
        expected = rewards + config.gamma * bootstrap
    td_error = expected - current
    loss = (torch.as_tensor(weights) * nn.functional.smooth_l1_loss(current, expected, reduction="none")).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), config.max_grad_norm)
    optimizer.step()
    replay.update_priorities(indices, np.abs(td_error.detach().numpy()) + 1e-5)
    return float(loss.detach())


def train_prioritized_dqn(
    splits: EpisodeSplits,
    *,
    episodes: int = 100,
    validation_interval: int = 25,
    config: DqnConfig | None = None,
    seed: int = 42,
) -> tuple[QNetwork, DatasetPolicyMetrics, dict[str, float]]:
    """Train PER-DQN and restore the validation-selected network in memory."""
    config = config or DqnConfig(batch_size=32, warmup_steps=64)
    if episodes < 1 or validation_interval < 1:
        raise ValueError("training counts must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online = QNetwork(config.state_dim, config.action_dim, config.hidden_dim)
    target = QNetwork(config.state_dim, config.action_dim, config.hidden_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = PrioritizedReplayBuffer(config.replay_capacity, seed=seed)
    best_metrics: DatasetPolicyMetrics | None = None
    best_key: tuple[float, float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    steps = updates = 0
    for episode_index in range(1, episodes + 1):
        episode = splits.train[int(rng.integers(len(splits.train)))]
        env = DatasetBackedIpsEnv(episode, seed=seed + episode_index)
        observation, info = env.reset(seed=seed + episode_index)
        while True:
            state = observation.as_array()
            action = select_action(
                online, state, info["action_mask"],
                epsilon=epsilon_at_step(steps, config), rng=rng,
            )
            result = env.step(action)
            finished = result.terminated or result.truncated
            executed = IpsAction[str(result.info["executed_action"])]
            transition = Transition(
                state, int(executed), result.reward, result.observation.as_array(), finished,
                np.asarray(result.info["action_mask"], dtype=bool).copy(),
            )
            replay.append(transition, priority=abs(result.reward) + 1e-3)
            steps += 1
            if steps >= config.warmup_steps and len(replay) >= config.batch_size:
                progress = min(1.0, steps / max(config.epsilon_decay_steps, 1))
                optimize_prioritized_batch(online, target, optimizer, replay, config, beta=0.4 + 0.6 * progress)
                updates += 1
            if steps % config.target_update_steps == 0:
                target.load_state_dict(online.state_dict())
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            if finished:
                break
        if episode_index % validation_interval == 0 or episode_index == episodes:
            metrics = evaluate_on_episodes(online, splits.validation, seed=seed + 1_000_000)
            key = (-metrics.compromise_rate, metrics.containment_rate, -metrics.false_preventions_per_episode, metrics.mean_return)
            if best_key is None or key > best_key:
                best_key, best_metrics = key, metrics
                best_state = {name: value.detach().clone() for name, value in online.state_dict().items()}
    if best_metrics is None or best_state is None:
        raise RuntimeError("PER-DQN did not create a validation checkpoint")
    online.load_state_dict(best_state)
    return online, best_metrics, {"steps": float(steps), "updates": float(updates)}
