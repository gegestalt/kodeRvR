"""Masked Double-DQN training for the adaptive IPS environment.

Training scenarios and validation scenarios use disjoint seed ranges. The best
checkpoint is selected by prevention outcomes (compromise, containment, false
prevention) before reward, because reward alone is not evidence of a safe IPS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data as D
from ips.actions import IpsAction
from ips.dqn import (
    DqnConfig,
    QNetwork,
    ReplayBuffer,
    Transition,
    epsilon_at_step,
    select_action,
)
from ips.environment import AdaptiveIpsEnv, IpsObservation
from ips.evaluate import PolicyMetrics, evaluate_policy


@dataclass(frozen=True)
class TrainConfig:
    """Episode-level settings kept separate from DQN hyperparameters."""

    episodes: int = 1_000
    validation_interval: int = 100
    validation_episodes: int = 200
    attack_probability: float = 0.50
    critical_probability: float = 0.25
    validation_seed_offset: int = 1_000_000


@dataclass(frozen=True)
class TrainingResult:
    """Summary and artifacts returned by a completed training run."""

    online: QNetwork
    best_checkpoint: Path
    history_path: Path
    summary_path: Path
    total_steps: int
    updates: int
    best_validation: PolicyMetrics


def _validate_config(config: DqnConfig, training: TrainConfig) -> None:
    if config.state_dim != 7:
        raise ValueError("AdaptiveIpsEnv observations currently have state_dim=7")
    if config.action_dim != len(IpsAction):
        raise ValueError(f"action_dim must be {len(IpsAction)}")
    if config.batch_size < 1 or config.replay_capacity < config.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if config.target_update_steps < 1 or config.warmup_steps < 0:
        raise ValueError(
            "target_update_steps must be positive and warmup_steps non-negative"
        )
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if training.episodes < 1 or training.validation_episodes < 1:
        raise ValueError("training and validation episode counts must be positive")
    if training.validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    for name, value in (
        ("attack_probability", training.attack_probability),
        ("critical_probability", training.critical_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def optimize_batch(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: DqnConfig,
) -> float:
    """Perform one safety-masked Double-DQN update and return scalar loss."""
    if len(replay) < config.batch_size:
        raise ValueError("replay does not yet contain a full batch")

    device = next(online.parameters()).device
    batch = replay.sample(config.batch_size)
    states = torch.as_tensor(
        np.stack([item.state for item in batch]), dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(
        [item.action for item in batch], dtype=torch.long, device=device
    )
    rewards = torch.as_tensor(
        [item.reward for item in batch], dtype=torch.float32, device=device
    )
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in batch]),
        dtype=torch.float32,
        device=device,
    )
    terminated = torch.as_tensor(
        [item.terminated for item in batch], dtype=torch.bool, device=device
    )
    next_masks = torch.as_tensor(
        np.stack([item.next_action_mask for item in batch]),
        dtype=torch.bool,
        device=device,
    )
    if not bool(next_masks.any(dim=1).all()):
        raise ValueError("every transition must allow at least one next action")

    online.train()
    current_q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next_q = online(next_states).masked_fill(~next_masks, -torch.inf)
        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
        target_next_q = target(next_states).gather(1, best_next_actions).squeeze(1)
        target_values = rewards + config.gamma * target_next_q * (~terminated).float()

    loss = nn.functional.smooth_l1_loss(current_q, target_values)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), config.max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu().item())


def _greedy_policy(
    network: QNetwork, seed: int
) -> Callable[[IpsObservation, np.ndarray], IpsAction]:
    """Adapt a Q-network to the shared policy-evaluation interface."""
    rng = np.random.default_rng(seed)

    def policy(observation: IpsObservation, action_mask: np.ndarray) -> IpsAction:
        return select_action(
            network,
            observation.as_array(),
            action_mask,
            epsilon=0.0,
            rng=rng,
        )

    return policy


def _selection_key(metrics: PolicyMetrics) -> tuple[float, float, float, float]:
    """Safety-first ordering for choosing a validation checkpoint."""
    return (
        -metrics.compromise_rate,
        metrics.containment_rate,
        -metrics.false_preventions_per_episode,
        metrics.mean_return,
    )


def _save_checkpoint(
    path: Path,
    network: QNetwork,
    *,
    episode: int,
    total_steps: int,
    config: DqnConfig,
    training: TrainConfig,
    metrics: PolicyMetrics,
) -> None:
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "episode": episode,
            "total_steps": total_steps,
            "dqn_config": asdict(config),
            "train_config": asdict(training),
            "validation_metrics": metrics.to_dict(),
        },
        path,
    )


def train(
    config: DqnConfig | None = None,
    training: TrainConfig | None = None,
    *,
    seed: int = D.RANDOM_STATE,
    output_dir: Path | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train DQN and select a checkpoint on disjoint validation scenarios."""
    config = config or DqnConfig()
    training = training or TrainConfig()
    _validate_config(config, training)
    output_dir = output_dir or (D.REPO_ROOT / "results" / "ips")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
    target = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, seed=seed)

    history: list[dict[str, object]] = []
    total_steps = updates = 0
    best_metrics: PolicyMetrics | None = None
    best_key: tuple[float, float, float, float] | None = None
    checkpoint_path = output_dir / "dqn_best.pt"

    for episode in range(1, training.episodes + 1):
        attack_present = bool(rng.random() < training.attack_probability)
        critical_service = bool(rng.random() < training.critical_probability)
        env_seed = seed + episode
        env = AdaptiveIpsEnv(seed=env_seed)
        observation, info = env.reset(
            seed=env_seed,
            attack_present=attack_present,
            critical_service=critical_service,
        )
        episode_return = 0.0
        episode_losses: list[float] = []

        while True:
            state = observation.as_array()
            epsilon = epsilon_at_step(total_steps, config)
            action = select_action(
                online, state, info["action_mask"], epsilon=epsilon, rng=rng
            )
            result = env.step(action)
            finished = bool(
                result.terminated or result.truncated or result.info["contained"]
            )
            executed_action = IpsAction[str(result.info["executed_action"])]
            replay.append(
                Transition(
                    state=state,
                    action=int(executed_action),
                    reward=result.reward,
                    next_state=result.observation.as_array(),
                    terminated=finished,
                    next_action_mask=np.asarray(
                        result.info["action_mask"], dtype=bool
                    ).copy(),
                )
            )
            total_steps += 1
            episode_return += result.reward

            if total_steps >= config.warmup_steps and len(replay) >= config.batch_size:
                episode_losses.append(
                    optimize_batch(online, target, optimizer, replay, config)
                )
                updates += 1
            if total_steps % config.target_update_steps == 0:
                target.load_state_dict(online.state_dict())

            observation = result.observation
            info = {"action_mask": result.info["action_mask"]}
            if finished:
                break

        row: dict[str, object] = {
            "episode": episode,
            "total_steps": total_steps,
            "episode_return": episode_return,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else None,
            "epsilon": epsilon_at_step(total_steps, config),
        }

        should_validate = (
            episode % training.validation_interval == 0
            or episode == training.episodes
        )
        if should_validate:
            validation_seed = seed + training.validation_seed_offset
            metrics = evaluate_policy(
                _greedy_policy(online, validation_seed),
                episodes=training.validation_episodes,
                seed=validation_seed,
                attack_probability=training.attack_probability,
                critical_probability=training.critical_probability,
            )
            row["validation"] = metrics.to_dict()
            key = _selection_key(metrics)
            if best_key is None or key > best_key:
                best_key, best_metrics = key, metrics
                _save_checkpoint(
                    checkpoint_path,
                    online,
                    episode=episode,
                    total_steps=total_steps,
                    config=config,
                    training=training,
                    metrics=metrics,
                )
        history.append(row)

    if best_metrics is None:
        raise RuntimeError("training completed without validation")

    history_path = output_dir / "dqn_history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    summary_path = output_dir / "dqn_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "total_steps": total_steps,
                "updates": updates,
                "best_validation": best_metrics.to_dict(),
                "checkpoint": str(checkpoint_path),
                "limitation": (
                    "Validation uses simulated IPS transitions; reserve a separate "
                    "test seed range and cyber-range evaluation for final claims."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainingResult(
        online=online,
        best_checkpoint=checkpoint_path,
        history_path=history_path,
        summary_path=summary_path,
        total_steps=total_steps,
        updates=updates,
        best_validation=best_metrics,
    )


if __name__ == "__main__":
    result = train()
    print(f"checkpoint: {result.best_checkpoint}")
    print(json.dumps(result.best_validation.to_dict(), indent=2, sort_keys=True))
