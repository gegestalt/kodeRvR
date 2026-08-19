"""Multi-seed DQN training and final testing on dataset-backed IPS episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ips.actions import IpsAction
from ips.dataset import EpisodeSplits, IpsEpisode, assert_group_disjoint
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.dqn import DqnConfig, QNetwork, ReplayBuffer, Transition, epsilon_at_step, select_action
from ips.train_dqn import optimize_batch


@dataclass(frozen=True)
class DatasetTrainConfig:
    episodes: int = 1_000
    validation_interval: int = 100


@dataclass(frozen=True)
class DatasetPolicyMetrics:
    episodes: int
    attack_episodes: int
    containment_rate: float
    compromise_rate: float
    false_preventions_per_episode: float
    disruptive_actions_per_episode: float
    mean_return: float
    return_std: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_on_episodes(
    network: QNetwork,
    episodes: tuple[IpsEpisode, ...],
    *,
    seed: int,
) -> DatasetPolicyMetrics:
    """Evaluate a frozen greedy policy over each supplied episode exactly once."""
    if not episodes:
        raise ValueError("evaluation episodes may not be empty")
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    attack_count = contained = compromised = false_preventions = disruptive = 0
    disruptive_actions = {
        "RATE_LIMIT", "DROP_FLOW", "TEMP_BLOCK_SOURCE",
        "BLOCK_DESTINATION_PORT", "ISOLATE_HOST",
    }
    network.eval()
    for index, episode in enumerate(episodes):
        attack_count += int(episode.contains_attack)
        env = DatasetBackedIpsEnv(episode, seed=seed + index)
        observation, info = env.reset(seed=seed + index)
        total = 0.0
        while True:
            action = select_action(
                network, observation.as_array(), info["action_mask"], epsilon=0.0, rng=rng
            )
            result = env.step(action)
            total += result.reward
            contained += int(bool(result.info["contained"]))
            compromised += int(bool(result.info["compromised"]))
            executed = str(result.info["executed_action"])
            disruptive += int(executed in disruptive_actions)
            false_preventions += int(
                not episode.contains_attack and executed in disruptive_actions
            )
            observation = result.observation
            info = {"action_mask": result.info["action_mask"]}
            if result.terminated or result.truncated:
                break
        returns.append(total)
    values = np.asarray(returns, dtype=float)
    return DatasetPolicyMetrics(
        episodes=len(episodes),
        attack_episodes=attack_count,
        containment_rate=contained / attack_count if attack_count else 0.0,
        compromise_rate=compromised / attack_count if attack_count else 0.0,
        false_preventions_per_episode=false_preventions / len(episodes),
        disruptive_actions_per_episode=disruptive / len(episodes),
        mean_return=float(values.mean()),
        return_std=float(values.std(ddof=0)),
    )


def _selection_key(metrics: DatasetPolicyMetrics) -> tuple[float, float, float, float]:
    return (
        -metrics.compromise_rate,
        metrics.containment_rate,
        -metrics.false_preventions_per_episode,
        metrics.mean_return,
    )


def train_one_seed(
    splits: EpisodeSplits,
    *,
    seed: int,
    dqn: DqnConfig,
    training: DatasetTrainConfig,
    output_dir: Path,
    evaluate_test: bool = True,
) -> dict[str, object]:
    """Train on train episodes, select on validation, test the winner once."""
    assert_group_disjoint(splits)
    if training.episodes < 1 or training.validation_interval < 1:
        raise ValueError("training counts must be positive")
    if dqn.batch_size > dqn.replay_capacity:
        raise ValueError("batch_size cannot exceed replay_capacity")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online = QNetwork(dqn.state_dim, dqn.action_dim, dqn.hidden_dim)
    target = QNetwork(dqn.state_dim, dqn.action_dim, dqn.hidden_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=dqn.learning_rate)
    replay = ReplayBuffer(dqn.replay_capacity, seed=seed)
    total_steps = updates = 0
    best_key: tuple[float, float, float, float] | None = None
    best_validation: DatasetPolicyMetrics | None = None
    checkpoint = output_dir / "best.pt"

    for training_episode in range(1, training.episodes + 1):
        episode = splits.train[int(rng.integers(len(splits.train)))]
        env = DatasetBackedIpsEnv(episode, seed=seed + training_episode)
        observation, info = env.reset(seed=seed + training_episode)
        while True:
            state = observation.as_array()
            action = select_action(
                online,
                state,
                info["action_mask"],
                epsilon=epsilon_at_step(total_steps, dqn),
                rng=rng,
            )
            result = env.step(action)
            finished = result.terminated or result.truncated
            executed = IpsAction[str(result.info["executed_action"])]
            replay.append(
                Transition(
                    state,
                    int(executed),
                    result.reward,
                    result.observation.as_array(),
                    finished,
                    np.asarray(result.info["action_mask"], dtype=bool).copy(),
                )
            )
            total_steps += 1
            if total_steps >= dqn.warmup_steps and len(replay) >= dqn.batch_size:
                optimize_batch(online, target, optimizer, replay, dqn)
                updates += 1
            if total_steps % dqn.target_update_steps == 0:
                target.load_state_dict(online.state_dict())
            observation = result.observation
            info = {"action_mask": result.info["action_mask"]}
            if finished:
                break

        if training_episode % training.validation_interval == 0 or training_episode == training.episodes:
            validation = evaluate_on_episodes(
                online, splits.validation, seed=seed + 1_000_000
            )
            key = _selection_key(validation)
            if best_key is None or key > best_key:
                best_key, best_validation = key, validation
                torch.save(
                    {
                        "model_state_dict": online.state_dict(),
                        "seed": seed,
                        "training_episode": training_episode,
                        "dqn_config": asdict(dqn),
                        "validation": validation.to_dict(),
                    },
                    checkpoint,
                )

    if best_validation is None:
        raise RuntimeError("no validation checkpoint was created")
    saved = torch.load(checkpoint, map_location="cpu")
    online.load_state_dict(saved["model_state_dict"])
    final_test = (
        evaluate_on_episodes(online, splits.test, seed=seed + 2_000_000)
        if evaluate_test else None
    )
    return {
        "seed": seed,
        "steps": total_steps,
        "updates": updates,
        "validation": best_validation.to_dict(),
        "final_test": final_test.to_dict() if final_test is not None else None,
        "checkpoint": str(checkpoint),
    }


def run_multi_seed(
    splits: EpisodeSplits,
    *,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    dqn: DqnConfig | None = None,
    training: DatasetTrainConfig | None = None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run independent seeds and save per-seed plus mean/std final-test metrics."""
    if len(seeds) < 3:
        raise ValueError("at least three seeds are required for a stability claim")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    dqn = dqn or DqnConfig()
    training = training or DatasetTrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        train_one_seed(
            splits,
            seed=seed,
            dqn=dqn,
            training=training,
            output_dir=output_dir / f"seed_{seed}",
        )
        for seed in seeds
    ]
    rows = pd.DataFrame(
        [{"seed": run["seed"], **run["final_test"]} for run in runs]
    )
    rows.to_csv(output_dir / "final_test_per_seed.csv", index=False)
    numeric = [column for column in rows.columns if column != "seed"]
    summary = {
        column: {
            "mean": float(rows[column].mean()),
            "std": float(rows[column].std(ddof=0)),
        }
        for column in numeric
    }
    (output_dir / "final_test_summary.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "final_test": summary,
                "protocol": (
                    "group-disjoint train/validation/test; validation selects each "
                    "checkpoint; final test evaluated once per seed"
                ),
                "limitation": "observations are dataset-backed; action outcomes remain counterfactual",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows
