"""Canonical dataset-backed episode schema for adaptive IPS experiments.

Detector probabilities must be generated out-of-fold for training events and by
a train-fitted detector for validation/test events. Ground-truth labels may
drive outcome auditing, but are never substituted for detector probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = frozenset(
    {
        "episode_id",
        "group_id",
        "timestamp",
        "threat_probability",
        "anomaly_score",
        "attack_present",
        "attack_stage",
        "critical_service",
        "attack_family",
    }
)


@dataclass(frozen=True)
class IpsEvent:
    timestamp: float
    threat_probability: float
    anomaly_score: float
    attack_present: bool
    attack_stage: float
    critical_service: bool
    attack_family: str


@dataclass(frozen=True)
class IpsEpisode:
    episode_id: str
    group_id: str
    events: tuple[IpsEvent, ...]

    @property
    def contains_attack(self) -> bool:
        return any(event.attack_present for event in self.events)


@dataclass(frozen=True)
class EpisodeSplits:
    train: tuple[IpsEpisode, ...]
    validation: tuple[IpsEpisode, ...]
    test: tuple[IpsEpisode, ...]


def _bounded_probability(value: object, column: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{column} must contain finite values in [0, 1]")
    return number


def build_episodes(frame: pd.DataFrame, *, max_events: int | None = None) -> tuple[IpsEpisode, ...]:
    """Build sorted episodes from an audited event table.

    ``group_id`` identifies the indivisible campaign/host/session unit used for
    splitting. Multiple episodes may share one group and must remain together.
    """
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required IPS event columns: {missing}")
    if frame.empty:
        raise ValueError("cannot build IPS episodes from an empty frame")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required IPS event columns may not contain missing values")

    episodes: list[IpsEpisode] = []
    for episode_id, rows in frame.groupby("episode_id", sort=True):
        groups = rows["group_id"].astype(str).unique()
        if len(groups) != 1:
            raise ValueError(f"episode {episode_id!r} crosses multiple group_id values")
        ordered = rows.sort_values("timestamp", kind="stable")
        if ordered["timestamp"].duplicated().any():
            raise ValueError(f"episode {episode_id!r} contains duplicate timestamps")
        if max_events is not None:
            ordered = ordered.iloc[:max_events]
        events = tuple(
            IpsEvent(
                timestamp=float(row.timestamp),
                threat_probability=_bounded_probability(
                    row.threat_probability, "threat_probability"
                ),
                anomaly_score=_bounded_probability(row.anomaly_score, "anomaly_score"),
                attack_present=bool(row.attack_present),
                attack_stage=_bounded_probability(row.attack_stage, "attack_stage"),
                critical_service=bool(row.critical_service),
                attack_family=str(row.attack_family),
            )
            for row in ordered.itertuples(index=False)
        )
        episodes.append(IpsEpisode(str(episode_id), str(groups[0]), events))
    return tuple(episodes)


def split_episodes_by_group(
    episodes: tuple[IpsEpisode, ...],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> EpisodeSplits:
    """Create deterministic, leakage-safe train/validation/final-test splits."""
    if not episodes:
        raise ValueError("episodes may not be empty")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    groups = np.array(sorted({episode.group_id for episode in episodes}), dtype=object)
    if len(groups) < 3:
        raise ValueError("at least three distinct groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = max(1, int(round(len(groups) * train_fraction)))
    n_validation = max(1, int(round(len(groups) * validation_fraction)))
    if n_train + n_validation >= len(groups):
        n_train = len(groups) - 2
        n_validation = 1
    train_groups = set(groups[:n_train])
    validation_groups = set(groups[n_train:n_train + n_validation])
    test_groups = set(groups[n_train + n_validation:])

    def select(group_set: set[object]) -> tuple[IpsEpisode, ...]:
        return tuple(ep for ep in episodes if ep.group_id in group_set)

    result = EpisodeSplits(
        train=select(train_groups),
        validation=select(validation_groups),
        test=select(test_groups),
    )
    if not result.train or not result.validation or not result.test:
        raise RuntimeError("group split unexpectedly produced an empty partition")
    return result


def assert_group_disjoint(splits: EpisodeSplits) -> None:
    """Fail loudly if any campaign/host/session leaks between partitions."""
    groups = [
        {episode.group_id for episode in split}
        for split in (splits.train, splits.validation, splits.test)
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("group leakage detected across episode splits")
