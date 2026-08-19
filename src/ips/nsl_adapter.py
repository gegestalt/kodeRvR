"""Convert real NSL-KDD rows into leakage-aware IPS sequence proxies.

NSL-KDD has official train/test partitions but no timestamps, host IDs, or
intervention outcomes. Consequently these episodes are ordered-row proxies for
policy comparison, not claims about temporal containment in a live network.
"""

from __future__ import annotations

from dataclasses import dataclass
import resource
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import ATTACK_FAMILY_MAP, CATEGORICAL_COLS, FEATURE_NAMES, NUMERIC_COLS
from ips.dataset import EpisodeSplits, build_episodes, split_episodes_by_group


@dataclass(frozen=True)
class NslIpsConfig:
    max_train_rows: int | None = 25_000
    max_test_rows: int | None = 10_000
    episode_size: int = 12
    folds: int = 3
    seed: int = 42


@dataclass(frozen=True)
class NslIpsEvidence:
    splits: EpisodeSplits
    train_events: pd.DataFrame
    test_events: pd.DataFrame
    metadata: dict[str, object]
    resources: dict[str, float]


def _prepare(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    required = {*FEATURE_NAMES, "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"NSL-KDD frame is missing columns: {missing}")
    work = frame.copy()
    if limit is not None and len(work) > limit:
        # Preserve every real attack family while bounding notebook cost.
        family = work["label"].map(ATTACK_FAMILY_MAP)
        fraction = limit / len(work)
        work = (
            work.assign(_family=family)
            .groupby("_family", group_keys=False, sort=False)
            .sample(frac=fraction, random_state=seed)
            .drop(columns="_family")
        )
        if len(work) > limit:
            work = work.sample(limit, random_state=seed)
    return work.reset_index(drop=False).rename(columns={"index": "source_row"})


def _pipeline() -> object:
    transform = ColumnTransformer(
        (
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_COLS),
        )
    )
    return make_pipeline(
        transform,
        LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear"),
    )


def _oof_and_test_scores(
    train: pd.DataFrame, test: pd.DataFrame, config: NslIpsConfig
) -> tuple[np.ndarray, np.ndarray]:
    y = (train["label"] != "normal").astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("NSL-KDD training rows must include normal and attack traffic")
    scores = np.full(len(train), np.nan)
    folds = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    for fit_index, held_index in folds.split(train, y):
        model = _pipeline()
        model.fit(train.iloc[fit_index][FEATURE_NAMES], y[fit_index])
        scores[held_index] = model.predict_proba(train.iloc[held_index][FEATURE_NAMES])[:, 1]
    final_model = _pipeline()
    final_model.fit(train[FEATURE_NAMES], y)
    test_scores = final_model.predict_proba(test[FEATURE_NAMES])[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("out-of-fold scoring left NSL-KDD rows unscored")
    return scores, test_scores


def _events(frame: pd.DataFrame, scores: np.ndarray, split: str, episode_size: int) -> pd.DataFrame:
    family = frame["label"].map(ATTACK_FAMILY_MAP)
    if family.isna().any():
        unknown = sorted(frame.loc[family.isna(), "label"].unique())
        raise ValueError(f"unmapped NSL-KDD labels: {unknown}")
    output = []
    # Keep each proxy episode family-pure so family containment is attributable.
    for family_name, indices in family.groupby(family).groups.items():
        ordered = list(indices)
        for chunk_number, start in enumerate(range(0, len(ordered), episode_size)):
            chunk = ordered[start : start + episode_size]
            if len(chunk) < 2:
                continue
            episode_id = f"nsl-{split}-{family_name}-{chunk_number}"
            for position, index in enumerate(chunk):
                attack = family_name != "normal"
                score = float(np.clip(scores[index], 0, 1))
                output.append(
                    {
                        "episode_id": episode_id,
                        "group_id": episode_id,
                        "timestamp": float(frame.iloc[index]["source_row"]),
                        "threat_probability": score,
                        "anomaly_score": float(abs(score - 0.5) * 2),
                        "attack_present": attack,
                        "attack_stage": (position + 1) / len(chunk) if attack else 0.0,
                        "critical_service": str(frame.iloc[index]["service"]) in {"http", "smtp", "ftp", "domain_u"},
                        "attack_family": family_name,
                        "split": split,
                        "source_row": int(frame.iloc[index]["source_row"]),
                        "raw_label": str(frame.iloc[index]["label"]),
                    }
                )
    return pd.DataFrame(output)


def build_nsl_ips_evidence(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: NslIpsConfig | None = None,
) -> NslIpsEvidence:
    """Score official partitions and construct honest sequence-proxy episodes."""
    config = config or NslIpsConfig()
    if config.episode_size < 2 or config.folds < 2:
        raise ValueError("episode_size and folds must both be at least two")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    train = _prepare(train_frame, config.max_train_rows, config.seed)
    test = _prepare(test_frame, config.max_test_rows, config.seed + 1)
    train_scores, test_scores = _oof_and_test_scores(train, test, config)
    train_events = _events(train, train_scores, "train_oof", config.episode_size)
    test_events = _events(test, test_scores, "official_test", config.episode_size)
    proxy_train = build_episodes(train_events)
    proxy_test = build_episodes(test_events)
    internal = split_episodes_by_group(proxy_train, train_fraction=0.80, validation_fraction=0.19, seed=config.seed)
    splits = EpisodeSplits(internal.train, internal.validation, proxy_test)
    wall = time.perf_counter() - wall_started
    cpu = time.process_time() - cpu_started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    metadata: dict[str, object] = {
        "dataset": "NSL-KDD",
        "evidence_kind": "real labelled flows; counterfactual IPS outcomes",
        "train_rows": len(train),
        "official_test_rows": len(test),
        "sequence_semantics": "ordered-row proxy; NSL-KDD has no timestamps",
        "detector_train_scoring": "out-of-fold",
        "detector_test_scoring": "fit on train only",
        "group_semantics": "family-pure fixed-size chunks; not hosts or campaigns",
    }
    resources = {
        "wall_time_s": wall,
        "cpu_time_s": cpu,
        "cpu_utilization_one_core_pct": 100 * cpu / max(wall, 1e-12),
        "max_rss_mb": float(rss),
        "train_rows_per_s": len(train) / max(wall, 1e-12),
    }
    return NslIpsEvidence(splits, train_events, test_events, metadata, resources)
