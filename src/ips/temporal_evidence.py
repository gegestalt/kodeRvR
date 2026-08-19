"""Leakage-safe temporal detector evidence and shadow-policy logging.

The outer split is chronological by complete source day. Training scores are
group-out-of-fold by hour; validation and final-test scores come from a detector
fitted only on the earlier training day. Dataset replay remains counterfactual
and is rejected by the observed-outcome OPE gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from ips.actions import IpsAction
from ips.dataset import EpisodeSplits, build_episodes
from ips.real_data_adapter import normalize_family


@dataclass(frozen=True)
class TemporalDetectorConfig:
    timestamp_col: str = "Timestamp"
    label_col: str = "Label"
    source_day_col: str = "source_day"
    folds: int = 5
    max_iter: int = 60
    window_seconds: int = 300
    seed: int = 42


def read_cse_day_sample(
    path: Path,
    *,
    benign_rows: int = 20_000,
    attack_rows_per_family: int = 5_000,
    chunksize: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Read a bounded, class-aware sample while scanning an official daily CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    if benign_rows < 1 or attack_rows_per_family < 1 or chunksize < 1:
        raise ValueError("sampling limits must be positive")
    rng = np.random.default_rng(seed)
    benign_parts: list[pd.DataFrame] = []
    attack_parts: dict[str, list[pd.DataFrame]] = {}
    seen_benign = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if not {"Timestamp", "Label"} <= set(chunk.columns):
            raise ValueError(f"{path.name} lacks Timestamp/Label")
        # The official generated CSVs contain repeated header records mid-file.
        chunk = chunk[
            chunk["Timestamp"].astype(str).str.strip().ne("Timestamp")
            & chunk["Label"].astype(str).str.strip().ne("Label")
        ]
        labels = chunk["Label"].astype(str).str.strip()
        benign = chunk[labels.str.casefold().eq("benign")]
        if not benign.empty:
            # Reservoir-like bounded random priorities avoid taking only morning traffic.
            benign = benign.assign(_sample_key=rng.random(len(benign)))
            benign_parts.append(benign)
            seen_benign += len(benign)
            if seen_benign > benign_rows * 4:
                merged = pd.concat(benign_parts, ignore_index=True).nsmallest(benign_rows, "_sample_key")
                benign_parts, seen_benign = [merged], len(merged)
        for family, rows in chunk[~labels.str.casefold().eq("benign")].groupby("Label"):
            bucket = attack_parts.setdefault(str(family).strip(), [])
            bucket.append(rows.assign(_sample_key=rng.random(len(rows))))
            if sum(len(part) for part in bucket) > attack_rows_per_family * 2:
                attack_parts[family] = [
                    pd.concat(bucket, ignore_index=True).nsmallest(attack_rows_per_family, "_sample_key")
                ]
    selected = [pd.concat(benign_parts, ignore_index=True).nsmallest(benign_rows, "_sample_key")]
    selected.extend(
        pd.concat(parts, ignore_index=True).nsmallest(attack_rows_per_family, "_sample_key")
        for parts in attack_parts.values()
    )
    result = pd.concat(selected, ignore_index=True).drop(columns="_sample_key")
    timestamps = pd.to_datetime(
        result["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    if timestamps.isna().any():
        examples = result.loc[timestamps.isna(), "Timestamp"].astype(str).head(3).tolist()
        raise ValueError(f"unparseable official timestamps in {path.name}: {examples}")
    result["source_day"] = timestamps.dt.strftime("%Y-%m-%d")
    return result.sort_values("Timestamp", kind="stable").reset_index(drop=True)


def _classifier(config: TemporalDetectorConfig):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=config.max_iter, random_state=config.seed),
    )


def _anomaly(config: TemporalDetectorConfig):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=60, random_state=config.seed, n_jobs=1),
    )


def _probability(model, X: pd.DataFrame) -> np.ndarray:
    values = model.predict_proba(X)
    if values.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0]))
    return values[:, list(model.classes_).index(1)]


def build_temporal_detector_events(
    daily_frames: list[pd.DataFrame],
    config: TemporalDetectorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build canonical events using train-day OOF and past-only heldout scores."""
    config = config or TemporalDetectorConfig()
    if len(daily_frames) != 3:
        raise ValueError("exactly three chronological day frames are required")
    work = pd.concat([frame.copy() for frame in daily_frames], ignore_index=True)
    required = {config.timestamp_col, config.label_col, config.source_day_col}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"missing temporal source columns: {sorted(missing)}")
    work["_timestamp"] = pd.to_datetime(
        work[config.timestamp_col], format="%d/%m/%Y %H:%M:%S", errors="raise", utc=True
    )
    days = sorted(work[config.source_day_col].astype(str).unique())
    if len(days) != 3:
        raise ValueError("three distinct source days are required")
    roles = {days[0]: "train", days[1]: "validation", days[2]: "final_test"}
    work["split_role"] = work[config.source_day_col].astype(str).map(roles)
    work["_hour_group"] = work[config.source_day_col].astype(str) + "|hour=" + work["_timestamp"].dt.hour.astype(str)
    work["_family"] = work[config.label_col].map(normalize_family)
    work["_attack"] = (work["_family"] != "normal").astype(int)
    excluded = required | {"_timestamp", "_hour_group", "_family", "_attack", "split_role"}
    candidate_features = [column for column in work.columns if column not in excluded]
    numeric = work[candidate_features].apply(pd.to_numeric, errors="coerce")
    features = [column for column in numeric.columns if numeric[column].notna().any()]
    if not features:
        raise ValueError("no numeric detector features available")
    X = numeric[features].replace([np.inf, -np.inf], np.nan)
    y = work["_attack"].to_numpy()
    train_idx = np.flatnonzero(work["split_role"].eq("train").to_numpy())
    held_idx = np.flatnonzero(~work["split_role"].eq("train").to_numpy())
    groups = work.iloc[train_idx]["_hour_group"].to_numpy()
    distinct_groups = np.unique(groups)
    if len(distinct_groups) < config.folds:
        raise ValueError("training day has fewer hour groups than folds")
    probabilities = np.full(len(work), np.nan)
    anomaly_scores = np.full(len(work), np.nan)
    for local_train, local_held in GroupKFold(config.folds).split(X.iloc[train_idx], y[train_idx], groups):
        fit_idx, score_idx = train_idx[local_train], train_idx[local_held]
        if np.unique(y[fit_idx]).size < 2:
            probabilities[score_idx] = y[fit_idx].mean()
        else:
            model = _classifier(config).fit(X.iloc[fit_idx], y[fit_idx])
            probabilities[score_idx] = _probability(model, X.iloc[score_idx])
        benign = fit_idx[y[fit_idx] == 0]
        if len(benign) < 2:
            anomaly_scores[score_idx] = probabilities[score_idx]
        else:
            detector = _anomaly(config).fit(X.iloc[benign])
            raw = -detector.score_samples(X.iloc[score_idx])
            anomaly_scores[score_idx] = (raw - raw.min()) / max(np.ptp(raw), 1e-12)
    full_model = _classifier(config).fit(X.iloc[train_idx], y[train_idx])
    probabilities[held_idx] = _probability(full_model, X.iloc[held_idx])
    benign_train = train_idx[y[train_idx] == 0]
    detector = _anomaly(config).fit(X.iloc[benign_train])
    raw_train = -detector.score_samples(X.iloc[train_idx])
    lo, span = raw_train.min(), max(np.ptp(raw_train), 1e-12)
    anomaly_scores[held_idx] = np.clip((-detector.score_samples(X.iloc[held_idx]) - lo) / span, 0, 1)
    if np.isnan(probabilities).any() or np.isnan(anomaly_scores).any():
        raise RuntimeError("temporal scoring left rows unscored")
    # Explicit conversion avoids pandas datetime64[us] vs datetime64[ns] unit drift.
    seconds = work["_timestamp"].map(lambda value: value.timestamp()).astype(float)
    window = (seconds // config.window_seconds).astype("int64")
    episode = work["_hour_group"] + "|window=" + window.astype(str)
    ordered = work.assign(_episode=episode, _seconds=seconds).sort_values(["_episode", "_seconds"], kind="stable")
    index = ordered.index.to_numpy()
    duplicate_rank = ordered.groupby(["_episode", "_seconds"]).cumcount().to_numpy()
    unique_seconds = ordered["_seconds"].to_numpy(float) + duplicate_rank * 1e-6
    positions = ordered.groupby("_episode").cumcount() + 1
    lengths = ordered.groupby("_episode")["_episode"].transform("size")
    events = pd.DataFrame({
        "episode_id": ordered["_episode"].to_numpy(),
        "group_id": ordered["_hour_group"].to_numpy(),
        "timestamp": unique_seconds,
        "threat_probability": probabilities[index],
        "anomaly_score": anomaly_scores[index],
        "attack_present": ordered["_attack"].to_numpy(bool),
        "attack_stage": np.where(ordered["_attack"].to_numpy() == 1, positions / lengths, 0.0),
        "critical_service": np.zeros(len(ordered), dtype=bool),
        "attack_family": ordered["_family"].to_numpy(str),
        "split_role": ordered["split_role"].to_numpy(str),
        "source_day": ordered[config.source_day_col].to_numpy(str),
        "score_origin": np.where(ordered["split_role"].eq("train"), "train_group_oof", "train_fitted_heldout"),
    })
    audit = work.groupby([config.source_day_col, "split_role", "_family"], as_index=False).size().rename(columns={config.source_day_col: "source_day", "_family": "attack_family", "size": "rows", "split_role": "role"})
    return events, audit


def split_events_by_role(events: pd.DataFrame, *, max_events: int | None = None) -> EpisodeSplits:
    """Build episodes using the fixed chronological role column."""
    required = {"split_role", "source_day"}
    if not required <= set(events):
        raise ValueError("events lack chronological role metadata")
    role_days = events.groupby("split_role")["source_day"].nunique()
    if not {"train", "validation", "final_test"} <= set(role_days.index):
        raise ValueError("all chronological roles are required")
    return EpisodeSplits(*(
        build_episodes(events[events.split_role.eq(role)], max_events=max_events)
        for role in ("train", "validation", "final_test")
    ))


def log_shadow_decision(
    *,
    episode_id: str,
    timestamp: float,
    proposed: IpsAction,
    executed: IpsAction,
    action_mask: np.ndarray,
    epsilon: float,
    evidence_kind: str,
    observed_reward: float | None = None,
) -> dict[str, object]:
    """Create an auditable epsilon-soft shadow-policy decision record."""
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (len(IpsAction),) or not mask.any() or not 0 <= epsilon <= 1:
        raise ValueError("invalid action mask or epsilon")
    valid = np.flatnonzero(mask)
    probabilities = np.zeros(len(IpsAction), dtype=float)
    probabilities[valid] = epsilon / len(valid)
    preferred = proposed if mask[int(proposed)] else IpsAction.MONITOR
    probabilities[int(preferred)] += 1 - epsilon
    return {
        "episode_id": episode_id,
        "timestamp": float(timestamp),
        "action": int(executed),
        "proposed_action": proposed.name,
        "executed_action": executed.name,
        "behavior_propensity": float(probabilities[int(executed)]),
        "behavior_probabilities": probabilities.tolist(),
        "valid_action_mask": mask.tolist(),
        "observed_reward": observed_reward,
        "evidence_kind": evidence_kind,
    }


def require_observed_ope_evidence(log: pd.DataFrame) -> None:
    """Fail closed before WIS/DR unless genuine observed shadow outcomes exist."""
    required = {"evidence_kind", "behavior_propensity", "observed_reward", "action"}
    missing = required - set(log)
    if missing:
        raise ValueError(f"OPE log is missing columns: {sorted(missing)}")
    if not log["evidence_kind"].eq("observed_shadow_deployment").all():
        raise ValueError("doubly robust OPE requires observed shadow deployment evidence")
    if log[list(required - {"evidence_kind"})].isna().any().any():
        raise ValueError("OPE log contains missing propensities, actions, or observed outcomes")
    if not log["behavior_propensity"].between(0, 1, inclusive="right").all():
        raise ValueError("behavior propensities must be in (0, 1]")
