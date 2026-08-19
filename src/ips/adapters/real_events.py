"""Build canonical IPS events from timestamped CSV/Parquet network data.

Detector scores are out-of-fold by campaign/group: each row is scored by a
classifier and anomaly detector that did not fit on that row's group.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class AdapterConfig:
    timestamp_col: str
    label_col: str
    group_cols: tuple[str, ...]
    episode_col: str | None = None
    critical_col: str | None = None
    window_seconds: int = 300
    folds: int = 5
    max_rows: int | None = None
    seed: int = 42


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("source must be CSV or Parquet")


def normalize_family(label: object) -> str:
    """Normalize common labels into prevention-oriented families."""
    raw = str(label).strip()
    compact = "".join(ch for ch in raw.lower() if ch.isalnum())
    if compact in {"normal", "benign", "benigntraffic", "0"}:
        return "normal"
    if "brute" in compact or "password" in compact or "patator" in compact:
        return "BruteForce"
    if compact in {"r2l", "remote2local"} or any(
        token in compact for token in ("guesspasswd", "ftpwrite", "httptunnel")
    ):
        return "R2L"
    if compact in {"u2r", "user2root"} or any(
        token in compact for token in ("rootkit", "bufferoverflow", "privilege")
    ):
        return "U2R"
    if any(token in compact for token in ("exploit", "injection", "xss", "malware")):
        return "Exploitation"
    if any(token in compact for token in ("probe", "recon", "scan", "sweep")):
        return "Probe"
    if "dos" in compact or "flood" in compact or "slowloris" in compact:
        return "DoS"
    return raw or "unknown"


def _timestamp_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("timestamp column contains unparseable values")
    return parsed.astype("int64") / 1e9


def _validate_columns(frame: pd.DataFrame, config: AdapterConfig) -> None:
    required = {config.timestamp_col, config.label_col, *config.group_cols}
    if config.episode_col:
        required.add(config.episode_col)
    if config.critical_col:
        required.add(config.critical_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source is missing configured columns: {missing}")
    if not config.group_cols:
        raise ValueError("at least one group column is required to prevent leakage")


def build_real_events(frame: pd.DataFrame, config: AdapterConfig) -> pd.DataFrame:
    """Create canonical events with group-safe out-of-fold detector scores."""
    _validate_columns(frame, config)
    if config.folds < 2:
        raise ValueError("folds must be at least 2")
    work = frame.copy()
    if config.max_rows and len(work) > config.max_rows:
        work = work.sample(config.max_rows, random_state=config.seed).sort_index()
    work["_timestamp"] = _timestamp_seconds(work[config.timestamp_col])
    work["_group"] = work[list(config.group_cols)].astype(str).agg("|".join, axis=1)
    work["_family"] = work[config.label_col].map(normalize_family)
    work["_attack"] = (work["_family"] != "normal").astype(int)
    if work["_group"].nunique() < config.folds:
        raise ValueError("number of distinct groups must be at least folds")
    if work["_attack"].nunique() < 2:
        raise ValueError("source must contain benign and attack rows")

    excluded = {
        config.timestamp_col,
        config.label_col,
        *config.group_cols,
        *(filter(None, (config.episode_col, config.critical_col))),
        "_timestamp", "_group", "_family", "_attack",
    }
    feature_cols = [
        column for column in work.select_dtypes(include="number").columns
        if column not in excluded
    ]
    if not feature_cols:
        raise ValueError("no numeric detector features remain after leakage columns are excluded")
    X = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = work["_attack"].to_numpy()
    groups = work["_group"].to_numpy()
    probabilities = np.full(len(work), np.nan)
    anomaly = np.full(len(work), np.nan)
    splitter = GroupKFold(n_splits=config.folds)
    for train_idx, held_idx in splitter.split(X, y, groups):
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=100, random_state=config.seed),
        )
        classifier.fit(X.iloc[train_idx], y[train_idx])
        probabilities[held_idx] = classifier.predict_proba(X.iloc[held_idx])[:, 1]
        benign_train = train_idx[y[train_idx] == 0]
        if len(benign_train) < 2:
            anomaly[held_idx] = probabilities[held_idx]
        else:
            detector = make_pipeline(
                SimpleImputer(strategy="median"),
                IsolationForest(n_estimators=100, random_state=config.seed, n_jobs=1),
            )
            detector.fit(X.iloc[benign_train])
            raw = -detector.score_samples(X.iloc[held_idx])
            anomaly[held_idx] = (raw - raw.min()) / max(np.ptp(raw), 1e-12)
    if np.isnan(probabilities).any() or np.isnan(anomaly).any():
        raise RuntimeError("out-of-fold scoring left unscored rows")

    if config.episode_col:
        episode = work[config.episode_col].astype(str)
    else:
        window = (work["_timestamp"] // config.window_seconds).astype("int64")
        episode = work["_group"] + "|window=" + window.astype(str)
    ordered = work.assign(_episode=episode).sort_values(
        ["_episode", "_timestamp"], kind="stable"
    )
    positions = ordered.groupby("_episode").cumcount() + 1
    lengths = ordered.groupby("_episode")["_episode"].transform("size")
    stages = np.where(ordered["_attack"].to_numpy() == 1, positions / lengths, 0.0)
    critical = (
        ordered[config.critical_col].astype(bool).to_numpy()
        if config.critical_col else np.zeros(len(ordered), dtype=bool)
    )
    original_index = ordered.index.to_numpy()
    probability_map = pd.Series(probabilities, index=work.index)
    anomaly_map = pd.Series(anomaly, index=work.index)
    return pd.DataFrame(
        {
            "episode_id": ordered["_episode"].astype(str).to_numpy(),
            "group_id": ordered["_group"].astype(str).to_numpy(),
            "timestamp": ordered["_timestamp"].astype(float).to_numpy(),
            "threat_probability": probability_map.loc[original_index].to_numpy(),
            "anomaly_score": anomaly_map.loc[original_index].to_numpy(),
            "attack_present": ordered["_attack"].astype(bool).to_numpy(),
            "attack_stage": stages.astype(float),
            "critical_service": critical,
            "attack_family": ordered["_family"].astype(str).to_numpy(),
        }
    )


def write_events(source: Path, output: Path, config: AdapterConfig) -> pd.DataFrame:
    events = build_real_events(read_source(source), config)
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output, index=False)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/ips_events/events.parquet"))
    parser.add_argument("--timestamp-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--group-cols", required=True, help="comma-separated campaign/host/session columns")
    parser.add_argument("--episode-col")
    parser.add_argument("--critical-col")
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    config = AdapterConfig(
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        group_cols=tuple(part.strip() for part in args.group_cols.split(",") if part.strip()),
        episode_col=args.episode_col,
        critical_col=args.critical_col,
        window_seconds=args.window_seconds,
        folds=args.folds,
        max_rows=args.max_rows,
    )
    events = write_events(args.source, args.output, config)
    print(f"Wrote {len(events):,} events across {events.episode_id.nunique():,} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
