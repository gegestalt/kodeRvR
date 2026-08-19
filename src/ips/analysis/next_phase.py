"""Scientific controls for the post-POMDP IPS evidence phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability >= low) & (probability <= high if high == 1 else probability < high)
        if selected.any():
            total += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(total)


def source_leakage_curve(frame: pd.DataFrame, features: list[str], *, folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Quantify dataset identity in all, individual, and leave-one-out views."""
    required = {"source_dataset", "group_id", *features}
    if missing := required - set(frame):
        raise ValueError(f"missing provenance columns: {sorted(missing)}")
    views = [("all", "all", features)]
    views += [("single_feature", feature, [feature]) for feature in features]
    views += [("leave_one_feature_out", feature, [x for x in features if x != feature]) for feature in features if len(features) > 1]
    y = frame.source_dataset.astype(str).to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    splits = min(folds, len(np.unique(groups)))
    if splits < 2 or len(np.unique(y)) < 2:
        raise ValueError("source leakage curve needs two sources and two groups")
    rows = []
    for view, feature, columns in views:
        X = frame[columns].apply(pd.to_numeric, errors="coerce")
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=80, random_state=seed, class_weight="balanced", n_jobs=1))
        predicted = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(splits))
        rows.append({"representation": "policy_belief", "view": view, "feature": feature,
                     "n_features": len(columns), "balanced_accuracy": balanced_accuracy_score(y, predicted)})
    return pd.DataFrame(rows).sort_values(["view", "balanced_accuracy"], ascending=[True, False])


def detector_temporal_benchmark(events: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Train evidence-fusion detectors on train only; score validation/dev test."""
    required = {"split_role", "attack_present", "threat_probability", "anomaly_score", "timestamp"}
    if missing := required - set(events):
        raise ValueError(f"missing detector benchmark columns: {sorted(missing)}")
    work = events.copy()
    work["split_role"] = work.split_role.replace({"final_test": "development_test"})
    work = work.sort_values("timestamp", kind="stable")
    work["score_delta"] = work.groupby("split_role").threat_probability.diff().fillna(0)
    work["score_rolling_mean"] = work.groupby("split_role").threat_probability.transform(lambda x: x.rolling(5, min_periods=1).mean())
    columns = ["threat_probability", "anomaly_score", "score_delta", "score_rolling_mean"]
    train = work.split_role.eq("train")
    X_train = work.loc[train, columns]
    y_train = work.loc[train, "attack_present"].astype(int).to_numpy()
    models = {
        "logistic_fusion": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=80, random_state=seed)),
    }
    anomaly = make_pipeline(SimpleImputer(strategy="median"), IsolationForest(n_estimators=80, random_state=seed, n_jobs=1)).fit(X_train[y_train == 0])
    rows = []
    for role in ("validation", "development_test"):
        selected = work.split_role.eq(role)
        X, y = work.loc[selected, columns], work.loc[selected, "attack_present"].astype(int).to_numpy()
        duration_hours = max((work.loc[selected, "timestamp"].max() - work.loc[selected, "timestamp"].min()) / 3600, 1 / 3600)
        candidates = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            candidates[name] = model.predict_proba(X)[:, 1]
        raw = -anomaly.score_samples(X)
        candidates["benign_isolation_forest"] = (raw - raw.min()) / max(np.ptp(raw), 1e-12)
        for name, probability in candidates.items():
            prediction = probability >= .5
            rows.append({"detector": name, "calibration": "uncalibrated", "evaluation_role": role,
                         "pr_auc": average_precision_score(y, probability),
                         "attack_recall": recall_score(y, prediction, zero_division=0),
                         "brier": brier_score_loss(y, probability), "ece": _ece(y, probability),
                         "false_alarms_per_hour": float(((prediction == 1) & (y == 0)).sum() / duration_hours),
                         "missed_attacks_per_hour": float(((prediction == 0) & (y == 1)).sum() / duration_hours),
                         "rows": int(len(y))})
    return pd.DataFrame(rows)


def build_locked_holdout_manifest(path: Path, filenames: Iterable[str]) -> dict[str, object]:
    """Declare publication holdouts without loading labels or computing metrics."""
    files = []
    for name in filenames:
        candidate = path.parent / name
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
        files.append({"name": name, "sha256": digest, "available": candidate.exists()})
    manifest = {"status": "LOCKED_UNINSPECTED", "purpose": "one-shot final publication evaluation",
                "selection_rule": "complete later CSE days, selected before detector/policy freeze",
                "development_test": "2018-02-16 (repeatedly observed; not a final holdout)", "files": files,
                "unlock_condition": "detector, calibration, policy, reward, and analysis code frozen"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class InterventionLogger:
    """Append-only, hash-chained observed-intervention record writer."""
    REQUIRED = {"episode_id", "timestamp", "proposed_action", "executed_action", "behavior_probabilities",
                "shield_reason", "attack_continued", "attack_succeeded", "service_disrupted",
                "legitimate_sessions_affected", "affected_identities", "rollback_performed",
                "recovery_seconds", "time_to_containment_seconds"}

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, object]) -> dict[str, object]:
        if missing := self.REQUIRED - set(record):
            raise ValueError(f"missing intervention fields: {sorted(missing)}")
        clean = {key: value for key, value in record.items() if key not in {"record_hash", "previous_hash"}}
        previous = None
        if self.path.exists() and self.path.stat().st_size:
            previous = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])["record_hash"]
        clean["previous_hash"] = previous
        clean["record_hash"] = hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, sort_keys=True) + "\n")
        return clean


def consequence_trace(transitions: pd.DataFrame) -> pd.DataFrame:
    """Classify terminal outcomes so non-containment cannot disappear in aggregates."""
    required = {"episode_id", "attack_present", "contained", "compromised", "terminated", "truncated"}
    if missing := required - set(transitions):
        raise ValueError(f"missing consequence fields: {sorted(missing)}")
    terminal = transitions.groupby("episode_id", as_index=False).tail(1).copy()
    terminal["outcome"] = np.select(
        [terminal.contained, terminal.compromised, terminal.attack_present & terminal.truncated],
        ["contained", "compromised", "uncontained_truncated"], default="benign_or_unresolved")
    return terminal.groupby("outcome", as_index=False).agg(episodes=("episode_id", "nunique"))
