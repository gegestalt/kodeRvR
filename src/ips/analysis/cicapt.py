"""Detector evaluation for the chronological CICAPT APT campaign."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ips.analysis.next_phase import _ece


def benchmark_detectors(events: pd.DataFrame, features: list[str], *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-day models and evaluate validation/development-test only."""
    required = {"split_role", "attack_present", "attack_tactic", *features}
    if missing := required - set(events):
        raise ValueError(f"missing CICAPT benchmark columns: {sorted(missing)}")
    train = events.split_role.eq("train")
    X_train = events.loc[train, features].replace([np.inf, -np.inf], np.nan)
    y_train = events.loc[train, "attack_present"].astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        raise ValueError("CICAPT training day needs benign and attack rows")
    candidates = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample", random_state=seed, n_jobs=1)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=seed)),
    }
    fitted = {}; train_seconds = {}
    for name, model in candidates.items():
        started = time.perf_counter(); fitted[name] = model.fit(X_train, y_train); train_seconds[name] = time.perf_counter()-started
    anomaly = make_pipeline(SimpleImputer(strategy="median"), IsolationForest(n_estimators=100, contamination="auto", random_state=seed, n_jobs=1))
    started = time.perf_counter(); anomaly.fit(X_train[y_train == 0]); train_seconds["benign_isolation_forest"] = time.perf_counter()-started
    validation = events.split_role.eq("validation")
    raw_validation = -anomaly.score_samples(events.loc[validation, features].replace([np.inf, -np.inf], np.nan))
    anomaly_lo, anomaly_span = raw_validation.min(), max(np.ptp(raw_validation), 1e-12)
    rows=[]; family_rows=[]
    for role in ("validation", "development_test"):
        selected = events.split_role.eq(role)
        X = events.loc[selected, features].replace([np.inf, -np.inf], np.nan)
        y = events.loc[selected, "attack_present"].astype(int).to_numpy()
        probabilities = {name: model.predict_proba(X)[:, 1] for name, model in fitted.items()}
        probabilities["benign_isolation_forest"] = np.clip((-anomaly.score_samples(X)-anomaly_lo)/anomaly_span, 0, 1)
        for name, probability in probabilities.items():
            latency_started = time.perf_counter()
            _ = fitted[name].predict_proba(X)[:, 1] if name in fitted else -anomaly.score_samples(X)
            inference_seconds = time.perf_counter() - latency_started
            prediction = probability >= .5
            false_positives = int(((prediction == 1) & (y == 0)).sum())
            timestamps = pd.to_numeric(events.loc[selected, "timestamp"], errors="coerce") if "timestamp" in events else pd.Series(dtype=float)
            duration_hours = max(float(timestamps.max() - timestamps.min()) / 3600, 1 / 3600) if len(timestamps) else np.nan
            rows.append({"detector":name,"evaluation_role":role,"rows":len(y),"attacks":int(y.sum()),
                         "pr_auc":average_precision_score(y,probability),"balanced_accuracy":balanced_accuracy_score(y,prediction),
                         "precision":precision_score(y,prediction,zero_division=0),"recall":recall_score(y,prediction,zero_division=0),
                         "brier":brier_score_loss(y,probability),"ece":_ece(y,probability),"false_alarms_per_hour":false_positives/duration_hours,
                         "inference_latency_us_per_row":1e6*inference_seconds/max(len(y),1),"training_wall_s":train_seconds[name],
                         "evaluation_unit":"flow_row"})
            tactics = events.loc[selected, "attack_tactic"].astype(str).to_numpy()
            for tactic in sorted(set(tactics[y == 1])):
                mask = (tactics == tactic) & (y == 1)
                family_rows.append({"detector":name,"evaluation_role":role,"attack_tactic":tactic,
                                    "attack_rows":int(mask.sum()),"recall":float(prediction[mask].mean())})
    return pd.DataFrame(rows), pd.DataFrame(family_rows)


def detector_mistakes(events: pd.DataFrame, features: list[str], *, seed: int = 42, limit: int = 10) -> pd.DataFrame:
    """Return the most confident development errors with local feature clues."""
    train = events.split_role.eq("train"); development = events.split_role.eq("development_test")
    model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=seed))
    X_train = events.loc[train, features].replace([np.inf, -np.inf], np.nan)
    model.fit(X_train, events.loc[train, "attack_present"].astype(int))
    X = events.loc[development, features].replace([np.inf, -np.inf], np.nan)
    probability = model.predict_proba(X)[:, 1]
    truth = events.loc[development, "attack_present"].astype(bool).to_numpy()
    error = (probability >= .5) != truth
    confidence = np.where(truth, 1 - probability, probability)
    selected = np.flatnonzero(error)[np.argsort(confidence[error])[::-1][:limit]]
    centre = X_train.median(); scale = (X_train.quantile(.75) - X_train.quantile(.25)).replace(0, 1)
    rows = []
    source = events.loc[development].reset_index(drop=True); X_reset = X.reset_index(drop=True)
    for index in selected:
        unusual = ((X_reset.iloc[index] - centre).abs() / scale).sort_values(ascending=False).head(3).index.tolist()
        row = source.iloc[index]
        rows.append({"timestamp": row.get("timestamp"), "error_type": "false_negative" if truth[index] else "false_positive",
                     "true_attack": bool(truth[index]), "threat_probability": float(probability[index]),
                     "tactic": row.get("attack_tactic", "unknown"), "technique": row.get("attack_technique", "unknown"),
                     "unusual_features": ", ".join(unusual), "evaluation_unit": "flow_row"})
    return pd.DataFrame(rows)
