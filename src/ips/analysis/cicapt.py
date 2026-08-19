"""Detector evaluation for the chronological CICAPT APT campaign."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
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
            prediction = probability >= .5
            rows.append({"detector":name,"evaluation_role":role,"rows":len(y),"attacks":int(y.sum()),
                         "pr_auc":average_precision_score(y,probability),"balanced_accuracy":balanced_accuracy_score(y,prediction),
                         "precision":precision_score(y,prediction,zero_division=0),"recall":recall_score(y,prediction,zero_division=0),
                         "brier":brier_score_loss(y,probability),"ece":_ece(y,probability),"training_wall_s":train_seconds[name]})
            tactics = events.loc[selected, "attack_tactic"].astype(str).to_numpy()
            for tactic in sorted(set(tactics[y == 1])):
                mask = (tactics == tactic) & (y == 1)
                family_rows.append({"detector":name,"evaluation_role":role,"attack_tactic":tactic,
                                    "attack_rows":int(mask.sum()),"recall":float(prediction[mask].mean())})
    return pd.DataFrame(rows), pd.DataFrame(family_rows)
