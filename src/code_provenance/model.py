"""Group-safe calibrated provenance classifier with OOD abstention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from code_provenance.features import FEATURE_NAMES, extract_features
from code_provenance.schema import AuthorshipLabel, CodeSample, ProvenanceEstimate


@dataclass(frozen=True)
class ModelConfig:
    folds: int = 5
    seed: int = 42
    confidence_threshold: float = 0.60
    ood_threshold: float = 0.70
    trees: int = 300


class ProvenanceClassifier:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.model: CalibratedClassifierCV | None = None
        self.ood: IsolationForest | None = None
        self.classes_: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self._ood_min = 0.0
        self._ood_range = 1.0

    @staticmethod
    def matrix(samples: list[CodeSample]) -> pd.DataFrame:
        return pd.DataFrame([extract_features(sample) for sample in samples], columns=FEATURE_NAMES)

    def fit(self, samples: list[CodeSample]) -> dict[str, float]:
        labelled = [sample for sample in samples if sample.label != AuthorshipLabel.UNKNOWN]
        if len(labelled) < 12:
            raise ValueError("at least 12 labelled samples are required")
        y = np.asarray([sample.label.value for sample in labelled])
        groups = np.asarray([sample.group_id for sample in labelled])
        if len(np.unique(y)) < 2:
            raise ValueError("at least two authorship classes are required")
        folds = min(self.config.folds, len(np.unique(groups)))
        if folds < 2:
            raise ValueError("at least two repository/author groups are required")
        X = self.matrix(labelled)
        base = RandomForestClassifier(
            n_estimators=self.config.trees, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=self.config.seed, n_jobs=1,
        )
        splits = list(GroupKFold(folds).split(X, y, groups))
        predictions = cross_val_predict(base, X, y, cv=splits)
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=splits)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.ood = IsolationForest(n_estimators=200, contamination="auto", random_state=self.config.seed, n_jobs=1)
        self.scaler = StandardScaler().fit(X)
        scaled = self.scaler.transform(X)
        self.ood.fit(scaled)
        raw = -self.ood.score_samples(scaled)
        self._ood_min, maximum = float(raw.min()), float(raw.max())
        self._ood_range = max(maximum - self._ood_min, 1e-9)
        return {"group_oof_macro_f1": float(f1_score(y, predictions, average="macro")), "samples": float(len(y)), "groups": float(len(np.unique(groups)))}

    def predict(self, sample: CodeSample, *, public_reuse_fraction: float = 0.0) -> ProvenanceEstimate:
        if self.model is None or self.ood is None or self.classes_ is None or self.scaler is None:
            raise RuntimeError("fit the classifier before prediction")
        X = self.matrix([sample])
        probabilities = self.model.predict_proba(X)[0]
        mapped = {label.value: 0.0 for label in AuthorshipLabel if label != AuthorshipLabel.UNKNOWN}
        mapped.update({str(name): float(value) for name, value in zip(self.classes_, probabilities)})
        total = sum(mapped.values())
        mapped = {name: value / total for name, value in mapped.items()}
        raw_ood = float(-self.ood.score_samples(self.scaler.transform(X))[0])
        ood_score = float(np.clip((raw_ood - self._ood_min) / self._ood_range, 0, 1))
        label, confidence = max(mapped.items(), key=lambda item: item[1])
        abstained = confidence < self.config.confidence_threshold or ood_score >= self.config.ood_threshold
        organic = None if abstained else float(np.clip(
            (mapped.get("human", 0) + .5 * mapped.get("hybrid", 0)) * (1 - public_reuse_fraction), 0, 1
        ))
        claim = (
            "Unknown/OOD: available evidence does not support an authorship estimate."
            if abstained else
            f"Statistically most consistent with {label}; this is not proof of authorship."
        )
        return ProvenanceEstimate(mapped, "unknown" if abstained else label, float(confidence),
                                  ood_score, abstained, public_reuse_fraction, organic, claim)
