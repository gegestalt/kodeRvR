"""Group-safe calibrated provenance classifier with OOD abstention."""

from __future__ import annotations

from dataclasses import dataclass
import time
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

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
        started = time.perf_counter()
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
        probabilities = cross_val_predict(base, X, y, cv=splits, method="predict_proba")
        class_names = np.unique(y)
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
        precision, recall, per_f1, support = precision_recall_fscore_support(
            y, predictions, labels=class_names, zero_division=0
        )
        matrix = confusion_matrix(y, predictions, labels=class_names)
        one_hot = label_binarize(y, classes=class_names)
        if len(class_names) == 2:
            one_hot = np.column_stack((1 - one_hot[:, 0], one_hot[:, 0]))
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        confidence = probabilities.max(axis=1)
        correct = predictions == y
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            mask = (confidence >= lower) & (confidence < lower + 0.1)
            if mask.any():
                ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
        metrics = {
            "group_oof_macro_f1": float(f1_score(y, predictions, average="macro")),
            "group_oof_weighted_f1": float(f1_score(y, predictions, average="weighted")),
            "group_oof_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "group_oof_mcc": float(matthews_corrcoef(y, predictions)),
            "group_oof_log_loss": float(log_loss(y, probabilities, labels=class_names)),
            "group_oof_multiclass_brier": brier,
            "group_oof_ece_10bin": ece,
            "samples": float(len(y)),
            "groups": float(len(np.unique(groups))),
            "features": float(len(FEATURE_NAMES)),
            "fit_seconds": float(time.perf_counter() - started),
        }
        rng = np.random.default_rng(self.config.seed)
        unique_groups = np.unique(groups)
        bootstrap_scores = []
        for _ in range(200):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
            bootstrap_scores.append(f1_score(y[indices], predictions[indices], average="macro"))
        metrics["group_oof_macro_f1_ci_low"] = float(np.quantile(bootstrap_scores, 0.025))
        metrics["group_oof_macro_f1_ci_high"] = float(np.quantile(bootstrap_scores, 0.975))
        try:
            metrics["group_oof_roc_auc_ovr_macro"] = float(
                roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
            )
            metrics["group_oof_pr_auc_macro"] = float(
                average_precision_score(one_hot, probabilities, average="macro")
            )
        except ValueError:
            pass
        for index, name in enumerate(class_names):
            key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
            metrics[f"class_{key}_precision"] = float(precision[index])
            metrics[f"class_{key}_recall"] = float(recall[index])
            metrics[f"class_{key}_f1"] = float(per_f1[index])
            metrics[f"class_{key}_support"] = float(support[index])
            for column, predicted in enumerate(class_names):
                predicted_key = re.sub(r"[^a-z0-9]+", "_", str(predicted).lower()).strip("_")
                metrics[f"confusion_{key}_as_{predicted_key}"] = float(matrix[index, column])
        return metrics

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
