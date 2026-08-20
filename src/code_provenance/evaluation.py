"""Leakage-safe evaluation and audit metrics for provenance experiments."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from code_provenance.reuse import token_shingles
from code_provenance.schema import AuthorshipLabel, CodeSample


def validate_group_disjoint(train: Sequence[CodeSample], test: Sequence[CodeSample]) -> None:
    overlap = {item.group_id for item in train} & {item.group_id for item in test}
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap)}")


def _ece(probabilities: np.ndarray, labels: np.ndarray, class_names: Sequence[str]) -> float:
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(class_names)[probabilities.argmax(axis=1)]
    correct = predictions == labels
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            result += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return result


def audit_near_duplicate_leakage(samples: Sequence[CodeSample], *, width: int = 7) -> dict[str, object]:
    shingles = {item.sample_id: token_shingles(item.code, width) for item in samples}
    overlaps: list[float] = []
    cross_group = 0
    for left, right in combinations(samples, 2):
        if left.group_id == right.group_id:
            continue
        cross_group += 1
        left_shingles, right_shingles = shingles[left.sample_id], shingles[right.sample_id]
        union = left_shingles | right_shingles
        overlaps.append(len(left_shingles & right_shingles) / len(union) if union else 1.0)
    return {
        "width": width,
        "cross_group_pair_count": cross_group,
        "duplicate_pair_count": sum(value >= 0.95 for value in overlaps),
        "max_overlap": max(overlaps, default=0.0),
        "claim": "near-duplicate fingerprints identify split risk; they do not identify authorship",
    }


def evaluate_predictions(
    samples: Sequence[CodeSample],
    *,
    predicted_labels: Sequence[str],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    abstained: Sequence[bool] = (),
    ood_scores: Sequence[float] = (),
    public_reuse_fractions: Sequence[float] = (),
    ood_truth: Sequence[bool] = (),
) -> dict[str, object]:
    if len(samples) != len(predicted_labels) or probabilities.shape != (len(samples), len(class_names)):
        raise ValueError("prediction dimensions do not match samples and classes")
    labels = np.asarray([item.label.value for item in samples])
    predictions = np.asarray(predicted_labels)
    if any(item == AuthorshipLabel.UNKNOWN.value for item in labels):
        raise ValueError("evaluation requires labelled samples")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_names, zero_division=0
    )
    per_class = {
        name: {"precision": float(precision[index]), "recall": float(recall[index]),
               "f1": float(f1[index]), "support": int(support[index])}
        for index, name in enumerate(class_names)
    }
    per_language: dict[str, dict[str, float | int]] = {}
    for language in sorted({item.language for item in samples}):
        mask = np.asarray([item.language == language for item in samples])
        per_language[language] = {
            "samples": int(mask.sum()),
            "macro_f1": float(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[mask], predictions[mask])),
        }
    selected = np.ones(len(samples), dtype=bool) if not abstained else ~np.asarray(abstained, dtype=bool)
    errors = predictions != labels
    report: dict[str, object] = {
        "sample_count": len(samples),
        "group_count": len({item.group_id for item in samples}),
        "per_class": per_class,
        "per_language": per_language,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_names).tolist(),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "ece_10bin": _ece(probabilities, labels, class_names),
        "multiclass_brier": float(np.mean(np.sum((probabilities - (labels[:, None] == np.asarray(class_names))) ** 2, axis=1))),
        "selective_coverage": float(selected.mean()),
        "selective_risk": float(errors[selected].mean()) if selected.any() else 0.0,
        "human_false_positive_rate": float(np.mean(predictions[labels == "human"] != "human")) if np.any(labels == "human") else 0.0,
        "public_reuse_mean": float(np.mean(public_reuse_fractions)) if public_reuse_fractions else 0.0,
    }
    if ood_scores and ood_truth and len(ood_scores) == len(ood_truth):
        report["ood_auroc"] = float(roc_auc_score(ood_truth, ood_scores)) if len(set(ood_truth)) == 2 else None
    else:
        report["ood_auroc"] = None
    return report