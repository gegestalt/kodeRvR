"""Calibration, drift, escalation, and operational metrics for IPS policies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Weighted absolute confidence/accuracy gap over equal-width bins."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.size == 0 or bins < 2:
        raise ValueError("labels/scores must be equal, non-empty; bins must be >= 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            total += selected.mean() * abs(labels[selected].mean() - scores[selected].mean())
    return float(total)


def fit_probability_calibrator(
    scores: np.ndarray, labels: np.ndarray, *, method: str
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit Platt or isotonic calibration on validation-only observations."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.shape != y.shape or x.size < 2 or np.unique(y).size != 2:
        raise ValueError("calibration requires equal scores/labels with both classes")
    if method == "platt":
        model = LogisticRegression(random_state=42).fit(x.reshape(-1, 1), y)
        return lambda values: model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        return lambda values: np.asarray(model.predict(np.asarray(values)), dtype=float)
    raise ValueError("method must be 'platt' or 'isotonic'")


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if not reference.size or not current.size or bins < 2:
        raise ValueError("reference/current must be non-empty; bins must be >= 2")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def escalation_decision(probability: float, *, uncertainty_threshold: float = 0.60) -> bool:
    """Request review for ambiguous, high-uncertainty detector outputs."""
    uncertainty = 1.0 - abs(float(probability) - 0.5) * 2.0
    return 0.30 <= probability <= 0.80 and uncertainty >= uncertainty_threshold
