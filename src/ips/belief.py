"""Observable belief-state estimates for the partially observable IPS."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def observable_belief(
    threat_probability: float,
    anomaly_score: float,
    score_history: Sequence[float],
) -> tuple[float, float, float]:
    """Estimate stage, compromise risk, and recent risk from detector history.

    Ground-truth attack labels/stages are intentionally absent from this API.
    """
    values = np.asarray(score_history[-8:] if score_history else [threat_probability], dtype=float)
    if not 0 <= threat_probability <= 1 or not 0 <= anomaly_score <= 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("belief inputs must be probabilities in [0, 1]")
    recent = float(values.mean())
    trend = float(max(0.0, values[-1] - values[0])) if len(values) > 1 else 0.0
    stage = float(np.clip(.55 * threat_probability + .35 * anomaly_score + .10 * trend, 0, 1))
    compromise = float(np.clip(.55 * stage + .25 * values.max() + .20 * anomaly_score, 0, 1))
    return stage, compromise, recent
