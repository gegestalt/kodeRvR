"""Tests for calibration, drift, and operational IPS evidence."""

from __future__ import annotations

import numpy as np

from ips.real_policy_analysis import (
    expected_calibration_error,
    fit_probability_calibrator,
    population_stability_index,
)


def test_calibration_and_drift_metrics_are_finite():
    scores = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert 0 <= expected_calibration_error(labels, scores, bins=3) <= 1
    calibrator = fit_probability_calibrator(scores, labels, method="platt")
    calibrated = calibrator(np.array([0.2, 0.8]))
    assert calibrated.shape == (2,)
    assert np.all((0 <= calibrated) & (calibrated <= 1))
    assert population_stability_index(scores, np.clip(scores + 0.1, 0, 1)) >= 0


def test_calibrator_rejects_unknown_method():
    try:
        fit_probability_calibrator(np.array([0.1, 0.9]), np.array([0, 1]), method="bad")
    except ValueError as error:
        assert "method" in str(error)
    else:
        raise AssertionError("unknown calibration method must fail")
