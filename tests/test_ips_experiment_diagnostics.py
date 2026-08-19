"""Tests for deeper automated experiment diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ips.analysis.diagnostics import (
    expected_vs_observed,
    matched_outlier_diagnostics,
    parameter_importance,
    repeated_run_summary,
)


def _grid() -> pd.DataFrame:
    rows = []
    for width in (16, 32, 64):
        for batch in (16, 32, 64):
            rows.append(
                {
                    "name": f"h{width}-b{batch}", "hidden_dim": width,
                    "batch_size": batch, "gamma": 0.99, "learning_rate": 1e-3,
                    "target_update_steps": 75, "safety_quality": width / 100 - batch / 1000,
                    "runtime_s": width / 10 + batch / 100, "checkpoint_mb": width / 100,
                    "throughput_steps_s": 1000 / batch, "peak_python_memory_mb": width + batch,
                    "is_outlier": width == 64 and batch == 64,
                }
            )
    return pd.DataFrame(rows)


def test_parameter_importance_and_expectation_tables_are_dynamic():
    frame = _grid()
    importance = parameter_importance(frame)
    assert set(importance.parameter) >= {"hidden_dim", "batch_size"}
    assert set(importance.outcome) >= {"safety_quality", "runtime_s"}
    assert importance["evidence_strength"].eq("exploratory: n<30").all()
    checks = expected_vs_observed(frame)
    assert {"expectation", "observed", "status"} <= set(checks.columns)
    assert set(checks.status) <= {"supported", "contradicted", "inconclusive"}


def test_outliers_are_compared_with_matched_non_outlier_controls():
    diagnosed = matched_outlier_diagnostics(_grid())
    assert len(diagnosed) == 1
    assert diagnosed.iloc[0].matched_control
    assert diagnosed.iloc[0].most_unusual_parameter in {"hidden_dim", "batch_size"}
    assert np.isfinite(diagnosed.iloc[0].runtime_pct_vs_control)


def test_repeated_summary_has_ci_and_stability():
    runs = pd.DataFrame(
        {
            "policy": ["A"] * 5 + ["B"] * 5,
            "seed": [1, 2, 3, 4, 5] * 2,
            "mean_return": [1, 2, 1, 2, 1, 0, 0, 1, 0, 1],
            "containment_rate": [0.8] * 5 + [0.6] * 5,
        }
    )
    summary = repeated_run_summary(runs)
    assert set(summary.policy) == {"A", "B"}
    assert (summary.seed_count == 5).all()
    assert (summary.return_ci95_halfwidth >= 0).all()
    assert set(summary.stability) <= {"high", "moderate", "low"}
