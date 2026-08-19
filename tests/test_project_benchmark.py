"""Fast tests for the project showcase report."""

from __future__ import annotations

import project_benchmark as PB


def test_saved_benchmarks_report_missing_or_best_metric():
    rows = PB.saved_benchmark_rows()
    assert rows
    assert all(row["status"] in {"available", "missing"} for row in rows)


def test_benchmark_markdown_has_core_sections():
    report = {
        "dependencies": {"python": "test"},
        "datasets": [],
        "features": [],
        "tracks": [],
        "model_training": [],
        "metric_definitions": [],
        "saved_benchmarks": [],
        "ips_baselines": [],
        "tests": None,
        "dqn_smoke": None,
    }
    text = PB.render_markdown(report)
    assert "Project Health" in text
    assert "Adaptive IPS" in text
    assert "How metrics are calculated" in text
    assert "Comparison rules" in text
    assert "Interpretation boundary" in text
