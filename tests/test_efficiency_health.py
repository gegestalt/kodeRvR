from __future__ import annotations

from pathlib import Path

import pytest

from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, ReviewAction, TrustDimension
from code_provenance.efficiency import (
    EfficiencyBudget,
    EfficiencyMeasurement,
    EfficiencyStatus,
    compare_efficiency,
    load_efficiency_evidence,
)


def measurement(runtime: float, memory: float, throughput: float) -> EfficiencyMeasurement:
    return EfficiencyMeasurement(
        runtime_ms=runtime,
        peak_rss_mb=memory,
        throughput_per_second=throughput,
        repeats=5,
    )


def test_efficiency_comparison_passes_within_budget():
    report = compare_efficiency(
        measurement(100, 200, 50),
        measurement(108, 210, 48),
        EfficiencyBudget(max_runtime_regression=0.10, max_memory_regression=0.10, max_throughput_regression=0.10),
    )

    assert report.status is EfficiencyStatus.PASS
    assert report.deltas["runtime_ms"] == pytest.approx(0.08)
    assert report.deltas["throughput_per_second"] == pytest.approx(-0.04)


def test_efficiency_comparison_fails_and_names_exceeded_metrics():
    report = compare_efficiency(
        measurement(100, 200, 50),
        measurement(140, 260, 35),
    )

    assert report.status is EfficiencyStatus.FAIL
    assert report.exceeded == ("peak_rss_mb", "runtime_ms", "throughput_per_second")


def test_invalid_or_insufficient_measurements_do_not_create_false_precision():
    with pytest.raises(ValueError, match="positive"):
        EfficiencyMeasurement(runtime_ms=0, repeats=5)

    report = compare_efficiency(
        EfficiencyMeasurement(runtime_ms=100, repeats=1),
        EfficiencyMeasurement(runtime_ms=101, repeats=1),
    )
    assert report.status is EfficiencyStatus.UNKNOWN
    assert report.confidence == pytest.approx(0.2)


def test_assessor_consumes_efficiency_measurements():
    root = Path(__file__).resolve().parents[1]

    result = PatchHealthAssessor().assess_repository(
        root,
        intent="Preserve performance while adding health review.",
        tests_passed=True,
        efficiency_baseline=measurement(100, 200, 50),
        efficiency_candidate=measurement(105, 205, 49),
    )

    efficiency = result.dimension(TrustDimension.EFFICIENCY_RISK)
    assert efficiency.status is EvidenceStatus.PASS
    assert "efficiency_risk" not in result.missing_evidence


def test_efficiency_regression_requires_human_validation():
    root = Path(__file__).resolve().parents[1]
    result = PatchHealthAssessor().assess_repository(
        root,
        intent="Preserve performance while adding health review.",
        tests_passed=True,
        efficiency_baseline=measurement(100, 200, 50),
        efficiency_candidate=measurement(160, 300, 30),
    )

    assert result.action is ReviewAction.REQUIRE_HUMAN_REWRITE_OR_VALIDATION
    assert TrustDimension.EFFICIENCY_RISK in result.decision_path


def test_efficiency_evidence_file_has_a_reproducible_contract(tmp_path: Path):
    path = tmp_path / "efficiency.json"
    path.write_text(
        '{"baseline":{"runtime_ms":100,"peak_rss_mb":200,"repeats":5},'
        '"candidate":{"runtime_ms":105,"peak_rss_mb":205,"repeats":5}}',
        encoding="utf-8",
    )

    baseline, candidate = load_efficiency_evidence(path)

    assert baseline.runtime_ms == 100
    assert candidate.peak_rss_mb == 205
