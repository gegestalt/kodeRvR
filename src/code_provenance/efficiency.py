"""Measured efficiency-regression evidence for patch health decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path


class EfficiencyStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EfficiencyMeasurement:
    runtime_ms: float | None = None
    peak_rss_mb: float | None = None
    throughput_per_second: float | None = None
    repeats: int = 1

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("measurement repeats must be positive")
        for name in ("runtime_ms", "peak_rss_mb", "throughput_per_second"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EfficiencyBudget:
    max_runtime_regression: float = 0.20
    max_memory_regression: float = 0.20
    max_throughput_regression: float = 0.20
    minimum_repeats: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_runtime_regression",
            "max_memory_regression",
            "max_throughput_regression",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_repeats < 1:
            raise ValueError("minimum_repeats must be positive")


@dataclass(frozen=True)
class EfficiencyReport:
    status: EfficiencyStatus
    confidence: float
    deltas: dict[str, float]
    exceeded: tuple[str, ...]
    baseline_repeats: int
    candidate_repeats: int


def load_efficiency_evidence(path: Path) -> tuple[EfficiencyMeasurement, EfficiencyMeasurement]:
    """Load reproducible baseline and candidate measurements from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"baseline", "candidate"}:
        raise ValueError("efficiency evidence needs baseline and candidate objects")
    allowed = {"runtime_ms", "peak_rss_mb", "throughput_per_second", "repeats"}
    measurements = []
    for name in ("baseline", "candidate"):
        values = payload[name]
        if not isinstance(values, dict) or not set(values) <= allowed:
            raise ValueError(f"{name} efficiency evidence has unsupported fields")
        measurements.append(EfficiencyMeasurement(**values))
    return measurements[0], measurements[1]


def compare_efficiency(
    baseline: EfficiencyMeasurement,
    candidate: EfficiencyMeasurement,
    budget: EfficiencyBudget | None = None,
) -> EfficiencyReport:
    """Compare compatible measured metrics; negative throughput delta is worse."""
    budget = budget or EfficiencyBudget()
    deltas: dict[str, float] = {}
    for name in ("runtime_ms", "peak_rss_mb", "throughput_per_second"):
        before = getattr(baseline, name)
        after = getattr(candidate, name)
        if before is not None and after is not None:
            deltas[name] = (after - before) / before

    exceeded = []
    if deltas.get("runtime_ms", 0.0) > budget.max_runtime_regression:
        exceeded.append("runtime_ms")
    if deltas.get("peak_rss_mb", 0.0) > budget.max_memory_regression:
        exceeded.append("peak_rss_mb")
    if deltas.get("throughput_per_second", 0.0) < -budget.max_throughput_regression:
        exceeded.append("throughput_per_second")

    minimum_observed = min(baseline.repeats, candidate.repeats)
    confidence = min(1.0, minimum_observed / max(budget.minimum_repeats, 5))
    if not deltas or minimum_observed < budget.minimum_repeats:
        status = EfficiencyStatus.UNKNOWN
    elif exceeded:
        status = EfficiencyStatus.FAIL
    else:
        status = EfficiencyStatus.PASS
    return EfficiencyReport(
        status=status,
        confidence=confidence,
        deltas=deltas,
        exceeded=tuple(sorted(exceeded)),
        baseline_repeats=baseline.repeats,
        candidate_repeats=candidate.repeats,
    )
