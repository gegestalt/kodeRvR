"""Pinned external-data acquisition and expected/observed evaluation contracts.

Fetched content is untrusted and is never executed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from code_provenance.security import scan_code
from code_provenance.testdata import TestFixtureRecord, validate_fixture_catalog


class AcquisitionFailureKind(StrEnum):
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SCHEMA_CHANGE = "schema_change"
    HASH_MISMATCH = "hash_mismatch"
    INVALID_REVISION = "invalid_source_revision"
    MALFORMED_PAYLOAD = "malformed_payload"


class ExternalDataError(RuntimeError):
    def __init__(self, kind: AcquisitionFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class AnalyzerOracleResult:
    fixture_id: str
    oracle: str
    detected: bool
    expected_detection: bool
    matched_rule_ids: tuple[str, ...]
    supported: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalyzerEvaluationSummary:
    total: int
    supported: int
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class ExternalEvaluationRecord:
    fixture_id: str
    expected: dict[str, object]
    observed: dict[str, object]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verified_payload(payload: bytes, expected_hash: str) -> bytes:
    observed = sha256_bytes(payload)
    if observed != expected_hash:
        raise ExternalDataError(
            AcquisitionFailureKind.HASH_MISMATCH,
            f"payload hash mismatch: expected {expected_hash}, observed {observed}",
        )
    return payload


def write_verified_cache(path: Path, payload: bytes, expected_hash: str, *, refresh: bool = False) -> None:
    verified_payload(payload, expected_hash)
    if path.exists() and not refresh:
        verified_payload(path.read_bytes(), expected_hash)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_catalog(path: Path) -> tuple[TestFixtureRecord, ...]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("catalog root is not a list")
        return validate_fixture_catalog(tuple(TestFixtureRecord.from_dict(row) for row in rows))
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ExternalDataError(AcquisitionFailureKind.SCHEMA_CHANGE, str(error)) from error


def validate_swebench_payload(row: dict[str, object]) -> None:
    required = {"instance_id", "repo", "base_commit", "problem_statement", "patch"}
    missing = sorted(key for key in required if not str(row.get(key, "")).strip())
    if missing:
        raise ExternalDataError(AcquisitionFailureKind.SCHEMA_CHANGE, f"SWE-bench fields missing: {missing}")


def validate_devgpt_payload(row: dict[str, object]) -> None:
    required = {"URL", "MentionedURL"}
    missing = sorted(key for key in required if not str(row.get(key, "")).strip())
    if missing:
        raise ExternalDataError(AcquisitionFailureKind.SCHEMA_CHANGE, f"DevGPT fields missing: {missing}")


def evaluate_codeql_fixture(
    fixture_id: str,
    oracle: str,
    content: str,
    *,
    expected_detection: bool,
    supported_rule_ids: tuple[str, ...],
) -> AnalyzerOracleResult:
    matched = tuple(sorted({signal.rule_id for signal in scan_code(content)}))
    supported = bool(supported_rule_ids)
    detected = bool(set(matched) & set(supported_rule_ids)) if supported else False
    notes = () if supported else ("oracle is outside the lightweight analyzer rule set",)
    return AnalyzerOracleResult(
        fixture_id, oracle, detected, expected_detection, matched, supported, notes
    )


def summarize_oracles(results: Iterable[AnalyzerOracleResult]) -> AnalyzerEvaluationSummary:
    rows = tuple(results)
    supported = tuple(row for row in rows if row.supported)
    tp = sum(row.expected_detection and row.detected for row in supported)
    fn = sum(row.expected_detection and not row.detected for row in supported)
    tn = sum(not row.expected_detection and not row.detected for row in supported)
    fp = sum(not row.expected_detection and row.detected for row in supported)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return AnalyzerEvaluationSummary(len(rows), len(supported), tp, fn, tn, fp, precision, recall, f1)


def evaluation_json(records: Iterable[ExternalEvaluationRecord]) -> str:
    return json.dumps([asdict(row) for row in records], indent=2, sort_keys=True) + "\n"


def assert_no_detection_regression(baseline: dict[str, object], current: AnalyzerEvaluationSummary) -> None:
    """Reject fewer supported detections; never mutates or rewrites the baseline."""
    expected_revision = str(baseline.get("revision", ""))
    if len(expected_revision) != 40:
        raise ValueError("evaluation baseline requires an immutable 40-character revision")
    prior = int(baseline.get("detected", -1))
    if prior < 0:
        raise ValueError("evaluation baseline requires a non-negative detected count")
    if current.true_positive < prior:
        raise AssertionError(
            f"security detection regression: baseline {prior}, current {current.true_positive}"
        )


def fetch_with_classified_failures(url: str, getter: Callable[..., object]) -> bytes:
    """Adapt a requests-like getter while preserving acquisition/product separation."""
    try:
        response = getter(url, timeout=60)
        status = int(getattr(response, "status_code", 200))
        if status == 429:
            raise ExternalDataError(AcquisitionFailureKind.RATE_LIMIT, f"rate limited: {url}")
        if status >= 500:
            raise ExternalDataError(AcquisitionFailureKind.UPSTREAM_UNAVAILABLE, f"upstream status {status}: {url}")
        if status >= 400:
            raise ExternalDataError(AcquisitionFailureKind.INVALID_REVISION, f"upstream status {status}: {url}")
        return bytes(getattr(response, "content"))
    except ExternalDataError:
        raise
    except TimeoutError as error:
        raise ExternalDataError(AcquisitionFailureKind.TIMEOUT, str(error)) from error
    except OSError as error:
        if "timeout" in error.__class__.__name__.lower():
            raise ExternalDataError(AcquisitionFailureKind.TIMEOUT, str(error)) from error
        raise ExternalDataError(AcquisitionFailureKind.UPSTREAM_UNAVAILABLE, str(error)) from error
