"""Explicit OOD and evidence-integrity gate for patch-health decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path


class EvidenceQualityStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceQualityInput:
    detector_id: str
    ood_score: float
    context_coverage: float
    schema_supported: bool
    integrity_verified: bool

    def __post_init__(self) -> None:
        if not self.detector_id.strip():
            raise ValueError("detector_id is required")
        if not 0 <= self.ood_score <= 1:
            raise ValueError("ood_score must be in [0, 1]")
        if not 0 <= self.context_coverage <= 1:
            raise ValueError("context_coverage must be in [0, 1]")


@dataclass(frozen=True)
class EvidenceQualityReport:
    status: EvidenceQualityStatus
    confidence: float
    detector_id: str
    ood_score: float
    context_coverage: float
    tags: frozenset[str]


def evaluate_evidence_quality(
    evidence: EvidenceQualityInput,
    *,
    ood_threshold: float = 0.70,
    minimum_context_coverage: float = 0.80,
) -> EvidenceQualityReport:
    tags: set[str] = set()
    if not evidence.schema_supported:
        tags.add("SCHEMA_UNSUPPORTED")
    if not evidence.integrity_verified:
        tags.add("EVIDENCE_INTEGRITY_UNVERIFIED")
    if evidence.context_coverage < minimum_context_coverage:
        tags.add("REPOSITORY_CONTEXT_PARTIAL")
    if evidence.ood_score >= ood_threshold:
        tags.add("OOD_INPUT")

    unverifiable = tags & {
        "SCHEMA_UNSUPPORTED",
        "EVIDENCE_INTEGRITY_UNVERIFIED",
        "REPOSITORY_CONTEXT_PARTIAL",
    }
    if unverifiable:
        status, confidence = EvidenceQualityStatus.UNKNOWN, 0.0
    elif "OOD_INPUT" in tags:
        status, confidence = EvidenceQualityStatus.FAIL, 1.0 - evidence.ood_score
    else:
        status, confidence = EvidenceQualityStatus.PASS, 1.0 - evidence.ood_score
    return EvidenceQualityReport(
        status=status,
        confidence=float(confidence),
        detector_id=evidence.detector_id,
        ood_score=evidence.ood_score,
        context_coverage=evidence.context_coverage,
        tags=frozenset(tags),
    )


def load_evidence_quality(path: Path) -> EvidenceQualityInput:
    """Load a named detector's OOD and evidence-integrity artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "detector_id",
        "ood_score",
        "context_coverage",
        "schema_supported",
        "integrity_verified",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"evidence quality needs exactly: {sorted(required)}")
    return EvidenceQualityInput(**payload)
