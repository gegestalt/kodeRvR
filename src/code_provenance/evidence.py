"""Versioned evidence ledger with commit-bound lineage and integrity audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import json
from pathlib import Path
import re


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    SUPPORTED = "supported"
    SPECULATIVE = "speculative"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class AttestationLevel(StrEnum):
    ASSERTED = "asserted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    DEMONSTRATION = "demonstration"


class IntegrityStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    kind: str
    producer: str
    producer_version: str
    target_commit: str
    content_hash: str
    integrity_verified: bool
    created_at: datetime
    source_uri: str = ""

    def __post_init__(self) -> None:
        for name in ("artifact_id", "kind", "producer", "producer_version", "target_commit"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 digest")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    category: str
    severity: str
    location: str
    claim: str
    producer: str
    producer_version: str
    confidence: float
    verification: VerificationStatus
    evidence_refs: tuple[str, ...]
    counter_evidence_refs: tuple[str, ...]
    target_commit: str
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("claim_id", "category", "claim", "producer", "producer_version", "target_commit"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.severity not in {"info", "low", "medium", "high", "critical"}:
            raise ValueError("unsupported claim severity")
        if not 0 <= self.confidence <= 1:
            raise ValueError("claim confidence must be in [0, 1]")
        if not self.evidence_refs:
            raise ValueError("claims require at least one evidence ref")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class LedgerIntegrityReport:
    status: IntegrityStatus
    confidence: float
    failed_artifacts: tuple[str, ...]
    tags: frozenset[str]


class EvidenceLedger:
    """Append-only-in-use ledger for one immutable target commit."""

    schema_version = "1.0"

    def __init__(self, *, target_commit: str) -> None:
        if not target_commit.strip():
            raise ValueError("target_commit is required")
        self.target_commit = target_commit
        self._artifacts: dict[str, EvidenceArtifact] = {}
        self._claims: dict[str, EvidenceClaim] = {}

    def add_artifact(self, artifact: EvidenceArtifact) -> None:
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"duplicate artifact id: {artifact.artifact_id}")
        if artifact.target_commit != self.target_commit:
            raise ValueError("artifact target commit does not match ledger")
        self._artifacts[artifact.artifact_id] = artifact

    def add_claim(self, claim: EvidenceClaim) -> None:
        if claim.claim_id in self._claims:
            raise ValueError(f"duplicate claim id: {claim.claim_id}")
        if claim.target_commit != self.target_commit:
            raise ValueError("claim target commit does not match ledger")
        refs = set((*claim.evidence_refs, *claim.counter_evidence_refs))
        if missing := refs - set(self._artifacts):
            raise ValueError(f"missing evidence refs: {sorted(missing)}")
        self._claims[claim.claim_id] = claim

    def audit_integrity(self) -> LedgerIntegrityReport:
        if not self._artifacts:
            return LedgerIntegrityReport(
                IntegrityStatus.UNKNOWN,
                0.0,
                (),
                frozenset({"EVIDENCE_INTEGRITY_UNKNOWN"}),
            )
        failed = tuple(sorted(
            item.artifact_id for item in self._artifacts.values()
            if not item.integrity_verified
        ))
        if failed:
            return LedgerIntegrityReport(
                IntegrityStatus.FAIL,
                1.0,
                failed,
                frozenset({"EVIDENCE_INTEGRITY_FAILED"}),
            )
        return LedgerIntegrityReport(IntegrityStatus.PASS, 1.0, (), frozenset())

    def to_dict(self) -> dict[str, object]:
        artifacts = [self._artifact_dict(self._artifacts[key]) for key in sorted(self._artifacts)]
        claims = [self._claim_dict(self._claims[key]) for key in sorted(self._claims)]
        integrity = self.audit_integrity()
        return {
            "schema_version": self.schema_version,
            "target_commit": self.target_commit,
            "artifacts": artifacts,
            "claims": claims,
            "integrity": {
                "status": integrity.status.value,
                "confidence": integrity.confidence,
                "failed_artifacts": list(integrity.failed_artifacts),
                "tags": sorted(integrity.tags),
            },
        }

    @staticmethod
    def _artifact_dict(item: EvidenceArtifact) -> dict[str, object]:
        return {
            "artifact_id": item.artifact_id,
            "kind": item.kind,
            "producer": item.producer,
            "producer_version": item.producer_version,
            "target_commit": item.target_commit,
            "content_hash": item.content_hash,
            "integrity_verified": item.integrity_verified,
            "created_at": item.created_at.isoformat(),
            "source_uri": item.source_uri,
        }

    @staticmethod
    def _claim_dict(item: EvidenceClaim) -> dict[str, object]:
        return {
            "claim_id": item.claim_id,
            "category": item.category,
            "severity": item.severity,
            "location": item.location,
            "claim": item.claim,
            "producer": item.producer,
            "producer_version": item.producer_version,
            "confidence": item.confidence,
            "verification": item.verification.value,
            "evidence_refs": list(item.evidence_refs),
            "counter_evidence_refs": list(item.counter_evidence_refs),
            "target_commit": item.target_commit,
            "created_at": item.created_at.isoformat(),
        }


def load_evidence_ledger(path: Path) -> EvidenceLedger:
    """Load and revalidate a serialized ledger; stored audit output is ignored."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != EvidenceLedger.schema_version:
        raise ValueError("unsupported evidence ledger schema")
    if not isinstance(payload.get("artifacts"), list) or not isinstance(payload.get("claims"), list):
        raise ValueError("evidence ledger needs artifact and claim lists")
    ledger = EvidenceLedger(target_commit=str(payload.get("target_commit", "")))
    for values in payload["artifacts"]:
        if not isinstance(values, dict):
            raise ValueError("artifact entries must be objects")
        artifact = dict(values)
        artifact["created_at"] = datetime.fromisoformat(str(artifact["created_at"]))
        ledger.add_artifact(EvidenceArtifact(**artifact))
    for values in payload["claims"]:
        if not isinstance(values, dict):
            raise ValueError("claim entries must be objects")
        item = dict(values)
        item["created_at"] = datetime.fromisoformat(str(item["created_at"]))
        item["verification"] = VerificationStatus(str(item["verification"]))
        item["evidence_refs"] = tuple(item["evidence_refs"])
        item["counter_evidence_refs"] = tuple(item["counter_evidence_refs"])
        ledger.add_claim(EvidenceClaim(**item))
    return ledger
