"""Snapshot-bound evidence ledger with independently checked integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
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
class EvidenceTarget:
    repository_id: str
    snapshot_id: str
    head_sha: str

    def __post_init__(self) -> None:
        if not self.repository_id.strip() or not self.snapshot_id.strip() or not self.head_sha.strip():
            raise ValueError("repository_id, snapshot_id, and head_sha are required")


def artifact_content_hash(payload: str) -> str:
    """Return the digest independently recomputed by the integrity checker."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    kind: str
    producer: str
    producer_version: str
    target: EvidenceTarget
    payload: str
    content_hash: str
    attestation: AttestationLevel
    execution_id: str
    complete: bool
    created_at: datetime
    source_uri: str = ""

    def __post_init__(self) -> None:
        for name in ("artifact_id", "kind", "producer", "producer_version", "execution_id"):
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
    target: EvidenceTarget
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("claim_id", "category", "claim", "producer", "producer_version"):
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
    """Artifact ledger for one exact repository snapshot."""

    schema_version = "2.0"

    def __init__(self, *, target: EvidenceTarget) -> None:
        self.target = target
        self._artifacts: dict[str, EvidenceArtifact] = {}
        self._claims: dict[str, EvidenceClaim] = {}

    def add_artifact(self, artifact: EvidenceArtifact) -> None:
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"duplicate artifact id: {artifact.artifact_id}")
        if artifact.target != self.target:
            raise ValueError("artifact target does not match ledger target")
        self._artifacts[artifact.artifact_id] = artifact

    def add_claim(self, claim: EvidenceClaim) -> None:
        if claim.claim_id in self._claims:
            raise ValueError(f"duplicate claim id: {claim.claim_id}")
        if claim.target != self.target:
            raise ValueError("claim target does not match ledger target")
        refs = set((*claim.evidence_refs, *claim.counter_evidence_refs))
        if missing := refs - set(self._artifacts):
            raise ValueError(f"missing evidence refs: {sorted(missing)}")
        self._claims[claim.claim_id] = claim

    def audit_integrity(self) -> LedgerIntegrityReport:
        if not self._artifacts:
            return LedgerIntegrityReport(
                IntegrityStatus.UNKNOWN, 0.0, (), frozenset({"EVIDENCE_INTEGRITY_UNKNOWN"})
            )
        failed = tuple(sorted(
            item.artifact_id for item in self._artifacts.values()
            if artifact_content_hash(item.payload) != item.content_hash
        ))
        if failed:
            return LedgerIntegrityReport(
                IntegrityStatus.FAIL, 1.0, failed, frozenset({"EVIDENCE_INTEGRITY_FAILED"})
            )
        unverifiable = tuple(sorted(
            item.artifact_id for item in self._artifacts.values()
            if not item.complete
            or item.attestation in {AttestationLevel.ASSERTED, AttestationLevel.DEMONSTRATION}
        ))
        if unverifiable:
            return LedgerIntegrityReport(
                IntegrityStatus.UNKNOWN,
                0.0,
                (),
                frozenset({"EVIDENCE_ATTESTATION_INSUFFICIENT"}),
            )
        return LedgerIntegrityReport(IntegrityStatus.PASS, 0.9, (), frozenset())

    def to_dict(self) -> dict[str, object]:
        artifacts = [self._artifact_dict(self._artifacts[key]) for key in sorted(self._artifacts)]
        claims = [self._claim_dict(self._claims[key]) for key in sorted(self._claims)]
        integrity = self.audit_integrity()
        return {
            "schema_version": self.schema_version,
            "target": asdict(self.target),
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
        values = asdict(item)
        values["target"] = asdict(item.target)
        values["attestation"] = item.attestation.value
        values["created_at"] = item.created_at.isoformat()
        return values

    @staticmethod
    def _claim_dict(item: EvidenceClaim) -> dict[str, object]:
        values = asdict(item)
        values["target"] = asdict(item.target)
        values["verification"] = item.verification.value
        values["evidence_refs"] = list(item.evidence_refs)
        values["counter_evidence_refs"] = list(item.counter_evidence_refs)
        values["created_at"] = item.created_at.isoformat()
        return values


def _target(values: object) -> EvidenceTarget:
    if not isinstance(values, dict):
        raise ValueError("evidence target must be an object")
    return EvidenceTarget(**values)


def load_evidence_ledger(path: Path) -> EvidenceLedger:
    """Load v2 data and independently recompute integrity from payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != EvidenceLedger.schema_version:
        raise ValueError("unsupported evidence ledger schema")
    if not isinstance(payload.get("artifacts"), list) or not isinstance(payload.get("claims"), list):
        raise ValueError("evidence ledger needs artifact and claim lists")
    ledger = EvidenceLedger(target=_target(payload.get("target")))
    for values in payload["artifacts"]:
        if not isinstance(values, dict):
            raise ValueError("artifact entries must be objects")
        item = dict(values)
        item["target"] = _target(item["target"])
        item["created_at"] = datetime.fromisoformat(str(item["created_at"]))
        item["attestation"] = AttestationLevel(str(item["attestation"]))
        ledger.add_artifact(EvidenceArtifact(**item))
    for values in payload["claims"]:
        if not isinstance(values, dict):
            raise ValueError("claim entries must be objects")
        item = dict(values)
        item["target"] = _target(item["target"])
        item["created_at"] = datetime.fromisoformat(str(item["created_at"]))
        item["verification"] = VerificationStatus(str(item["verification"]))
        item["evidence_refs"] = tuple(item["evidence_refs"])
        item["counter_evidence_refs"] = tuple(item["counter_evidence_refs"])
        ledger.add_claim(EvidenceClaim(**item))
    return ledger
