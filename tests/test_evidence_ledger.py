from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from code_provenance.assessment import (
    AssessmentRequest,
    EvidenceFinding,
    EvidenceStatus,
    PatchHealthAssessor,
    ReviewAction,
    TrustDimension,
)
from code_provenance.evidence import (
    EvidenceArtifact,
    EvidenceClaim,
    EvidenceLedger,
    IntegrityStatus,
    VerificationStatus,
    load_evidence_ledger,
)


HASH = "a" * 64


def artifact(**changes: object) -> EvidenceArtifact:
    values = {
        "artifact_id": "diff:hunk:193",
        "kind": "diff_hunk",
        "producer": "git-adapter",
        "producer_version": "1.0.0",
        "target_commit": "abc123",
        "content_hash": HASH,
        "integrity_verified": True,
        "created_at": datetime(2026, 8, 20, tzinfo=UTC),
    }
    values.update(changes)
    return EvidenceArtifact(**values)


def claim(**changes: object) -> EvidenceClaim:
    values = {
        "claim_id": "finding:auth-bypass",
        "category": "security.authentication",
        "severity": "high",
        "location": "src/auth/middleware.py:84-97",
        "claim": "Authentication may be bypassed when the token is absent.",
        "producer": "security-reviewer",
        "producer_version": "3.0.0",
        "confidence": 0.91,
        "verification": VerificationStatus.SUPPORTED,
        "evidence_refs": ("diff:hunk:193",),
        "counter_evidence_refs": (),
        "target_commit": "abc123",
        "created_at": datetime(2026, 8, 20, tzinfo=UTC),
    }
    values.update(changes)
    return EvidenceClaim(**values)


def test_ledger_preserves_claim_lineage_and_versions():
    ledger = EvidenceLedger(target_commit="abc123")
    ledger.add_artifact(artifact())
    ledger.add_claim(claim())

    payload = ledger.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["claims"][0]["evidence_refs"] == ["diff:hunk:193"]
    assert payload["claims"][0]["producer_version"] == "3.0.0"


def test_claim_cannot_reference_missing_or_wrong_commit_evidence():
    ledger = EvidenceLedger(target_commit="abc123")

    with pytest.raises(ValueError, match="missing evidence refs"):
        ledger.add_claim(claim())
    with pytest.raises(ValueError, match="target commit"):
        ledger.add_artifact(artifact(target_commit="different"))


def test_ledger_rejects_duplicate_identifiers():
    ledger = EvidenceLedger(target_commit="abc123")
    ledger.add_artifact(artifact())

    with pytest.raises(ValueError, match="duplicate artifact"):
        ledger.add_artifact(artifact())


def test_integrity_audit_distinguishes_verified_failed_and_missing():
    verified = EvidenceLedger(target_commit="abc123")
    verified.add_artifact(artifact())
    missing = EvidenceLedger(target_commit="abc123")
    failed = EvidenceLedger(target_commit="abc123")
    failed.add_artifact(artifact(integrity_verified=False))

    assert verified.audit_integrity().status is IntegrityStatus.PASS
    assert missing.audit_integrity().status is IntegrityStatus.UNKNOWN
    assert failed.audit_integrity().status is IntegrityStatus.FAIL


def test_integrity_failure_blocks_patch_assessment():
    integrity = EvidenceFinding(
        TrustDimension.EVIDENCE_INTEGRITY,
        EvidenceStatus.FAIL,
        1.0,
        "critical",
        "Evidence artifact does not match the target commit.",
        ("artifact:ci:wrong-sha",),
        frozenset({"EVIDENCE_INTEGRITY_FAILED"}),
    )
    result = PatchHealthAssessor().assess(AssessmentRequest(
        "pr:12",
        "pull_request",
        (integrity,),
    ))

    assert result.action is ReviewAction.BLOCK_PENDING_EVIDENCE
    assert result.decision_path == (
        TrustDimension.EVIDENCE_INTEGRITY,
        TrustDimension.EVIDENCE_SUFFICIENCY,
    )


def test_ledger_json_round_trip_preserves_lineage(tmp_path: Path):
    original = EvidenceLedger(target_commit="abc123")
    original.add_artifact(artifact())
    original.add_claim(claim())
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")

    restored = load_evidence_ledger(path)

    assert restored.to_dict() == original.to_dict()
