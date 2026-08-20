from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from code_provenance.assessment import (
    AssessmentRequest, EvidenceFinding, EvidenceStatus, PatchHealthAssessor,
    ReviewAction, TrustDimension,
)
from code_provenance.evidence import (
    AttestationLevel, EvidenceArtifact, EvidenceClaim, EvidenceLedger,
    EvidenceTarget, VerificationStatus, artifact_content_hash,
    load_evidence_ledger,
)


TARGET = EvidenceTarget("repo:test", "snap_abc", "a" * 40)
PAYLOAD = '{"diff":"+ secure = true"}'


def artifact(**changes: object) -> EvidenceArtifact:
    values = {
        "artifact_id": "diff:hunk:193", "kind": "diff_hunk",
        "producer": "git-adapter", "producer_version": "1.0.0",
        "target": TARGET, "payload": PAYLOAD,
        "content_hash": artifact_content_hash(PAYLOAD),
        "attestation": AttestationLevel.OBSERVED, "execution_id": "git:1",
        "complete": True, "created_at": datetime(2026, 8, 20, tzinfo=UTC),
    }
    values.update(changes)
    return EvidenceArtifact(**values)


def claim(**changes: object) -> EvidenceClaim:
    values = {
        "claim_id": "finding:auth-bypass", "category": "security.authentication",
        "severity": "high", "location": "src/auth.py:84",
        "claim": "Authentication may be bypassed.", "producer": "security-reviewer",
        "producer_version": "3.0.0", "confidence": 0.91,
        "verification": VerificationStatus.SUPPORTED,
        "evidence_refs": ("diff:hunk:193",), "counter_evidence_refs": (),
        "target": TARGET, "created_at": datetime(2026, 8, 20, tzinfo=UTC),
    }
    values.update(changes)
    return EvidenceClaim(**values)


def test_ledger_preserves_claim_lineage_and_versions():
    ledger = EvidenceLedger(target=TARGET)
    ledger.add_artifact(artifact())
    ledger.add_claim(claim())
    payload = ledger.to_dict()
    assert payload["schema_version"] == "2.0"
    assert payload["claims"][0]["evidence_refs"] == ["diff:hunk:193"]


def test_claim_rejects_missing_refs_and_mismatched_target():
    ledger = EvidenceLedger(target=TARGET)
    with pytest.raises(ValueError, match="missing evidence refs"):
        ledger.add_claim(claim())
    wrong = EvidenceTarget("repo:test", "snap_wrong", "a" * 40)
    with pytest.raises(ValueError, match="target"):
        ledger.add_artifact(artifact(target=wrong))


def test_ledger_rejects_duplicate_identifiers():
    ledger = EvidenceLedger(target=TARGET)
    ledger.add_artifact(artifact())
    with pytest.raises(ValueError, match="duplicate artifact"):
        ledger.add_artifact(artifact())


def test_integrity_failure_blocks_patch_assessment():
    finding = EvidenceFinding(
        TrustDimension.EVIDENCE_INTEGRITY, EvidenceStatus.FAIL, 1.0, "critical",
        "Artifact hash mismatch.", ("artifact:ci:wrong-sha",),
        frozenset({"EVIDENCE_INTEGRITY_FAILED"}),
    )
    result = PatchHealthAssessor().assess(
        AssessmentRequest("pr:12", "pull_request", (finding,))
    )
    assert result.action is ReviewAction.BLOCK_PENDING_EVIDENCE
    assert result.decision_path == (
        TrustDimension.EVIDENCE_INTEGRITY, TrustDimension.EVIDENCE_SUFFICIENCY,
    )


def test_ledger_json_round_trip_recomputes_integrity(tmp_path: Path):
    original = EvidenceLedger(target=TARGET)
    original.add_artifact(artifact())
    original.add_claim(claim())
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    restored = load_evidence_ledger(path)
    assert restored.to_dict() == original.to_dict()


def test_loaded_tampered_payload_fails_integrity(tmp_path: Path):
    ledger = EvidenceLedger(target=TARGET)
    ledger.add_artifact(artifact())
    serialized = ledger.to_dict()
    serialized["artifacts"][0]["payload"] = "tampered"
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(serialized), encoding="utf-8")
    restored = load_evidence_ledger(path)
    assert restored.audit_integrity().status.value == "fail"
