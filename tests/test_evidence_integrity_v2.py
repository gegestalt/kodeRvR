from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys

from code_provenance.evidence import (
    AttestationLevel,
    EvidenceArtifact,
    EvidenceLedger,
    EvidenceTarget,
    IntegrityStatus,
    artifact_content_hash,
)
from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, ReviewAction, TrustDimension
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import (
    run_pytest_evidence,
    test_evidence_artifact as build_test_artifact,
)


def target() -> EvidenceTarget:
    return EvidenceTarget("repo:test", "snap_123", "a" * 40)


def artifact(payload: str, claimed_hash: str | None = None) -> EvidenceArtifact:
    return EvidenceArtifact(
        artifact_id="pytest:report",
        kind="test_report",
        producer="pytest",
        producer_version="9.1.1",
        target=target(),
        payload=payload,
        content_hash=claimed_hash or artifact_content_hash(payload),
        attestation=AttestationLevel.OBSERVED,
        execution_id="local:123",
        complete=True,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def test_integrity_is_computed_from_artifact_payload_not_asserted():
    ledger = EvidenceLedger(target=target())
    ledger.add_artifact(artifact('{"passed": 2}'))

    assert ledger.audit_integrity().status is IntegrityStatus.PASS


def test_tampered_payload_fails_independent_hash_verification():
    original = '{"passed": 2}'
    ledger = EvidenceLedger(target=target())
    ledger.add_artifact(artifact('{"passed": 999}', artifact_content_hash(original)))

    report = ledger.audit_integrity()

    assert report.status is IntegrityStatus.FAIL
    assert report.failed_artifacts == ("pytest:report",)


def test_asserted_or_incomplete_artifact_cannot_pass_integrity():
    asserted = artifact('{"passed": 2}')
    asserted = EvidenceArtifact(**{
        **asserted.__dict__,
        "attestation": AttestationLevel.ASSERTED,
    })
    ledger = EvidenceLedger(target=target())
    ledger.add_artifact(asserted)

    assert ledger.audit_integrity().status is IntegrityStatus.UNKNOWN


def test_artifact_target_must_match_complete_ledger_target():
    ledger = EvidenceLedger(target=target())
    wrong = EvidenceArtifact(**{
        **artifact("ok").__dict__,
        "target": EvidenceTarget("repo:test", "snap_other", "a" * 40),
    })

    try:
        ledger.add_artifact(wrong)
    except ValueError as error:
        assert "target" in str(error)
    else:
        raise AssertionError("mismatched target was accepted")


def test_real_pytest_artifact_verifies_and_tampering_blocks_assessment(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import VALUE\n\ndef test_value(): assert VALUE == 1\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    snapshot = capture_code_snapshot(tmp_path)
    tests = run_pytest_evidence(
        tmp_path, snapshot=snapshot, command=(sys.executable, "-m", "pytest", "-q")
    )
    produced = build_test_artifact(tests, repository_id=snapshot.repository_id)
    ledger = EvidenceLedger(target=produced.target)
    ledger.add_artifact(produced)

    accepted = PatchHealthAssessor().assess_repository(
        tmp_path, intent="Keep VALUE stable", test_evidence=tests, evidence_ledger=ledger
    )
    assert accepted.dimension(TrustDimension.EVIDENCE_INTEGRITY).status is EvidenceStatus.PASS

    tampered = EvidenceArtifact(**{**produced.__dict__, "payload": produced.payload + " "})
    corrupt = EvidenceLedger(target=produced.target)
    corrupt.add_artifact(tampered)
    blocked = PatchHealthAssessor().assess_repository(
        tmp_path, intent="Keep VALUE stable", test_evidence=tests, evidence_ledger=corrupt
    )
    assert blocked.action is ReviewAction.BLOCK_PENDING_EVIDENCE
