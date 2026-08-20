from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime

from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, ReviewAction, TrustDimension
from code_provenance.efficiency import EfficiencyMeasurement
from code_provenance.evidence import (
    AttestationLevel,
    EvidenceArtifact,
    EvidenceLedger,
    EvidenceTarget,
    artifact_content_hash,
)
from code_provenance.evidence_quality import (
    EvidenceQualityInput,
    EvidenceQualityStatus,
    evaluate_evidence_quality,
    load_evidence_quality,
)
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import TestEvidence as PytestEvidence


def quality(ood_score: float) -> EvidenceQualityInput:
    return EvidenceQualityInput(
        detector_id="code-embedding-ood:v1",
        ood_score=ood_score,
        context_coverage=0.95,
        schema_supported=True,
    )


def efficiency() -> EfficiencyMeasurement:
    return EfficiencyMeasurement(runtime_ms=100, peak_rss_mb=200, repeats=5)


def test_supported_high_coverage_input_passes_quality_gate():
    report = evaluate_evidence_quality(quality(0.2))

    assert report.status is EvidenceQualityStatus.PASS
    assert report.confidence == 0.8
    assert report.tags == frozenset()


def test_ood_input_fails_and_carries_abstention_tag():
    report = evaluate_evidence_quality(quality(0.85))

    assert report.status is EvidenceQualityStatus.FAIL
    assert report.tags == frozenset({"OOD_INPUT"})


def test_unsupported_evidence_is_unknown():
    report = evaluate_evidence_quality(EvidenceQualityInput(
        detector_id="detector:v1",
        ood_score=0.1,
        context_coverage=0.9,
        schema_supported=False,
    ))

    assert report.status is EvidenceQualityStatus.UNKNOWN
    assert report.confidence == 0.0
    assert report.tags >= {"SCHEMA_UNSUPPORTED"}


def test_quality_json_contract_is_reproducible(tmp_path: Path):
    path = tmp_path / "quality.json"
    path.write_text(
        '{"detector_id":"detector:v1","ood_score":0.2,"context_coverage":0.9,'
        '"schema_supported":true}',
        encoding="utf-8",
    )

    loaded = load_evidence_quality(path)

    assert loaded == EvidenceQualityInput("detector:v1", 0.2, 0.9, True)


def verified_ledger(snapshot) -> EvidenceLedger:
    target = EvidenceTarget(snapshot.repository_id, snapshot.snapshot_id, snapshot.head_sha)
    payload = '{"status":"passed"}'
    ledger = EvidenceLedger(target=target)
    ledger.add_artifact(EvidenceArtifact(
        artifact_id="ci:test",
        kind="test_report",
        producer="ci",
        producer_version="1.0",
        target=target,
        payload=payload,
        content_hash=artifact_content_hash(payload),
        attestation=AttestationLevel.OBSERVED,
        execution_id="fixture:1",
        complete=True,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
    ))
    return ledger


def observed_tests(snapshot_id: str, target_sha: str) -> PytestEvidence:
    return PytestEvidence(
        snapshot_id=snapshot_id,
        target_sha=target_sha,
        command=("python", "-m", "pytest", "-q"),
        framework="pytest",
        framework_version="fixture",
        discovered=1,
        selected=1,
        deselected=0,
        passed=1,
        failed=0,
        skipped=0,
        errors=0,
        xfailed=0,
        xpassed=0,
        collection_errors=0,
        interrupted=False,
        duration_seconds=0.1,
        exit_code=0,
        report_hash="c" * 64,
        complete=True,
        repository_changed=False,
        attestation=AttestationLevel.OBSERVED,
    )


def test_assessor_blocks_ood_and_accepts_in_distribution_evidence():
    root = Path(__file__).resolve().parents[1]
    snapshot = capture_code_snapshot(root)
    shared = dict(
        root=root,
        intent="Add explicit OOD evidence quality.",
        test_evidence=observed_tests(snapshot.snapshot_id, snapshot.head_sha),
        efficiency_baseline=efficiency(),
        efficiency_candidate=efficiency(),
        evidence_ledger=verified_ledger(snapshot),
    )

    accepted = PatchHealthAssessor().assess_repository(
        **shared,
        evidence_quality=quality(0.2),
    )
    blocked = PatchHealthAssessor().assess_repository(
        **shared,
        evidence_quality=quality(0.9),
    )

    assert accepted.dimension(TrustDimension.OOD_EVIDENCE_QUALITY).status is EvidenceStatus.PASS
    assert accepted.action is ReviewAction.ALLOW_STANDARD_REVIEW
    assert blocked.action is ReviewAction.BLOCK_PENDING_EVIDENCE
    assert "OOD_INPUT" in blocked.tags
