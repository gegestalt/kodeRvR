from __future__ import annotations

from pathlib import Path

from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, ReviewAction, TrustDimension
from code_provenance.efficiency import EfficiencyMeasurement
from code_provenance.evidence_quality import (
    EvidenceQualityInput,
    EvidenceQualityStatus,
    evaluate_evidence_quality,
    load_evidence_quality,
)


def quality(ood_score: float) -> EvidenceQualityInput:
    return EvidenceQualityInput(
        detector_id="code-embedding-ood:v1",
        ood_score=ood_score,
        context_coverage=0.95,
        schema_supported=True,
        integrity_verified=True,
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


def test_unsupported_or_unverified_evidence_is_unknown():
    report = evaluate_evidence_quality(EvidenceQualityInput(
        detector_id="detector:v1",
        ood_score=0.1,
        context_coverage=0.9,
        schema_supported=False,
        integrity_verified=False,
    ))

    assert report.status is EvidenceQualityStatus.UNKNOWN
    assert report.confidence == 0.0
    assert report.tags >= {"EVIDENCE_INTEGRITY_UNVERIFIED", "SCHEMA_UNSUPPORTED"}


def test_quality_json_contract_is_reproducible(tmp_path: Path):
    path = tmp_path / "quality.json"
    path.write_text(
        '{"detector_id":"detector:v1","ood_score":0.2,"context_coverage":0.9,'
        '"schema_supported":true,"integrity_verified":true}',
        encoding="utf-8",
    )

    loaded = load_evidence_quality(path)

    assert loaded == EvidenceQualityInput("detector:v1", 0.2, 0.9, True, True)


def test_assessor_blocks_ood_and_accepts_in_distribution_evidence():
    root = Path(__file__).resolve().parents[1]
    shared = dict(
        root=root,
        intent="Add explicit OOD evidence quality.",
        tests_passed=True,
        efficiency_baseline=efficiency(),
        efficiency_candidate=efficiency(),
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
