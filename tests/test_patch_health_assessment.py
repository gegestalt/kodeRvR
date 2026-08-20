from __future__ import annotations

from pathlib import Path

from code_provenance.assessment import (
    AssessmentRequest,
    EvidenceFinding,
    EvidenceStatus,
    PatchHealthAssessor,
    ReviewAction,
    TrustDimension,
)
from code_provenance.report import descriptive_repository_report


def finding(
    dimension: TrustDimension,
    status: EvidenceStatus,
    *,
    confidence: float = 0.9,
    severity: str = "info",
) -> EvidenceFinding:
    return EvidenceFinding(
        dimension=dimension,
        status=status,
        confidence=confidence,
        severity=severity,
        summary=f"{dimension.value}: {status.value}",
        evidence_refs=("fixture:1",),
    )


def healthy_request() -> AssessmentRequest:
    return AssessmentRequest(
        target_id="pr:17",
        target_kind="pull_request",
        findings=(
            finding(TrustDimension.EVIDENCE_SUFFICIENCY, EvidenceStatus.PASS),
            finding(TrustDimension.EVIDENCE_INTEGRITY, EvidenceStatus.PASS),
            finding(TrustDimension.INTENT_ALIGNMENT, EvidenceStatus.PASS),
            finding(TrustDimension.FUNCTIONAL_EVIDENCE, EvidenceStatus.PASS),
            finding(TrustDimension.ARCHITECTURAL_COMPATIBILITY, EvidenceStatus.PASS),
            finding(TrustDimension.SECURITY_RISK, EvidenceStatus.PASS),
            finding(TrustDimension.EFFICIENCY_RISK, EvidenceStatus.PASS),
            finding(TrustDimension.OOD_EVIDENCE_QUALITY, EvidenceStatus.PASS),
            finding(TrustDimension.PROVENANCE, EvidenceStatus.UNKNOWN, confidence=0.2),
        ),
    )


def test_healthy_patch_reaches_standard_review_without_provenance_claim():
    result = PatchHealthAssessor().assess(healthy_request())

    assert result.action is ReviewAction.ALLOW_STANDARD_REVIEW
    assert result.target_id == "pr:17"
    assert result.dimension(TrustDimension.PROVENANCE).status is EvidenceStatus.UNKNOWN
    assert "provenance" not in result.action_reason.lower()


def test_critical_security_finding_propagates_to_review_action():
    request = healthy_request()
    findings = tuple(
        finding(
            item.dimension,
            EvidenceStatus.FAIL,
            severity="critical",
        )
        if item.dimension is TrustDimension.SECURITY_RISK
        else item
        for item in request.findings
    )

    result = PatchHealthAssessor().assess(
        AssessmentRequest(request.target_id, request.target_kind, findings)
    )

    assert result.action is ReviewAction.REQUIRE_SECURITY_REVIEW
    assert TrustDimension.SECURITY_RISK in result.decision_path


def test_missing_tests_request_evidence_instead_of_accusing_authorship():
    request = healthy_request()
    findings = tuple(
        finding(item.dimension, EvidenceStatus.UNKNOWN, confidence=0.0)
        if item.dimension is TrustDimension.FUNCTIONAL_EVIDENCE
        else item
        for item in request.findings
    )

    result = PatchHealthAssessor().assess(
        AssessmentRequest(request.target_id, request.target_kind, findings)
    )

    assert result.action is ReviewAction.REQUEST_TARGETED_EVIDENCE
    assert "functional_evidence" in result.missing_evidence
    assert result.tags >= {"TEST_EVIDENCE_MISSING"}


def test_ood_or_incomplete_context_blocks_confident_automation():
    request = AssessmentRequest(
        target_id="commit:abc",
        target_kind="commit",
        findings=(
            finding(TrustDimension.EVIDENCE_SUFFICIENCY, EvidenceStatus.UNKNOWN, confidence=0.0),
            finding(TrustDimension.OOD_EVIDENCE_QUALITY, EvidenceStatus.FAIL, severity="high"),
        ),
        tags=frozenset({"OOD_INPUT", "REPOSITORY_CONTEXT_PARTIAL"}),
    )

    result = PatchHealthAssessor().assess(request)

    assert result.action is ReviewAction.BLOCK_PENDING_EVIDENCE
    assert result.confidence == 0.0
    assert result.tags >= {"OOD_INPUT", "REPOSITORY_CONTEXT_PARTIAL"}


def test_repository_assessment_uses_one_patch_level_interface():
    root = Path(__file__).resolve().parents[1]

    result = PatchHealthAssessor().assess_repository(root)

    assert result.target_kind == "working_tree"
    assert result.action is ReviewAction.REQUEST_TARGETED_EVIDENCE
    assert result.dimension(TrustDimension.SECURITY_RISK).evidence_refs
    assert result.tags >= {"INTENT_UNVERIFIED", "TEST_EVIDENCE_MISSING"}


def test_assessment_is_deterministic_for_identical_evidence():
    assessor = PatchHealthAssessor()
    request = healthy_request()

    assert assessor.assess(request).to_dict() == assessor.assess(request).to_dict()


def test_omitted_required_dimension_cannot_produce_standard_review():
    request = healthy_request()
    incomplete = AssessmentRequest(
        request.target_id,
        request.target_kind,
        tuple(
            item for item in request.findings
            if item.dimension is not TrustDimension.SECURITY_RISK
        ),
    )

    result = PatchHealthAssessor().assess(incomplete)

    assert result.action is ReviewAction.REQUEST_TARGETED_EVIDENCE
    assert "security_risk" in result.missing_evidence


def test_repository_report_exposes_patch_health_decision():
    root = Path(__file__).resolve().parents[1]

    report = descriptive_repository_report(
        root,
        intent="Add a dependency-aware patch health assessment.",
        tests_passed=True,
    )

    assert report["patch_health"]["target_kind"] == "working_tree"
    assert "action" in report["patch_health"]
    assert "findings" in report["patch_health"]
