"""Dependency-aware patch health assessment and review routing.

The assessor is the product-facing module. Individual analyzers produce typed
evidence; this module preserves severe findings, propagates missing evidence,
and chooses an explainable review action. Provenance evidence never determines
patch safety by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from code_provenance.architecture import analyze_python_architecture
from code_provenance.efficiency import EfficiencyMeasurement, compare_efficiency
from code_provenance.evidence import EvidenceLedger
from code_provenance.evidence_quality import EvidenceQualityInput, evaluate_evidence_quality
from code_provenance.repository import working_tree_samples
from code_provenance.security import scan_code
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import TestEvidence


class TrustDimension(StrEnum):
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"
    EVIDENCE_INTEGRITY = "evidence_integrity"
    PROVENANCE = "provenance"
    INTENT_ALIGNMENT = "intent_alignment"
    FUNCTIONAL_EVIDENCE = "functional_evidence"
    ARCHITECTURAL_COMPATIBILITY = "architectural_compatibility"
    SECURITY_RISK = "security_risk"
    EFFICIENCY_RISK = "efficiency_risk"
    OOD_EVIDENCE_QUALITY = "ood_evidence_quality"


class EvidenceStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ReviewAction(StrEnum):
    ALLOW_STANDARD_REVIEW = "allow_standard_review"
    REQUEST_TARGETED_EVIDENCE = "request_targeted_evidence"
    REQUIRE_SECURITY_REVIEW = "require_security_review"
    REQUIRE_ARCHITECTURE_REVIEW = "require_architecture_review"
    REQUIRE_HUMAN_REWRITE_OR_VALIDATION = "require_human_rewrite_or_validation"
    BLOCK_PENDING_EVIDENCE = "block_pending_evidence"


@dataclass(frozen=True)
class EvidenceFinding:
    dimension: TrustDimension
    status: EvidenceStatus
    confidence: float
    severity: str
    summary: str
    evidence_refs: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("finding confidence must be in [0, 1]")
        if self.severity not in {"info", "low", "medium", "high", "critical"}:
            raise ValueError("unsupported finding severity")


@dataclass(frozen=True)
class AssessmentRequest:
    target_id: str
    target_kind: str
    findings: tuple[EvidenceFinding, ...]
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.target_kind not in {"pull_request", "commit", "working_tree"}:
            raise ValueError("target_kind must be pull_request, commit, or working_tree")
        dimensions = [finding.dimension for finding in self.findings]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("each trust dimension may have only one finding")


@dataclass(frozen=True)
class PatchHealthAssessment:
    target_id: str
    target_kind: str
    action: ReviewAction
    action_reason: str
    confidence: float
    findings: tuple[EvidenceFinding, ...]
    decision_path: tuple[TrustDimension, ...]
    missing_evidence: tuple[str, ...]
    tags: frozenset[str]

    def dimension(self, dimension: TrustDimension) -> EvidenceFinding:
        for finding in self.findings:
            if finding.dimension is dimension:
                return finding
        raise KeyError(f"no finding for {dimension.value}")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "action": self.action.value,
            "action_reason": self.action_reason,
            "confidence": self.confidence,
            "decision_path": [item.value for item in self.decision_path],
            "missing_evidence": list(self.missing_evidence),
            "tags": sorted(self.tags),
            "findings": [
                {
                    "dimension": item.dimension.value,
                    "status": item.status.value,
                    "confidence": item.confidence,
                    "severity": item.severity,
                    "summary": item.summary,
                    "evidence_refs": list(item.evidence_refs),
                    "tags": sorted(item.tags),
                }
                for item in self.findings
            ],
        }


_DEPENDENCY_ORDER = (
    TrustDimension.EVIDENCE_INTEGRITY,
    TrustDimension.EVIDENCE_SUFFICIENCY,
    TrustDimension.PROVENANCE,
    TrustDimension.INTENT_ALIGNMENT,
    TrustDimension.FUNCTIONAL_EVIDENCE,
    TrustDimension.ARCHITECTURAL_COMPATIBILITY,
    TrustDimension.SECURITY_RISK,
    TrustDimension.EFFICIENCY_RISK,
    TrustDimension.OOD_EVIDENCE_QUALITY,
)

_ACTION_RELEVANT = frozenset({
    TrustDimension.EVIDENCE_SUFFICIENCY,
    TrustDimension.EVIDENCE_INTEGRITY,
    TrustDimension.INTENT_ALIGNMENT,
    TrustDimension.FUNCTIONAL_EVIDENCE,
    TrustDimension.ARCHITECTURAL_COMPATIBILITY,
    TrustDimension.SECURITY_RISK,
    TrustDimension.EFFICIENCY_RISK,
    TrustDimension.OOD_EVIDENCE_QUALITY,
})


class PatchHealthAssessor:
    """Evaluate dependent patch evidence through one deterministic interface."""

    def assess(self, request: AssessmentRequest) -> PatchHealthAssessment:
        by_dimension = {finding.dimension: finding for finding in request.findings}
        ordered = tuple(by_dimension[item] for item in _DEPENDENCY_ORDER if item in by_dimension)
        tags = set(request.tags)
        for finding in ordered:
            tags.update(finding.tags)

        missing = tuple(
            dimension.value
            for dimension in _DEPENDENCY_ORDER
            if dimension in _ACTION_RELEVANT
            and (
                dimension not in by_dimension
                or by_dimension[dimension].status is EvidenceStatus.UNKNOWN
                or (
                    dimension is TrustDimension.FUNCTIONAL_EVIDENCE
                    and "TEST_RESULT_UNVERIFIED" in by_dimension[dimension].tags
                )
            )
        )
        if TrustDimension.FUNCTIONAL_EVIDENCE.value in missing:
            tags.add("TEST_EVIDENCE_MISSING")

        action, reason, path = self._route(by_dimension, missing, tags)
        relevant_confidence = [
            item.confidence
            for item in ordered
            if item.dimension in _ACTION_RELEVANT
        ]
        confidence = min(relevant_confidence, default=0.0)
        if missing:
            confidence = 0.0
        if "OOD_INPUT" in tags or "REPOSITORY_CONTEXT_PARTIAL" in tags:
            confidence = min(confidence, 0.25)

        return PatchHealthAssessment(
            target_id=request.target_id,
            target_kind=request.target_kind,
            action=action,
            action_reason=reason,
            confidence=float(confidence),
            findings=ordered,
            decision_path=path,
            missing_evidence=missing,
            tags=frozenset(tags),
        )

    def assess_repository(
        self,
        root: Path,
        *,
        intent: str | None = None,
        tests_passed: bool | None = None,
        efficiency_baseline: EfficiencyMeasurement | None = None,
        efficiency_candidate: EfficiencyMeasurement | None = None,
        evidence_quality: EvidenceQualityInput | None = None,
        evidence_ledger: EvidenceLedger | None = None,
        test_evidence: TestEvidence | None = None,
    ) -> PatchHealthAssessment:
        """Assess a working tree without executing repository code or inferring authorship."""
        root = root.resolve()
        snapshot = capture_code_snapshot(root)
        samples = working_tree_samples(root)
        security_hits = [
            (sample.path, signal)
            for sample in samples
            for signal in scan_code(sample.code)
        ]
        severe = [item for item in security_hits if item[1].severity in {"high", "critical"}]
        production_severe = [
            item for item in severe
            if not item[0].startswith(("tests/", "test/"))
        ]
        security_status = (
            EvidenceStatus.FAIL if production_severe
            else EvidenceStatus.WARN if severe
            else EvidenceStatus.PASS
        )
        security_severity = (
            "critical" if any(item[1].severity == "critical" for item in production_severe)
            else "high" if production_severe
            else "low" if severe
            else "info"
        )
        security_refs = tuple(
            f"{path}:{signal.line}:{signal.rule_id}" for path, signal in security_hits
        ) or ("static-scan:no-findings",)
        architecture = analyze_python_architecture(root)
        architecture_status = EvidenceStatus(architecture.status.value)
        architecture_refs = tuple(
            f"{signal.path}:{signal.rule_id}" for signal in architecture.signals
        ) or (
            f"architecture:{architecture.modules_analyzed}-modules:"
            f"{architecture.dependency_edges}-edges:no-cycles",
        )
        efficiency = (
            compare_efficiency(efficiency_baseline, efficiency_candidate)
            if efficiency_baseline is not None and efficiency_candidate is not None
            else None
        )
        efficiency_status = (
            EvidenceStatus(efficiency.status.value)
            if efficiency is not None else EvidenceStatus.UNKNOWN
        )
        efficiency_refs = (
            tuple(
                f"efficiency:{name}:{delta:+.4f}"
                for name, delta in sorted(efficiency.deltas.items())
            ) or ("efficiency:no-comparable-metrics",)
            if efficiency is not None else ("efficiency:missing",)
        )
        quality = evaluate_evidence_quality(evidence_quality) if evidence_quality is not None else None
        quality_status = EvidenceStatus(quality.status.value) if quality is not None else EvidenceStatus.UNKNOWN
        integrity = evidence_ledger.audit_integrity() if evidence_ledger is not None else None
        ledger_snapshot_mismatch = (
            evidence_ledger is not None
            and evidence_ledger.target_commit not in {snapshot.snapshot_id, snapshot.head_sha}
        )
        integrity_status = EvidenceStatus(integrity.status.value) if integrity is not None else EvidenceStatus.UNKNOWN
        if ledger_snapshot_mismatch:
            integrity_status = EvidenceStatus.FAIL

        intent_status = EvidenceStatus.PASS if intent and intent.strip() else EvidenceStatus.UNKNOWN
        test_snapshot_matches = (
            test_evidence is not None
            and test_evidence.snapshot_id == snapshot.snapshot_id
            and test_evidence.target_sha == snapshot.head_sha
        )
        if test_evidence is not None and not test_snapshot_matches:
            test_status, test_confidence = EvidenceStatus.FAIL, 1.0
            test_summary = "Observed test artifact does not match the assessed code snapshot."
            test_refs = (f"tests:snapshot-mismatch:{test_evidence.snapshot_id}",)
            test_tags = frozenset({"TEST_SNAPSHOT_MISMATCH"})
        elif test_evidence is not None and not test_evidence.complete:
            test_status, test_confidence = EvidenceStatus.UNKNOWN, 0.0
            test_summary = "Observed pytest execution was incomplete."
            test_refs = (f"tests:{test_evidence.output_hash}:incomplete",)
            test_tags = frozenset({"TEST_RUN_INCOMPLETE"})
        elif test_evidence is not None and test_evidence.tests_collected == 0:
            test_status, test_confidence = EvidenceStatus.UNKNOWN, 0.0
            test_summary = "Observed pytest execution collected no tests."
            test_refs = (f"tests:{test_evidence.output_hash}:empty",)
            test_tags = frozenset({"TEST_SUITE_EMPTY"})
        elif test_evidence is not None and test_evidence.exit_code == 0 and test_evidence.failed == 0 and test_evidence.errors == 0:
            test_status, test_confidence = EvidenceStatus.PASS, 0.9
            test_summary = (
                f"Observed pytest {test_evidence.framework_version}: {test_evidence.passed} passed, "
                f"{test_evidence.failed} failed in {test_evidence.duration_seconds:.3f}s."
            )
            test_refs = (f"tests:{test_evidence.output_hash}:observed",)
            test_tags = frozenset({"TEST_EVIDENCE_OBSERVED"})
        elif test_evidence is not None:
            test_status, test_confidence = EvidenceStatus.FAIL, 0.9
            test_summary = (
                f"Observed pytest failed: {test_evidence.failed} failed, "
                f"{test_evidence.errors} errors, exit {test_evidence.exit_code}."
            )
            test_refs = (f"tests:{test_evidence.output_hash}:failed",)
            test_tags = frozenset({"TEST_EVIDENCE_OBSERVED"})
        elif tests_passed is not None:
            test_status, test_confidence = EvidenceStatus.WARN, 0.35
            test_summary = (
                "Test success was asserted by the caller without a snapshot-bound artifact."
                if tests_passed else
                "Test failure was asserted by the caller without a snapshot-bound artifact."
            )
            test_refs = ("tests:caller-asserted",)
            test_tags = frozenset({"TEST_RESULT_UNVERIFIED"})
        else:
            test_status, test_confidence = EvidenceStatus.UNKNOWN, 0.0
            test_summary = "No test execution evidence was supplied."
            test_refs = ("tests:missing",)
            test_tags = frozenset({"TEST_EVIDENCE_MISSING"})
        findings = (
            EvidenceFinding(
                TrustDimension.EVIDENCE_SUFFICIENCY,
                EvidenceStatus.WARN,
                0.5,
                "medium",
                "Working-tree evidence lacks authoritative PR and specification context.",
                (f"repository:{root}",),
                frozenset({"REPOSITORY_CONTEXT_PARTIAL"}),
            ),
            EvidenceFinding(
                TrustDimension.EVIDENCE_INTEGRITY,
                integrity_status,
                integrity.confidence if integrity is not None else 0.0,
                "critical" if integrity_status is EvidenceStatus.FAIL
                else "medium" if integrity_status is EvidenceStatus.UNKNOWN
                else "info",
                (
                    "Evidence ledger targets a different code snapshot."
                    if ledger_snapshot_mismatch
                    else f"Ledger integrity audit found {len(integrity.failed_artifacts)} failed artifact(s)."
                    if integrity is not None
                    else "No commit-bound evidence ledger was supplied."
                ),
                (
                    (f"ledger:snapshot-mismatch:{evidence_ledger.target_commit}",)
                    if ledger_snapshot_mismatch
                    else tuple(f"artifact:{item}" for item in integrity.failed_artifacts)
                    or (f"ledger:{evidence_ledger.target_commit}:verified",)
                    if integrity is not None
                    else ("ledger:missing",)
                ),
                integrity.tags if integrity is not None else frozenset({"EVIDENCE_INTEGRITY_UNKNOWN"}),
            ),
            EvidenceFinding(
                TrustDimension.PROVENANCE,
                EvidenceStatus.UNKNOWN,
                0.0,
                "info",
                "No verified provenance model or attestation was supplied.",
                ("provenance:unavailable",),
                frozenset({"PROVENANCE_UNCERTAIN"}),
            ),
            EvidenceFinding(
                TrustDimension.INTENT_ALIGNMENT,
                intent_status,
                0.8 if intent_status is EvidenceStatus.PASS else 0.0,
                "info" if intent_status is EvidenceStatus.PASS else "medium",
                "Intent text is available for alignment review." if intent_status is EvidenceStatus.PASS
                else "No issue, specification, or PR intent was supplied.",
                ("intent:user-supplied",) if intent_status is EvidenceStatus.PASS else ("intent:missing",),
                frozenset() if intent_status is EvidenceStatus.PASS else frozenset({"INTENT_UNVERIFIED"}),
            ),
            EvidenceFinding(
                TrustDimension.FUNCTIONAL_EVIDENCE,
                test_status,
                test_confidence,
                "info" if test_status is EvidenceStatus.PASS
                else "high" if test_status is EvidenceStatus.FAIL
                else "medium",
                test_summary,
                test_refs,
                test_tags,
            ),
            EvidenceFinding(
                TrustDimension.ARCHITECTURAL_COMPATIBILITY,
                architecture_status,
                architecture.confidence,
                "high" if architecture_status is EvidenceStatus.FAIL
                else "medium" if architecture_status is EvidenceStatus.UNKNOWN
                else "info",
                f"Dependency analysis inspected {architecture.modules_analyzed} module(s), "
                f"{architecture.dependency_edges} edge(s), and found "
                f"{len(architecture.cycles)} cycle(s).",
                architecture_refs,
            ),
            EvidenceFinding(
                TrustDimension.SECURITY_RISK,
                security_status,
                0.7,
                security_severity,
                f"Static scan found {len(production_severe)} production and "
                f"{len(severe) - len(production_severe)} test-only severe signal(s).",
                security_refs,
            ),
            EvidenceFinding(
                TrustDimension.EFFICIENCY_RISK,
                efficiency_status,
                efficiency.confidence if efficiency is not None else 0.0,
                "high" if efficiency_status is EvidenceStatus.FAIL
                else "medium" if efficiency_status is EvidenceStatus.UNKNOWN
                else "info",
                (
                    f"Measured efficiency comparison exceeded: {', '.join(efficiency.exceeded)}."
                    if efficiency is not None and efficiency.exceeded
                    else "Measured efficiency comparison remained within configured budgets."
                    if efficiency is not None and efficiency_status is EvidenceStatus.PASS
                    else "Efficiency measurements were missing or statistically insufficient."
                ),
                efficiency_refs,
            ),
            EvidenceFinding(
                TrustDimension.OOD_EVIDENCE_QUALITY,
                quality_status,
                quality.confidence if quality is not None else 0.0,
                "high" if quality_status is EvidenceStatus.FAIL
                else "medium" if quality_status is EvidenceStatus.UNKNOWN
                else "info",
                (
                    f"Detector {quality.detector_id} reported OOD={quality.ood_score:.3f} "
                    f"with context coverage={quality.context_coverage:.3f}."
                    if quality is not None
                    else "No named OOD and evidence-integrity artifact was supplied."
                ),
                (
                    (f"ood:{quality.detector_id}:{quality.ood_score:.4f}",)
                    if quality is not None else ("ood:missing",)
                ),
                quality.tags if quality is not None else frozenset(),
            ),
        )
        return self.assess(AssessmentRequest(
            target_id=f"working-tree:{root.name}",
            target_kind="working_tree",
            findings=findings,
            tags=frozenset({"REPOSITORY_CONTEXT_PARTIAL"}),
        ))

    @staticmethod
    def _route(
        findings: dict[TrustDimension, EvidenceFinding],
        missing: tuple[str, ...],
        tags: set[str],
    ) -> tuple[ReviewAction, str, tuple[TrustDimension, ...]]:
        integrity = findings.get(TrustDimension.EVIDENCE_INTEGRITY)
        if integrity and integrity.status is EvidenceStatus.FAIL:
            return (
                ReviewAction.BLOCK_PENDING_EVIDENCE,
                "Evidence integrity failure prevents trustworthy automation.",
                (TrustDimension.EVIDENCE_INTEGRITY, TrustDimension.EVIDENCE_SUFFICIENCY),
            )

        ood = findings.get(TrustDimension.OOD_EVIDENCE_QUALITY)
        if "OOD_INPUT" in tags or (ood and ood.status is EvidenceStatus.FAIL):
            return (
                ReviewAction.BLOCK_PENDING_EVIDENCE,
                "Out-of-distribution or materially incomplete evidence prevents confident automation.",
                (TrustDimension.EVIDENCE_SUFFICIENCY, TrustDimension.OOD_EVIDENCE_QUALITY),
            )

        security = findings.get(TrustDimension.SECURITY_RISK)
        if security and security.status is EvidenceStatus.FAIL and security.severity in {"high", "critical"}:
            return (
                ReviewAction.REQUIRE_SECURITY_REVIEW,
                "High-severity security evidence requires specialist review.",
                (TrustDimension.EVIDENCE_SUFFICIENCY, TrustDimension.SECURITY_RISK),
            )

        architecture = findings.get(TrustDimension.ARCHITECTURAL_COMPATIBILITY)
        if architecture and architecture.status is EvidenceStatus.FAIL:
            return (
                ReviewAction.REQUIRE_ARCHITECTURE_REVIEW,
                "Architecture evidence indicates an incompatible change.",
                (TrustDimension.EVIDENCE_SUFFICIENCY, TrustDimension.ARCHITECTURAL_COMPATIBILITY),
            )

        intent = findings.get(TrustDimension.INTENT_ALIGNMENT)
        functional = findings.get(TrustDimension.FUNCTIONAL_EVIDENCE)
        efficiency = findings.get(TrustDimension.EFFICIENCY_RISK)
        if any(item and item.status is EvidenceStatus.FAIL for item in (intent, functional, efficiency)):
            failed = tuple(
                item.dimension for item in (intent, functional, efficiency)
                if item and item.status is EvidenceStatus.FAIL
            )
            return (
                ReviewAction.REQUIRE_HUMAN_REWRITE_OR_VALIDATION,
                "Failed intent, functional, or efficiency evidence requires correction or validation.",
                (TrustDimension.EVIDENCE_SUFFICIENCY, *failed),
            )

        if missing:
            path = tuple(
                TrustDimension(name) for name in missing
            )
            return (
                ReviewAction.REQUEST_TARGETED_EVIDENCE,
                "Required review evidence is missing; request only the unresolved evidence.",
                (TrustDimension.EVIDENCE_SUFFICIENCY, *path),
            )

        return (
            ReviewAction.ALLOW_STANDARD_REVIEW,
            "Available intent, functional, architecture, security, and efficiency evidence supports standard review.",
            tuple(item for item in _DEPENDENCY_ORDER if item in findings and item is not TrustDimension.PROVENANCE),
        )
