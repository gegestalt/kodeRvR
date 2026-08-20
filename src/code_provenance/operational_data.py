"""Durable snapshot-bound records for the provenance review product."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import json
from pathlib import Path

from code_provenance.evidence import EvidenceTarget
from code_provenance.schema import ProvenanceEstimate


class DecisionOutcome(StrEnum):
    STANDARD_REVIEW = "standard_review"
    TARGETED_EVIDENCE = "targeted_evidence"
    HUMAN_REVIEW = "human_review"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    model_version: str
    feature_schema_version: str
    training_dataset_id: str
    evaluation_protocol: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class ReuseMatch:
    match_kind: str
    source_url: str
    source_revision: str
    license: str
    overlap_fraction: float

    def __post_init__(self) -> None:
        if not self.source_url.startswith("https://"):
            raise ValueError("reuse source_url must be HTTPS")
        if not self.source_revision.strip() or not self.license.strip():
            raise ValueError("reuse source revision and license are required")
        if not 0 <= self.overlap_fraction <= 1:
            raise ValueError("reuse overlap_fraction must be in [0, 1]")


@dataclass(frozen=True)
class HumanOverride:
    reviewer_id: str
    decision: DecisionOutcome
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if not self.reason.strip():
            raise ValueError("override reason is required")
        if self.created_at.tzinfo is None:
            raise ValueError("override timestamp must be timezone-aware")


@dataclass(frozen=True)
class DecisionExplanation:
    top_features: tuple[dict[str, object], ...]
    evidence_used: tuple[str, ...]
    public_reuse_considered: bool
    ood_status: str
    abstention_reason: str | None
    missing_evidence: tuple[str, ...]
    reviewer_action: str

    def __post_init__(self) -> None:
        if not self.evidence_used:
            raise ValueError("explanations require evidence references")
        if self.ood_status not in {"in_distribution", "ood", "unknown"}:
            raise ValueError("unsupported OOD status")
        if not self.reviewer_action.strip():
            raise ValueError("reviewer_action is required")
        for feature in self.top_features:
            if not str(feature.get("name", "")).strip() or "value" not in feature:
                raise ValueError("top features require name and value")


@dataclass(frozen=True)
class OperationalRun:
    run_id: str
    target: EvidenceTarget
    created_at: datetime
    snapshot: dict[str, object]
    change_features: dict[str, float]
    evidence_artifact_ids: tuple[str, ...]
    model_run: ModelRun
    estimate: ProvenanceEstimate | None
    reuse_matches: tuple[ReuseMatch, ...]
    decision: DecisionOutcome
    human_override: HumanOverride | None
    explanation: DecisionExplanation | None = None
    observed_label: str = "unknown"
    label_source: str = "unlabelled"

    schema_version = "1.0"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.created_at.tzinfo is None:
            raise ValueError("run timestamp must be timezone-aware")
        if not self.evidence_artifact_ids:
            raise ValueError("run requires at least one evidence artifact")
        if any(not item.strip() for item in self.evidence_artifact_ids):
            raise ValueError("evidence artifact IDs must be non-empty")
        if self.human_override is not None and self.human_override.decision != self.decision:
            raise ValueError("human override decision must match run decision")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "target": {
                "repository_id": self.target.repository_id,
                "snapshot_id": self.target.snapshot_id,
                "head_sha": self.target.head_sha,
            },
            "created_at": self.created_at.isoformat(),
            "snapshot": self.snapshot,
            "change_features": self.change_features,
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "model_run": self.model_run.__dict__,
            "estimate": None if self.estimate is None else {
                **self.estimate.__dict__,
            },
            "reuse_matches": [item.__dict__ for item in self.reuse_matches],
            "decision": self.decision.value,
            "human_override": None if self.human_override is None else {
                **self.human_override.__dict__,
                "decision": self.human_override.decision.value,
                "created_at": self.human_override.created_at.isoformat(),
            },
            "explanation": None if self.explanation is None else self.explanation.__dict__,
            "observed_label": self.observed_label,
            "label_source": self.label_source,
        }


def write_operational_run(path: Path, run: OperationalRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_operational_run(path: Path) -> OperationalRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != OperationalRun.schema_version:
        raise ValueError("unsupported operational run schema version")
    target = EvidenceTarget(**payload["target"])
    estimate_payload = payload.get("estimate")
    estimate = ProvenanceEstimate(**estimate_payload) if estimate_payload is not None else None
    override_payload = payload.get("human_override")
    override = None if override_payload is None else HumanOverride(
        reviewer_id=override_payload["reviewer_id"],
        decision=DecisionOutcome(override_payload["decision"]),
        reason=override_payload["reason"],
        created_at=datetime.fromisoformat(override_payload["created_at"]),
    )
    explanation_payload = payload.get("explanation")
    explanation = None if explanation_payload is None else DecisionExplanation(
        top_features=tuple(explanation_payload["top_features"]),
        evidence_used=tuple(explanation_payload["evidence_used"]),
        public_reuse_considered=bool(explanation_payload["public_reuse_considered"]),
        ood_status=explanation_payload["ood_status"],
        abstention_reason=explanation_payload["abstention_reason"],
        missing_evidence=tuple(explanation_payload["missing_evidence"]),
        reviewer_action=explanation_payload["reviewer_action"],
    )
    return OperationalRun(
        run_id=payload["run_id"], target=target,
        created_at=datetime.fromisoformat(payload["created_at"]),
        snapshot=payload["snapshot"], change_features=payload["change_features"],
        evidence_artifact_ids=tuple(payload["evidence_artifact_ids"]),
        model_run=ModelRun(**payload["model_run"]), estimate=estimate,
        reuse_matches=tuple(ReuseMatch(**item) for item in payload["reuse_matches"]),
        decision=DecisionOutcome(payload["decision"]), human_override=override,
        explanation=explanation,
        observed_label=payload.get("observed_label", "unknown"),
        label_source=payload.get("label_source", "unlabelled"),
    )


def render_review_comment(run: OperationalRun) -> str:
    """Render a deterministic, evidence-linked review summary for PRs or commits."""
    estimate = run.estimate
    if estimate is None:
        prediction = "Unavailable"
        confidence = "Unavailable"
        uncertainty = "No model estimate supplied"
    else:
        prediction = estimate.predicted_label
        confidence = f"{estimate.confidence:.0%}"
        uncertainty = f"OOD: {run.explanation.ood_status if run.explanation else 'unknown'}; " \
            f"abstained: {'yes' if estimate.abstained else 'no'}"
    lines = [
        "## Provenance review signal",
        "",
        f"- **Estimate:** `{prediction}`",
        f"- **Confidence:** `{confidence}`",
        f"- **Uncertainty:** {uncertainty}",
        f"- **Decision:** `{run.decision.value}`",
        f"- **Evidence:** {', '.join(f'`{item}`' for item in run.evidence_artifact_ids)}",
        f"- **Public reuse considered:** `{'yes' if run.reuse_matches else 'no'}`",
    ]
    if run.explanation is not None:
        lines.extend([
            "",
            "### Review guidance",
            *[f"- `{item['name']}`: {item.get('direction', 'supporting signal')}" for item in run.explanation.top_features],
            f"- **Next action:** `{run.explanation.reviewer_action}`",
            f"- **Missing evidence:** {', '.join(run.explanation.missing_evidence) or 'none recorded'}",
        ])
    if run.human_override is not None:
        lines.extend(["", f"**Human override:** `{run.human_override.decision.value}` by `{run.human_override.reviewer_id}`."])
    lines.extend(["", "> These signals are not authorship proof and should not replace human review."])
    return "\n".join(lines) + "\n"