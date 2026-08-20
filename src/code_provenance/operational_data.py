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
    return OperationalRun(
        run_id=payload["run_id"], target=target,
        created_at=datetime.fromisoformat(payload["created_at"]),
        snapshot=payload["snapshot"], change_features=payload["change_features"],
        evidence_artifact_ids=tuple(payload["evidence_artifact_ids"]),
        model_run=ModelRun(**payload["model_run"]), estimate=estimate,
        reuse_matches=tuple(ReuseMatch(**item) for item in payload["reuse_matches"]),
        decision=DecisionOutcome(payload["decision"]), human_override=override,
        observed_label=payload.get("observed_label", "unknown"),
        label_source=payload.get("label_source", "unlabelled"),
    )