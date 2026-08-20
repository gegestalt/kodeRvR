from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest

from code_provenance.evidence import EvidenceTarget
from code_provenance.operational_data import (
    DecisionOutcome,
    DecisionExplanation,
    HumanOverride,
    ModelRun,
    OperationalRun,
    ReuseMatch,
    load_operational_run,
    write_operational_run,
)
from code_provenance.schema import ProvenanceEstimate


def target() -> EvidenceTarget:
    return EvidenceTarget("repo", "snapshot", "a" * 40)


def run() -> OperationalRun:
    return OperationalRun(
        run_id="run-1", target=target(), created_at=datetime(2026, 8, 20, tzinfo=UTC),
        snapshot={"tree_hash": "tree", "dirty": False},
        change_features={"change_churn": 12.0}, evidence_artifact_ids=("pytest:abc",),
        model_run=ModelRun("model-1", "1.0", "features-v2", "controlled-v1", "language_stratified"),
        estimate=ProvenanceEstimate(
            {"human": 0.2, "ai": 0.3, "hybrid": 0.5}, "hybrid", 0.5, 0.1,
            False, 0.2, 0.4, "statistical signal; not proof",
        ),
        reuse_matches=(ReuseMatch("token_shingle", "https://github.com/source", "b" * 40, "MIT", 0.3),),
        decision=DecisionOutcome.HUMAN_REVIEW,
        human_override=None,
        explanation=DecisionExplanation(
            top_features=(
                {"name": "change_churn", "value": 12.0, "direction": "raises_review_risk"},
            ),
            evidence_used=("pytest:abc",),
            public_reuse_considered=True,
            ood_status="in_distribution",
            abstention_reason=None,
            missing_evidence=("verified_ci",),
            reviewer_action="inspect_changed_symbols_and_ci",
        ),
    )


def test_operational_run_serializes_all_product_evidence(tmp_path):
    path = tmp_path / "run.json"
    write_operational_run(path, run())
    loaded = load_operational_run(path)

    assert loaded == run()
    payload = json.loads(path.read_text())
    assert payload["target"]["snapshot_id"] == "snapshot"
    assert payload["model_run"]["evaluation_protocol"] == "language_stratified"
    assert payload["reuse_matches"][0]["license"] == "MIT"
    assert payload["explanation"]["reviewer_action"] == "inspect_changed_symbols_and_ci"


def test_human_override_requires_reviewer_and_reason():
    with pytest.raises(ValueError, match="reviewer_id"):
        HumanOverride("", DecisionOutcome.STANDARD_REVIEW, "reason", datetime.now(UTC))
    with pytest.raises(ValueError, match="reason"):
        HumanOverride("reviewer", DecisionOutcome.STANDARD_REVIEW, "", datetime.now(UTC))


def test_operational_run_rejects_invalid_evidence_reference():
    with pytest.raises(ValueError, match="evidence artifact"):
        OperationalRun(**{**run().__dict__, "evidence_artifact_ids": ()})