"""Repository-level descriptive report with no unsupported authorship claim."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from code_provenance.assessment import PatchHealthAssessor
from code_provenance.efficiency import EfficiencyMeasurement
from code_provenance.evidence import EvidenceLedger
from code_provenance.evidence_quality import EvidenceQualityInput
from code_provenance.features import extract_features
from code_provenance.repository import recent_commit_metadata, working_tree_samples


def descriptive_repository_report(
    root: Path,
    *,
    intent: str | None = None,
    tests_passed: bool | None = None,
    efficiency_baseline: EfficiencyMeasurement | None = None,
    efficiency_candidate: EfficiencyMeasurement | None = None,
    evidence_quality: EvidenceQualityInput | None = None,
    evidence_ledger: EvidenceLedger | None = None,
) -> dict[str, object]:
    samples = working_tree_samples(root)
    commits = recent_commit_metadata(root)
    features = [extract_features(sample) for sample in samples]
    health = PatchHealthAssessor().assess_repository(
        root,
        intent=intent,
        tests_passed=tests_passed,
        efficiency_baseline=efficiency_baseline,
        efficiency_candidate=efficiency_candidate,
        evidence_quality=evidence_quality,
        evidence_ledger=evidence_ledger,
    )
    return {
        "repository": str(root.resolve()),
        "files_analyzed": len(samples),
        "commits_analyzed": len(commits),
        "languages": dict(Counter(sample.language for sample in samples)),
        "total_nonblank_lines": int(sum(item["nonblank_lines"] for item in features)),
        "median_file_lines": float(np.median([item["lines"] for item in features])) if features else 0.0,
        "median_commit_additions": float(np.median([item["additions"] for item in commits])) if commits else 0.0,
        "median_commit_files": float(np.median([item["files_changed"] for item in commits])) if commits else 0.0,
        "patch_health": health.to_dict(),
        "evidence_ledger": (
            evidence_ledger.to_dict()
            if evidence_ledger is not None
            else {"status": "UNAVAILABLE", "reason": "no commit-bound evidence ledger supplied"}
        ),
        "authorship_estimate": "UNAVAILABLE_UNTIL_A_LABELLED_GROUP_DISJOINT_MODEL_IS_FITTED",
        "claim_boundary": "Git metadata, style, or reuse alone cannot prove human or AI authorship",
    }
