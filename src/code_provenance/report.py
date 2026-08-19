"""Repository-level descriptive report with no unsupported authorship claim."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from code_provenance.features import extract_features
from code_provenance.repository import recent_commit_metadata, working_tree_samples


def descriptive_repository_report(root: Path) -> dict[str, object]:
    samples = working_tree_samples(root)
    commits = recent_commit_metadata(root)
    features = [extract_features(sample) for sample in samples]
    return {
        "repository": str(root.resolve()),
        "files_analyzed": len(samples),
        "commits_analyzed": len(commits),
        "languages": dict(Counter(sample.language for sample in samples)),
        "total_nonblank_lines": int(sum(item["nonblank_lines"] for item in features)),
        "median_file_lines": float(np.median([item["lines"] for item in features])) if features else 0.0,
        "median_commit_additions": float(np.median([item["additions"] for item in commits])) if commits else 0.0,
        "median_commit_files": float(np.median([item["files_changed"] for item in commits])) if commits else 0.0,
        "authorship_estimate": "UNAVAILABLE_UNTIL_A_LABELLED_GROUP_DISJOINT_MODEL_IS_FITTED",
        "claim_boundary": "Git metadata, style, or reuse alone cannot prove human or AI authorship",
    }
