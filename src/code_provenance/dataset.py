"""Labelled corpus loading with mandatory provenance and group controls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from code_provenance.schema import AuthorshipLabel, CodeSample, EvidenceSource


REQUIRED = {
    "sample_id", "repository_id", "group_id", "language", "path", "label",
    "label_source", "generator_family",
}


def load_manifest(path: Path, *, code_root: Path | None = None) -> list[CodeSample]:
    """Load CSV metadata; code may be inline or referenced by a relative path."""
    frame = pd.read_csv(path)
    if missing := REQUIRED - set(frame):
        raise ValueError(f"missing corpus manifest columns: {sorted(missing)}")
    if "code" not in frame and code_root is None:
        raise ValueError("manifest needs an inline code column or code_root")
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError("sample_id must be unique")
    samples = []
    for row in frame.to_dict("records"):
        label = AuthorshipLabel(str(row["label"]))
        source = EvidenceSource(str(row["label_source"]))
        if label != AuthorshipLabel.UNKNOWN and source in {EvidenceSource.HEURISTIC, EvidenceSource.UNLABELLED}:
            raise ValueError("training labels may not be heuristic or unlabelled")
        code = str(row.get("code", ""))
        if "code" not in frame:
            candidate = (code_root / str(row["path"])).resolve()
            if code_root.resolve() not in candidate.parents:
                raise ValueError("manifest path escapes code_root")
            code = candidate.read_text(encoding="utf-8", errors="replace")
        samples.append(CodeSample(
            sample_id=str(row["sample_id"]), repository_id=str(row["repository_id"]),
            group_id=str(row["group_id"]), language=str(row["language"]), code=code,
            path=str(row["path"]), label=label, label_source=source,
            generator_family=str(row["generator_family"]),
        ))
    return samples
