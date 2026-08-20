"""Labelled corpus loading with mandatory provenance and group controls."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pandas as pd

from code_provenance.schema import AuthorshipLabel, CodeSample, DatasetRole, EvidenceSource


REQUIRED = {
    "sample_id", "repository_id", "group_id", "language", "path", "label",
    "label_source", "generator_family", "dataset_id", "dataset_version", "author_group_id",
    "dataset_role", "provenance_source", "source_url", "source_revision", "content_hash",
    "license", "acquisition_date",
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
        role = DatasetRole(str(row["dataset_role"]))
        code = str(row.get("code", ""))
        if "code" not in frame:
            candidate = (code_root / str(row["path"])).resolve()
            if code_root.resolve() not in candidate.parents:
                raise ValueError("manifest path escapes code_root")
            code = candidate.read_text(encoding="utf-8", errors="replace")
        if label != AuthorshipLabel.UNKNOWN and source in {EvidenceSource.HEURISTIC, EvidenceSource.UNLABELLED}:
            raise ValueError("training labels may not be heuristic or unlabelled")
        if role in {DatasetRole.OOD, DatasetRole.STRUCTURAL_ONLY}:
            if label is not AuthorshipLabel.UNKNOWN or source is not EvidenceSource.UNLABELLED:
                raise ValueError("OOD and structural_only records must remain unlabelled")
        elif label is AuthorshipLabel.UNKNOWN:
            raise ValueError("labelled dataset roles require an authorship label")
        if not str(row["author_group_id"]).strip():
            raise ValueError("author_group_id is required")
        if not str(row["source_url"]).startswith("https://"):
            raise ValueError("source_url must be HTTPS")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row["source_revision"])):
            raise ValueError("source_revision must be an immutable Git SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["content_hash"])):
            raise ValueError("content_hash must be a SHA-256 digest")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != str(row["content_hash"]):
            raise ValueError("content_hash does not match code")
        for field in ("dataset_id", "dataset_version", "provenance_source", "license", "acquisition_date"):
            if not str(row[field]).strip():
                raise ValueError(f"{field} is required")
        samples.append(CodeSample(
            sample_id=str(row["sample_id"]), repository_id=str(row["repository_id"]),
            group_id=str(row["group_id"]), language=str(row["language"]), code=code,
            path=str(row["path"]), label=label, label_source=source,
            generator_family=str(row["generator_family"]), dataset_id=str(row["dataset_id"]),
            dataset_version=str(row["dataset_version"]), author_group_id=str(row["author_group_id"]),
            dataset_role=role, provenance_source=str(row["provenance_source"]),
            source_url=str(row["source_url"]), source_revision=str(row["source_revision"]),
            content_hash=str(row["content_hash"]), license=str(row["license"]),
            acquisition_date=str(row["acquisition_date"]),
        ))
    return samples
