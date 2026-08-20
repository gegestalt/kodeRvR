"""Labelled corpus loading with mandatory provenance and group controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
import re

import pandas as pd

from code_provenance.schema import AuthorshipLabel, CodeSample, DatasetRole, EvidenceSource
from code_provenance.reuse import token_shingles


REQUIRED = {
    "sample_id", "repository_id", "group_id", "language", "path", "label",
    "label_source", "generator_family", "dataset_id", "dataset_version", "author_group_id",
    "dataset_role", "provenance_source", "source_url", "source_revision", "content_hash",
    "license", "acquisition_date",
}


@dataclass(frozen=True)
class SplitPlan:
    train: tuple[CodeSample, ...]
    validation: tuple[CodeSample, ...]
    test: tuple[CodeSample, ...]
    audit: dict[str, object]


def _duplicate_clusters(samples: list[CodeSample], width: int) -> list[tuple[CodeSample, ...]]:
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    shingles = [token_shingles(item.code, width) for item in samples]
    for left, right in combinations(range(len(samples)), 2):
        union_size = len(shingles[left] | shingles[right])
        similarity = len(shingles[left] & shingles[right]) / union_size if union_size else 1.0
        if similarity >= 0.95:
            union(left, right)
    clusters: dict[int, list[CodeSample]] = {}
    for index, sample in enumerate(samples):
        clusters.setdefault(find(index), []).append(sample)
    return [tuple(sorted(items, key=lambda item: item.sample_id)) for items in clusters.values()]


def build_split_plan(
    samples: list[CodeSample], *, seed: int = 42, duplicate_width: int = 7
) -> SplitPlan:
    """Build deterministic train/validation/test partitions without known leakage."""
    if duplicate_width < 2:
        raise ValueError("duplicate_width must be at least two")
    if any(item.dataset_role in {DatasetRole.OOD, DatasetRole.STRUCTURAL_ONLY} for item in samples):
        raise ValueError("OOD and structural_only records cannot enter training partitions")
    if any(item.label is AuthorshipLabel.UNKNOWN for item in samples):
        raise ValueError("split planning requires labelled samples")
    languages = sorted({item.language for item in samples})
    if len(languages) < 3:
        raise ValueError("language holdout requires three language groups")
    clusters = _duplicate_clusters(samples, duplicate_width)
    dimensions = ("repository_id", "author_group_id", "dataset_id", "generator_family")
    used: list[set[str]] = [set(), set(), set()]
    partitions: list[list[CodeSample]] = [[], [], []]
    language_owners: dict[str, int] = {}
    for cluster in sorted(clusters, key=lambda items: (len(items), items[0].sample_id), reverse=True):
        cluster_languages = {item.language for item in cluster}
        cluster_keys = {
            f"{dimension}:{getattr(item, dimension)}"
            for item in cluster for dimension in dimensions
            if getattr(item, dimension) != "unknown"
        }
        candidates = []
        for partition_index in range(3):
            if any(language_owners.get(language, partition_index) != partition_index for language in cluster_languages):
                continue
            if used[partition_index] & cluster_keys:
                continue
            candidates.append(partition_index)
        if not candidates:
            raise ValueError("repository, author, dataset, generator, or language groups conflict")
        target = min(candidates, key=lambda index: (len(partitions[index]), index))
        partitions[target].extend(cluster)
        used[target].update(cluster_keys)
        for language in cluster_languages:
            language_owners[language] = target
    return SplitPlan(
        tuple(sorted(partitions[0], key=lambda item: item.sample_id)),
        tuple(sorted(partitions[1], key=lambda item: item.sample_id)),
        tuple(sorted(partitions[2], key=lambda item: item.sample_id)),
        {
            "duplicate_cluster_count": len(clusters),
            "duplicate_width": duplicate_width,
            "language_owners": {language: index for language, index in sorted(language_owners.items())},
            "disjoint_dimensions": (*dimensions, "language", "near_duplicate_cluster"),
            "seed": seed,
        },
    )


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
