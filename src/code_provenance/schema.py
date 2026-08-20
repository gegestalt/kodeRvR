"""Typed evidence objects and conservative provenance terminology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AuthorshipLabel(StrEnum):
    HUMAN = "human"
    AI = "ai"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class EvidenceSource(StrEnum):
    DECLARED = "declared"
    CONTROLLED_GENERATION = "controlled_generation"
    VERIFIED_AGENT_ACCOUNT = "verified_agent_account"
    HUMAN_REVIEWED = "human_reviewed"
    HEURISTIC = "heuristic"
    UNLABELLED = "unlabelled"


class DatasetRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"
    STRUCTURAL_ONLY = "structural_only"


@dataclass(frozen=True)
class CodeSample:
    sample_id: str
    repository_id: str
    group_id: str
    language: str
    code: str
    path: str = ""
    commit_sha: str = ""
    commit_message: str = ""
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0
    label: AuthorshipLabel = AuthorshipLabel.UNKNOWN
    label_source: EvidenceSource = EvidenceSource.UNLABELLED
    generator_family: str = "unknown"
    dataset_id: str = "unknown"
    dataset_version: str = "unknown"
    author_group_id: str = "unknown"
    dataset_role: DatasetRole = DatasetRole.STRUCTURAL_ONLY
    provenance_source: str = ""
    source_url: str = ""
    source_revision: str = ""
    content_hash: str = ""
    license: str = "unknown"
    acquisition_date: str = ""
    parent_authorship: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvenanceEstimate:
    probabilities: dict[str, float]
    predicted_label: str
    confidence: float
    ood_score: float
    abstained: bool
    public_reuse_fraction: float
    organic_fraction: float | None
    claim: str

    def __post_init__(self) -> None:
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("authorship probabilities must sum to one")
        for value in (*self.probabilities.values(), self.confidence, self.ood_score,
                      self.public_reuse_fraction):
            if not 0 <= value <= 1:
                raise ValueError("probabilities and fractions must be in [0, 1]")
        if self.organic_fraction is not None and not 0 <= self.organic_fraction <= 1:
            raise ValueError("organic_fraction must be in [0, 1]")
