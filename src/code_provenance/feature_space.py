"""Typed, multi-scope feature metadata and aggregate extractors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
import subprocess

from code_provenance.change_context import ChangeContext
from code_provenance.evidence import EvidenceTarget
from code_provenance.features import FEATURE_NAMES, extract_features
from code_provenance.schema import CodeSample


class FeatureScope(StrEnum):
    REPOSITORY = "repository"
    CHANGE = "change"
    FILE = "file"
    SYMBOL = "symbol"
    HUNK = "hunk"


class FeatureFamily(StrEnum):
    SIZE = "size_composition"
    COMPLEXITY = "complexity"
    AST = "ast_structure"
    IDENTIFIER = "identifier_documentation"
    DEPENDENCY = "dependency_graph"
    CHANGE = "change_structure"
    TESTING = "testing"
    DUPLICATION = "duplication_reuse"
    SECURITY = "security"
    REPOSITORY = "repository_behavior"
    PROVENANCE = "provenance_support"
    QUALITY = "data_quality"


class FeatureReliability(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    scope: FeatureScope
    family: FeatureFamily
    producer: str
    languages: frozenset[str]
    missingness: str
    normalization: str
    leakage_risk: str
    reliability: FeatureReliability
    scientific_role: str


@dataclass(frozen=True)
class FeatureValue:
    definition: FeatureDefinition
    value: float
    target_id: str


@dataclass(frozen=True)
class FeatureVector:
    target_id: str
    scope: FeatureScope
    values: tuple[FeatureValue, ...]

    def as_dict(self) -> dict[str, float]:
        return {item.definition.name: item.value for item in self.values}


@dataclass(frozen=True)
class FeatureExtractionReport:
    target: EvidenceTarget
    vectors: tuple[FeatureVector, ...]
    unsupported_features: tuple[str, ...]
    warnings: tuple[str, ...]


_COMPLEXITY = {"max_indent", "mean_indent", "mean_function_lines", "function_length_std", "max_function_lines", "max_ast_depth", "cyclomatic_complexity", "branch_density"}
_IDENTIFIER = {name for name in FEATURE_NAMES if "identifier" in name or "docstring" in name or "comment" in name}
_CHANGE = {"message_length", "message_line_count", "files_changed", "additions", "deletions", "change_balance", "commit_word_count", "commit_issue_reference_count", "churn", "change_size_log", "deletion_fraction"}
_AST = {"function_count", "class_count", "branch_count", "exception_count", "import_count", "async_function_count", "decorator_count", "typed_argument_fraction", "return_annotation_fraction", "lambda_count", "comprehension_count", "assert_count", "with_count", "await_count", "yield_count", "call_count", "assignment_count"}


def _family(name: str) -> FeatureFamily:
    if name in _CHANGE:
        return FeatureFamily.CHANGE
    if name in _COMPLEXITY:
        return FeatureFamily.COMPLEXITY
    if name in _IDENTIFIER:
        return FeatureFamily.IDENTIFIER
    if name in _AST:
        return FeatureFamily.AST
    return FeatureFamily.SIZE


FEATURE_DEFINITIONS = {
    name: FeatureDefinition(
        name=name,
        scope=FeatureScope.CHANGE if name in _CHANGE else FeatureScope.FILE,
        family=_family(name),
        producer="code_provenance.features:v2",
        languages=frozenset({"*"}) if name not in _AST | _COMPLEXITY else frozenset({"python"}),
        missingness="zero_when_not_observed",
        normalization="fit_on_training_partition",
        leakage_risk="medium" if name in _CHANGE else "low",
        reliability=FeatureReliability.HIGH if name in _CHANGE else FeatureReliability.MEDIUM,
        scientific_role="supporting statistical signal; never authorship proof",
    )
    for name in FEATURE_NAMES
}


def sample_feature_vector(sample: CodeSample) -> FeatureVector:
    values = extract_features(sample)
    return FeatureVector(
        sample.sample_id,
        FeatureScope.FILE,
        tuple(FeatureValue(FEATURE_DEFINITIONS[name], values[name], sample.sample_id) for name in FEATURE_NAMES),
    )


def _vector(target_id: str, scope: FeatureScope, family: FeatureFamily, values: dict[str, float]) -> FeatureVector:
    definitions = [
        FeatureDefinition(
            name, scope, family, f"code_provenance.feature_space:{scope.value}",
            frozenset({"*"}), "explicit_zero", "per_repository_or_change",
            "low", FeatureReliability.HIGH, "descriptive context; not authorship evidence",
        )
        for name in sorted(values)
    ]
    return FeatureVector(target_id, scope, tuple(
        FeatureValue(definition, float(values[definition.name]), target_id)
        for definition in definitions
    ))


def extract_repository_features(root: Path, target: EvidenceTarget) -> FeatureVector:
    root = root.resolve()
    paths = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.split(b"\0")
    files = [item.decode("utf-8", errors="surrogateescape") for item in paths if item]
    extensions = Counter((Path(path).suffix.lower() or "<none>") for path in files)
    total = max(len(files), 1)
    probabilities = [count / total for count in extensions.values()]
    language_entropy = -sum(value * math.log2(value) for value in probabilities)
    test_files = sum(
        Path(path).name.startswith("test_") or "tests" in Path(path).parts for path in files
    )
    line_counts = []
    for relative in files:
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            try:
                line_counts.append(len(candidate.read_text(encoding="utf-8").splitlines()))
            except UnicodeDecodeError:
                pass
    values = {
        "repository_files": len(files),
        "repository_text_lines": sum(line_counts),
        "repository_mean_file_lines": sum(line_counts) / max(len(line_counts), 1),
        "repository_max_file_lines": max(line_counts, default=0),
        "repository_test_files": test_files,
        "repository_test_file_ratio": test_files / total,
        "repository_extension_count": len(extensions),
        "repository_language_entropy": language_entropy,
    }
    return _vector(target.snapshot_id, FeatureScope.REPOSITORY, FeatureFamily.REPOSITORY, values)


def extract_change_features(context: ChangeContext) -> FeatureVector:
    additions = sum(item.additions for item in context.changed_files)
    deletions = sum(item.deletions for item in context.changed_files)
    paths = [Path(item.path) for item in context.changed_files]
    subsystems = {path.parts[0] if len(path.parts) > 1 else "." for path in paths}
    values = {
        "change_files": len(paths),
        "change_hunks": len(context.changed_hunks),
        "change_additions": additions,
        "change_deletions": deletions,
        "change_churn": additions + deletions,
        "change_subsystems": len(subsystems),
        "change_test_file_ratio": sum(path.name.startswith("test_") or "tests" in path.parts for path in paths) / max(len(paths), 1),
        "change_binary_files": sum(item.binary for item in context.changed_files),
    }
    return _vector(context.target.snapshot_id, FeatureScope.CHANGE, FeatureFamily.CHANGE, values)
