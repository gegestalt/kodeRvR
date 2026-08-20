"""Deterministic local CODEOWNERS parsing and changed-path ownership facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import shlex

from code_provenance.change_context import ChangeContext
from code_provenance.dependency_context import DependencyContext
from code_provenance.evidence import EvidenceTarget
from code_provenance.symbol_index import ChangedSymbolIndex
from code_provenance.test_relevance import TestRelevanceContext


class OwnerKind(StrEnum):
    USER = "user"
    TEAM = "team"
    EMAIL = "email"


class OwnershipIssueKind(StrEnum):
    MALFORMED_RULE = "malformed_rule"
    UNSUPPORTED_PATTERN = "unsupported_pattern"
    INVALID_OWNER = "invalid_owner"
    READ_FAILURE = "read_failure"


@dataclass(frozen=True)
class OwnerRef:
    identifier: str
    kind: OwnerKind


@dataclass(frozen=True)
class OwnershipRule:
    pattern: str
    owners: tuple[OwnerRef, ...]
    source_path: str
    line: int


@dataclass(frozen=True)
class OwnershipIssue:
    kind: OwnershipIssueKind
    source_path: str
    line: int | None
    detail: str


@dataclass(frozen=True)
class PathOwnership:
    path: str
    owners: tuple[OwnerRef, ...]
    matched_rule: OwnershipRule | None


@dataclass(frozen=True)
class SymbolOwnership:
    symbol_id: str
    path: str
    owners: tuple[OwnerRef, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class OwnershipContext:
    target: EvidenceTarget
    source_path: str | None
    rules: tuple[OwnershipRule, ...]
    paths: tuple[PathOwnership, ...]
    symbols: tuple[SymbolOwnership, ...]
    declared_owners: tuple[OwnerRef, ...]
    unowned_paths: tuple[str, ...]
    complete: bool
    issues: tuple[OwnershipIssue, ...]


_LOCATIONS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _owner(value: str) -> OwnerRef | None:
    if value.startswith("@") and len(value) > 1:
        return OwnerRef(value, OwnerKind.TEAM if "/" in value else OwnerKind.USER)
    if _EMAIL.fullmatch(value):
        return OwnerRef(value, OwnerKind.EMAIL)
    return None


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    if pattern.startswith("!") or "[" in pattern or "]" in pattern:
        raise ValueError("negation and character ranges are unsupported by CODEOWNERS")
    rooted = pattern.startswith("/")
    value = pattern[1:] if rooted else pattern
    directory = value.endswith("/")
    if directory:
        value += "**"
    has_slash = "/" in value.rstrip("/")
    output = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "*":
            if index + 1 < len(value) and value[index + 1] == "*":
                index += 2
                if index < len(value) and value[index] == "/":
                    output.append("(?:.*/)?")
                    index += 1
                else:
                    output.append(".*")
                continue
            output.append("[^/]*")
        elif character == "?":
            output.append("[^/]")
        else:
            output.append(re.escape(character))
        index += 1
    prefix = "^" if rooted or has_slash else r"^(?:.*/)?"
    return re.compile(prefix + "".join(output) + "$" )


def _parse(path: Path, source_path: str) -> tuple[tuple[OwnershipRule, ...], tuple[OwnershipIssue, ...]]:
    rules = []
    issues = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return (), (OwnershipIssue(OwnershipIssueKind.READ_FAILURE, source_path, None, str(error)),)
    for number, raw in enumerate(lines, 1):
        try:
            fields = shlex.split(raw, comments=True, posix=True)
        except ValueError as error:
            issues.append(OwnershipIssue(OwnershipIssueKind.MALFORMED_RULE, source_path, number, str(error)))
            continue
        if not fields:
            continue
        if len(fields) < 2:
            issues.append(OwnershipIssue(OwnershipIssueKind.MALFORMED_RULE, source_path, number, "rule has no owners"))
            continue
        pattern = fields[0]
        try:
            _pattern_regex(pattern)
        except ValueError as error:
            issues.append(OwnershipIssue(OwnershipIssueKind.UNSUPPORTED_PATTERN, source_path, number, str(error)))
            continue
        owners = []
        for value in fields[1:]:
            parsed = _owner(value)
            if parsed is None:
                issues.append(OwnershipIssue(OwnershipIssueKind.INVALID_OWNER, source_path, number, value))
            elif parsed not in owners:
                owners.append(parsed)
        if not owners:
            issues.append(OwnershipIssue(OwnershipIssueKind.MALFORMED_RULE, source_path, number, "rule has no valid owners"))
            continue
        rules.append(OwnershipRule(pattern, tuple(owners), source_path, number))
    return tuple(rules), tuple(issues)


def build_ownership_context(
    root: Path,
    change: ChangeContext,
    symbols: ChangedSymbolIndex,
    dependencies: DependencyContext,
    relevance: TestRelevanceContext,
) -> OwnershipContext:
    """Resolve local path declarations; dependency edges never propagate owners."""
    if not (change.target == symbols.target == dependencies.target == relevance.target):
        raise ValueError("all ownership inputs must share one target")
    root = root.resolve()
    source_path = next((item for item in _LOCATIONS if (root / item).is_file()), None)
    rules: tuple[OwnershipRule, ...] = ()
    issues: tuple[OwnershipIssue, ...] = ()
    if source_path is not None:
        rules, issues = _parse(root / source_path, source_path)
    compiled = [(rule, _pattern_regex(rule.pattern)) for rule in rules]
    paths = []
    for changed in sorted(change.changed_files, key=lambda item: item.path):
        matched = None
        for rule, matcher in compiled:
            if matcher.fullmatch(changed.path):
                matched = rule
        paths.append(PathOwnership(changed.path, matched.owners if matched else (), matched))
    path_map = {item.path: item for item in paths}
    symbol_rows = []
    for changed in symbols.changes:
        symbol = changed.after or changed.before
        if symbol is None or symbol.path not in path_map:
            continue
        ownership = path_map[symbol.path]
        evidence = (
            (f"{ownership.matched_rule.source_path}:{ownership.matched_rule.line}",)
            if ownership.matched_rule else ()
        )
        symbol_rows.append(SymbolOwnership(symbol.symbol_id, symbol.path, ownership.owners, evidence))
    declared = tuple(sorted(
        {owner for rule in rules for owner in rule.owners},
        key=lambda item: (item.identifier, item.kind.value),
    ))
    return OwnershipContext(
        change.target,
        source_path,
        rules,
        tuple(paths),
        tuple(sorted(symbol_rows, key=lambda item: (item.path, item.symbol_id))),
        declared,
        tuple(item.path for item in paths if not item.owners),
        not issues,
        tuple(sorted(issues, key=lambda item: (item.line or 0, item.kind.value, item.detail))),
    )
