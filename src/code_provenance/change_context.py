"""Typed, deterministic context for a working-tree or commit-range change."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from code_provenance.evidence import EvidenceTarget
from code_provenance.snapshot import capture_code_snapshot


@dataclass(frozen=True)
class ChangeIntent:
    text: str
    source: str

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.source.strip():
            raise ValueError("intent text and source are required")


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str
    additions: int
    deletions: int
    binary: bool = False


@dataclass(frozen=True)
class ChangedHunk:
    path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryContext:
    repository_id: str
    root: str
    branch: str | None
    dirty: bool


@dataclass(frozen=True)
class ChangeContext:
    target: EvidenceTarget
    base_sha: str | None
    head_sha: str
    changed_files: tuple[ChangedFile, ...]
    changed_hunks: tuple[ChangedHunk, ...]
    intent: ChangeIntent | None
    repository: RepositoryContext
    context_completeness: float
    missing_context: frozenset[str]


def _git(root: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and process.returncode:
        raise ValueError(f"invalid Git revision or repository state: {process.stderr.strip()}")
    return process.stdout


def _revision(root: Path, value: str) -> str:
    return _git(root, "rev-parse", "--verify", f"{value}^{{commit}}").strip()


def _status(value: str) -> str:
    return {
        "A": "added", "M": "modified", "D": "deleted", "R": "renamed",
        "C": "copied", "T": "type_changed", "U": "unmerged",
    }.get(value[:1], "unknown")


def _changed_files(root: Path, revisions: tuple[str, ...]) -> tuple[ChangedFile, ...]:
    names: dict[str, str] = {}
    for line in _git(root, "diff", "--name-status", *revisions, "--").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            names[fields[-1]] = _status(fields[0])
    counts: dict[str, tuple[int, int, bool]] = {}
    for line in _git(root, "diff", "--numstat", *revisions, "--").splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3:
            binary = fields[0] == "-" or fields[1] == "-"
            counts[fields[2]] = (
                0 if binary else int(fields[0]), 0 if binary else int(fields[1]), binary
            )
    if revisions == ("HEAD",):
        for path in _git(root, "ls-files", "--others", "--exclude-standard").splitlines():
            names[path] = "added"
            candidate = root / path
            if candidate.is_file() and not candidate.is_symlink():
                try:
                    additions = len(candidate.read_text(encoding="utf-8").splitlines())
                    counts[path] = (additions, 0, False)
                except UnicodeDecodeError:
                    counts[path] = (0, 0, True)
    return tuple(
        ChangedFile(path, names[path], *counts.get(path, (0, 0, False)))
        for path in sorted(names)
    )


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _changed_hunks(root: Path, revisions: tuple[str, ...]) -> tuple[ChangedHunk, ...]:
    output = _git(root, "diff", "--no-ext-diff", "--unified=0", *revisions, "--")
    path = ""
    current: dict[str, object] | None = None
    hunks: list[ChangedHunk] = []

    def finish() -> None:
        nonlocal current
        if current is not None:
            hunks.append(ChangedHunk(**current))
            current = None

    for line in output.splitlines():
        if line.startswith("+++ "):
            finish()
            raw = line[4:]
            path = raw[2:] if raw.startswith("b/") else raw
        elif match := _HUNK.match(line):
            finish()
            current = {
                "path": path,
                "old_start": int(match.group(1)), "old_lines": int(match.group(2) or 1),
                "new_start": int(match.group(3)), "new_lines": int(match.group(4) or 1),
                "header": match.group(5).strip(), "added_lines": (), "removed_lines": (),
            }
        elif current is not None and line.startswith("+"):
            current["added_lines"] = (*current["added_lines"], line[1:])
        elif current is not None and line.startswith("-"):
            current["removed_lines"] = (*current["removed_lines"], line[1:])
    finish()
    return tuple(sorted(hunks, key=lambda item: (item.path, item.new_start, item.old_start)))


def build_change_context(
    root: Path,
    *,
    base_sha: str | None = None,
    head_sha: str | None = None,
    intent: ChangeIntent | None = None,
) -> ChangeContext:
    """Build bounded context without executing repository code."""
    root = root.resolve()
    snapshot = capture_code_snapshot(root)
    head = _revision(root, head_sha or "HEAD")
    if head != snapshot.head_sha:
        raise ValueError("head revision must match the checked-out HEAD")
    base = _revision(root, base_sha) if base_sha else None
    revisions = (base, head) if base else ("HEAD",)
    missing = frozenset(
        name for name, present in (("base_sha", base is not None), ("intent", intent is not None))
        if not present
    )
    target = EvidenceTarget(snapshot.repository_id, snapshot.snapshot_id, snapshot.head_sha)
    branch = _git(root, "branch", "--show-current", check=False).strip() or None
    return ChangeContext(
        target=target,
        base_sha=base,
        head_sha=head,
        changed_files=_changed_files(root, revisions),
        changed_hunks=_changed_hunks(root, revisions),
        intent=intent,
        repository=RepositoryContext(snapshot.repository_id, str(root), branch, snapshot.dirty),
        context_completeness=(2 - len(missing)) / 2,
        missing_context=missing,
    )
