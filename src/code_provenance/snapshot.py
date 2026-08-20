"""Deterministic identity for clean and dirty Git repository states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

from code_provenance.repository import repository_id


@dataclass(frozen=True)
class CodeSnapshot:
    repository_id: str
    head_sha: str
    tree_hash: str
    diff_hash: str | None
    dirty: bool
    snapshot_id: str


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def capture_code_snapshot(root: Path) -> CodeSnapshot:
    """Hash HEAD, binary diff, and untracked contents without executing code."""
    root = root.resolve()
    repo = repository_id(root)
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    status = _git(root, "status", "--porcelain=v1", "-z")
    dirty = bool(status)
    diff_hash: str | None = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        for raw in sorted(item for item in untracked.split(b"\0") if item):
            relative = raw.decode("utf-8", errors="surrogateescape")
            path = root / relative
            digest.update(b"\0UNTRACKED\0")
            digest.update(raw)
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        diff_hash = digest.hexdigest()
    identity = "\x1f".join((repo, head, tree, diff_hash or "clean"))
    snapshot_id = "snap_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    return CodeSnapshot(repo, head, tree, diff_hash, dirty, snapshot_id)
