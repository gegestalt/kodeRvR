"""Read-only Git repository extraction for working-tree and commit evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess

from code_provenance.schema import CodeSample


LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".rs": "rust",
    ".go": "go", ".rb": "ruby", ".php": "php", ".swift": "swift",
}

_EXPLICIT_AI_MARKERS = re.compile(
    r"(?:co-authored-by:.*(?:copilot|chatgpt|openai|claude|gemini)|"
    r"(?:generated|written|assisted)\s+by\s+(?:copilot|chatgpt|openai|claude|gemini)|"
    r"(?:copilot|chatgpt|openai|claude|gemini)\s+(?:generated|assisted))",
    re.IGNORECASE,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout


def repository_id(root: Path) -> str:
    try:
        remote = _git(root, "config", "--get", "remote.origin.url").strip()
    except subprocess.CalledProcessError:
        remote = str(root.resolve())
    return hashlib.sha256(remote.encode()).hexdigest()[:16]


def working_tree_samples(root: Path, *, max_bytes: int = 250_000) -> list[CodeSample]:
    """Extract supported tracked files without executing repository code."""
    root = root.resolve()
    # Include non-ignored untracked files so a PR/working-tree scan sees the
    # proposed change before it is committed; ignored corpora and secrets stay out.
    paths = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    repo = repository_id(root)
    samples = []
    for relative in paths:
        path = root / relative
        language = LANGUAGES.get(path.suffix.lower())
        if not relative or language is None or not path.is_file() or path.stat().st_size > max_bytes:
            continue
        code = path.read_text(encoding="utf-8", errors="replace")
        samples.append(CodeSample(
            sample_id=hashlib.sha256(f"{repo}:{relative}".encode()).hexdigest()[:20],
            repository_id=repo, group_id=repo, language=language, code=code, path=relative,
        ))
    return samples


def recent_commit_metadata(root: Path, *, limit: int = 500) -> list[dict[str, object]]:
    """Extract stable commit/numstat metadata; no authorship label is inferred."""
    if limit < 1:
        raise ValueError("limit must be positive")
    marker = "--PROVENANCE-COMMIT--"
    output = _git(root.resolve(), "log", f"-{limit}", "--numstat", f"--format={marker}%n%H%x1f%ct%x1f%B")
    commits: list[dict[str, object]] = []
    for block in output.split(marker)[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split("\x1f", 2)
        if len(header) != 3:
            continue
        additions = deletions = files = 0
        message_lines = []
        for line in lines[1:]:
            pieces = line.split("\t")
            if len(pieces) == 3 and pieces[0].isdigit() and pieces[1].isdigit():
                additions += int(pieces[0]); deletions += int(pieces[1]); files += 1
            elif line.strip():
                message_lines.append(line.strip())
        commits.append({
            "commit_sha": header[0], "timestamp": int(header[1]),
            "commit_message": "\n".join(message_lines) or header[2].strip(),
            "files_changed": files, "additions": additions, "deletions": deletions,
        })
    return commits


def build_provenance_observations(commits: list[dict[str, object]]) -> dict[str, object]:
    """Summarize explicit provenance clues without converting them to labels."""
    observed = [
        {
            "commit_sha": commit["commit_sha"],
            "commit_message": commit["commit_message"],
            "explicit_ai_marker": bool(_EXPLICIT_AI_MARKERS.search(str(commit["commit_message"]))),
        }
        for commit in commits
    ]
    return {
        "commit_count": len(observed),
        "explicit_ai_marker_count": sum(item["explicit_ai_marker"] for item in observed),
        "commits": observed,
        "claim": "observations are not authorship ground truth",
    }
