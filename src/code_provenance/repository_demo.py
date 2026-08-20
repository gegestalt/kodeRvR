"""Safe, static demonstrations over a curated set of pinned public repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Callable

import requests

from code_provenance.change_context import build_change_context
from code_provenance.dependency_context import build_dependency_context
from code_provenance.feature_space import (
    FEATURE_DEFINITIONS,
    extract_change_features,
    extract_repository_features,
    sample_feature_vector,
)
from code_provenance.repository import (
    build_provenance_observations,
    recent_commit_metadata,
    working_tree_samples,
)
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.symbol_index import build_changed_symbol_index


@dataclass(frozen=True)
class DemoRepository:
    fixture_id: str
    source_url: str
    base_revision: str
    head_revision: str
    license: str


def load_demo_repositories(path: Path) -> tuple[DemoRepository, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("repositories")
    if not isinstance(rows, list) or not rows:
        raise ValueError("demo manifest requires a non-empty repositories list")
    fixtures = tuple(DemoRepository(**row) for row in rows)
    seen: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            raise ValueError(f"duplicate demo fixture: {fixture.fixture_id}")
        seen.add(fixture.fixture_id)
        if not fixture.source_url.startswith("https://github.com/"):
            raise ValueError("demo repositories must be curated public GitHub URLs")
        if not all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (
            fixture.base_revision, fixture.head_revision
        )):
            raise ValueError("demo revisions must be immutable 40-character Git SHAs")
    return fixtures


def select_demo_repository(
    fixtures: tuple[DemoRepository, ...], *, seed: int | None = None, fixture_id: str | None = None
) -> DemoRepository:
    if fixture_id is not None:
        try:
            return next(item for item in fixtures if item.fixture_id == fixture_id)
        except StopIteration as error:
            raise ValueError(f"unknown curated demo repository: {fixture_id}") from error
    return random.Random(seed).choice(fixtures)


def discover_random_repository(
    *,
    seed: int | None = None,
    language: str | None = None,
    max_size_kb: int = 500_000,
    candidates: int = 10,
    request_get: Callable[..., Any] = requests.get,
) -> tuple[DemoRepository, dict[str, object]]:
    """Discover one bounded public GitHub repository and pin head to its parent."""
    if max_size_kb < 1 or candidates < 1:
        raise ValueError("repository size and candidate limits must be positive")
    language_filter = f"language:{language} " if language else ""
    query = f"{language_filter}size:<={max_size_kb}"
    response = request_get(
        "https://api.github.com/search/repositories",
        params={"q": query, "sort": "updated", "order": "desc", "per_page": candidates},
        headers={"Accept": "application/vnd.github+json"},
        timeout=30,
    )
    response.raise_for_status()
    items = [
        item for item in response.json().get("items", [])
        if not item.get("archived", False) and not item.get("fork", False)
        and int(item.get("size", max_size_kb + 1)) <= max_size_kb
    ]
    if not items:
        raise ValueError("GitHub returned no eligible repositories")
    selected = random.Random(seed).choice(items)
    full_name = str(selected.get("full_name", ""))
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", full_name):
        raise ValueError("GitHub returned an invalid repository name")
    commits_response = request_get(
        f"https://api.github.com/repos/{full_name}/commits",
        params={"per_page": 1},
        headers={"Accept": "application/vnd.github+json"},
        timeout=30,
    )
    commits_response.raise_for_status()
    commits = commits_response.json()
    if not isinstance(commits, list) or not commits or not commits[0].get("parents"):
        raise ValueError("GitHub repository has no commit parent suitable for analysis")
    head = str(commits[0].get("sha", ""))
    base = str(commits[0]["parents"][0].get("sha", ""))
    if not all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (base, head)):
        raise ValueError("GitHub returned invalid commit SHAs")
    license_data = selected.get("license") or {}
    license_name = license_data.get("spdx_id", "unknown") if isinstance(license_data, dict) else "unknown"
    return DemoRepository(
        full_name.replace("/", "--"), f"https://github.com/{full_name}.git",
        base, head, str(license_name),
    ), {
        "source": "github_api",
        "query": query,
        "language": language,
        "max_size_kb": max_size_kb,
        "candidate_limit": candidates,
        "seed": seed,
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()


def checkout_demo_repository(fixture: DemoRepository, cache_root: Path) -> Path:
    """Clone one curated fixture and verify its exact HEAD; never execute its code."""
    destination = cache_root.resolve() / fixture.fixture_id / fixture.head_revision
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", fixture.source_url, str(destination)],
            check=True, capture_output=True, text=True,
        )
        _git(destination, "checkout", "--detach", fixture.head_revision)
    if _git(destination, "config", "--get", "remote.origin.url") != fixture.source_url:
        raise ValueError("cached repository origin does not match the curated manifest")
    if _git(destination, "rev-parse", "HEAD") != fixture.head_revision:
        raise ValueError("cached repository HEAD does not match the curated immutable revision")
    if _git(destination, "status", "--porcelain"):
        raise ValueError("cached demo repository is dirty; refusing to alter or analyze it")
    _git(destination, "cat-file", "-e", f"{fixture.base_revision}^{{commit}}")
    return destination


def analyze_demo_repository(
    root: Path, fixture: DemoRepository, *, max_file_vectors: int = 20,
    selection_scope: str = "curated_manifest_only",
) -> dict[str, object]:
    if max_file_vectors < 1:
        raise ValueError("max_file_vectors must be positive")
    snapshot = capture_code_snapshot(root)
    if snapshot.head_sha != fixture.head_revision or snapshot.dirty:
        raise ValueError("analysis root is not the clean pinned demo revision")
    change = build_change_context(
        root, base_sha=fixture.base_revision, head_sha=fixture.head_revision
    )
    symbols = build_changed_symbol_index(root, change)
    dependencies = build_dependency_context(root, change, symbols)
    repository_features = extract_repository_features(root, change.target)
    change_features = extract_change_features(change)
    samples = sorted(working_tree_samples(root), key=lambda item: item.path or "")
    commits = recent_commit_metadata(root)
    file_vectors = [sample_feature_vector(sample) for sample in samples[:max_file_vectors]]
    feature_metadata = {
        name: {
            "scope": definition.scope.value,
            "family": definition.family.value,
            "producer": definition.producer,
            "languages": sorted(definition.languages),
            "missingness": definition.missingness,
            "normalization": definition.normalization,
            "leakage_risk": definition.leakage_risk,
            "reliability": definition.reliability.value,
            "scientific_role": definition.scientific_role,
        }
        for name, definition in FEATURE_DEFINITIONS.items()
    }
    return {
        "fixture": asdict(fixture),
        "safety": {
            "code_executed": False,
            "analysis_mode": "static_read_only",
            "selection_scope": selection_scope,
        },
        "snapshot": asdict(snapshot),
        "pipeline": [
            "pinned_checkout", "code_snapshot", "change_context", "changed_symbol_index",
            "dependency_context", "repository_features", "change_features", "file_features",
        ],
        "extraction": {
            "repository": {
                "source": "tracked files at pinned head revision",
                "feature_count": len(repository_features.values),
                "feature_names": sorted(repository_features.as_dict()),
            },
            "change": {
                "source": f"git diff {fixture.base_revision}..{fixture.head_revision}",
                "feature_count": len(change_features.values),
                "feature_names": sorted(change_features.as_dict()),
            },
            "symbols": {
                "source": "Python AST comparison across changed files",
                "language_scope": ("python",),
                "count": len(symbols.changes),
                "items": [item.qualified_name for item in symbols.changes],
                "partial": symbols.partial,
            },
            "dependencies": {
                "source": "static Python imports, calls, and decorators",
                "language_scope": ("python",),
                "node_count": len(dependencies.nodes),
                "edge_count": len(dependencies.edges),
                "unresolved_count": len(dependencies.unresolved),
                "impact_count": len(dependencies.impacts),
            },
            "files": {
                "source": "supported tracked files at pinned head revision",
                "available_count": len(samples),
                "included_count": len(file_vectors),
                "language_counts": dict(sorted(Counter(sample.language for sample in samples).items())),
                "feature_count_per_file": len(FEATURE_DEFINITIONS),
                "paths": [sample.path for sample in samples[:max_file_vectors]],
            },
        },
        "summary": {
            "tracked_supported_files": len(samples),
            "file_vectors_included": len(file_vectors),
            "feature_count_per_file": len(FEATURE_DEFINITIONS),
            "changed_files": len(change.changed_files),
            "changed_hunks": len(change.changed_hunks),
            "changed_symbols": len(symbols.changes),
            "dependency_nodes": len(dependencies.nodes),
            "dependency_edges": len(dependencies.edges),
            "unresolved_dependencies": len(dependencies.unresolved),
        },
        "provenance_observations": build_provenance_observations(commits),
        "calculations": {
            "repository_features": "aggregates over tracked files at the pinned head revision",
            "change_features": "aggregates over git diff(base_revision, head_revision)",
            "file_features": "60 lexical, AST, complexity, documentation, and change-compatible signals per supported file",
            "changed_symbols": "Python AST comparison across the pinned base/head revisions",
            "dependency_context": "conservative static Python import/call graph; unresolved edges remain explicit",
        },
        "repository_features": repository_features.as_dict(),
        "change_features": change_features.as_dict(),
        "changed_files": [asdict(item) for item in change.changed_files],
        "changed_symbols": [asdict(item) for item in symbols.changes],
        "dependency_impacts": [asdict(item) for item in dependencies.impacts],
        "unresolved_dependencies": [asdict(item) for item in dependencies.unresolved],
        "file_feature_vectors": [
            {"path": sample.path, "language": sample.language, "values": vector.as_dict()}
            for sample, vector in zip(samples[:max_file_vectors], file_vectors, strict=True)
        ],
        "feature_metadata": feature_metadata,
        "claim_boundary": (
            "The report describes static structure and the pinned change. It does not prove "
            "authorship, correctness, vulnerability, or semantic blast radius."
        ),
    }
