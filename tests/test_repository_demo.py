from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from code_provenance.repository_demo import (
    DemoRepository,
    analyze_demo_repository,
    load_demo_repositories,
    select_demo_repository,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def _repository(root: Path) -> tuple[DemoRepository, Path]:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Demo")
    _git(root, "config", "user.email", "demo@example.test")
    (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "app.py").write_text("def value():\n    return 2\n\ndef added(flag: bool):\n    return flag\n", encoding="utf-8")
    (root / "tests.py").write_text("from app import value\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8")
    _git(root, "add", "app.py", "tests.py")
    _git(root, "commit", "-qm", "change value")
    head = _git(root, "rev-parse", "HEAD")
    return DemoRepository("local", "https://github.com/example/local.git", base, head, "test"), root


def test_curated_manifest_requires_github_urls_and_immutable_revisions(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"repositories": [{
        "fixture_id": "bad", "source_url": "https://example.test/repo.git",
        "base_revision": "main", "head_revision": "latest", "license": "unknown",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="GitHub"):
        load_demo_repositories(path)


def test_seeded_random_repository_selection_is_deterministic():
    fixtures = tuple(
        DemoRepository(str(index), f"https://github.com/o/r{index}.git", "a" * 40, "b" * 40, "x")
        for index in range(3)
    )
    assert select_demo_repository(fixtures, seed=42) == select_demo_repository(fixtures, seed=42)


def test_unknown_repository_id_is_rejected():
    fixtures = (DemoRepository("known", "https://github.com/o/r.git", "a" * 40, "b" * 40, "x"),)
    with pytest.raises(ValueError, match="unknown curated"):
        select_demo_repository(fixtures, fixture_id="arbitrary-user-input")


def test_demo_report_explains_feature_calculations_without_executing_code(tmp_path: Path):
    fixture, root = _repository(tmp_path)
    report = analyze_demo_repository(root, fixture, max_file_vectors=1)
    assert report["safety"] == {
        "code_executed": False,
        "analysis_mode": "static_read_only",
        "selection_scope": "curated_manifest_only",
    }
    assert report["summary"]["tracked_supported_files"] == 2
    assert report["summary"]["file_vectors_included"] == 1
    assert report["summary"]["feature_count_per_file"] == 60
    assert report["summary"]["changed_files"] == 2
    assert report["summary"]["changed_symbols"] >= 2
    assert len(report["file_feature_vectors"][0]["values"]) == 60
    assert report["calculations"]["change_features"].startswith("aggregates over git diff")
    assert "authorship" in report["claim_boundary"]


def test_demo_report_is_deterministic_for_same_pinned_checkout(tmp_path: Path):
    fixture, root = _repository(tmp_path)
    first = analyze_demo_repository(root, fixture, max_file_vectors=2)
    second = analyze_demo_repository(root, fixture, max_file_vectors=2)
    assert first == second
