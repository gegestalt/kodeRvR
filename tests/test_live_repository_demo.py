from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from code_provenance.repository_demo import (
    analyze_demo_repository,
    checkout_demo_repository,
    load_demo_repositories,
    select_demo_repository,
)


pytestmark = [pytest.mark.live_data, pytest.mark.external_fixture, pytest.mark.slow]


@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_random_github_demo_is_traceable(tmp_path: Path, seed: int):
    manifest = Path(__file__).resolve().parents[1] / "data/code_health/demo_repositories.json"
    fixtures = load_demo_repositories(manifest)
    fixture = select_demo_repository(fixtures, seed=seed)
    root = checkout_demo_repository(fixture, tmp_path / "cache")
    report = cast(dict[str, Any], analyze_demo_repository(root, fixture, max_file_vectors=1))

    assert report["fixture"]["fixture_id"] == fixture.fixture_id
    assert report["fixture"]["source_url"] == fixture.source_url
    assert report["snapshot"]["head_sha"] == fixture.head_revision
    assert report["safety"]["selection_scope"] == "curated_manifest_only"
    assert report["extraction"]["repository"]["feature_count"] > 0
    assert report["extraction"]["files"]["included_count"] <= 1


def test_pinned_github_repository_feeds_static_feature_pipeline(tmp_path: Path):
    manifest = Path(__file__).resolve().parents[1] / "data/code_health/demo_repositories.json"
    fixtures = load_demo_repositories(manifest)
    fixture = select_demo_repository(fixtures, fixture_id="modular")

    root = checkout_demo_repository(fixture, tmp_path / "cache")
    report = cast(dict[str, Any], analyze_demo_repository(root, fixture, max_file_vectors=5))

    assert report["fixture"]["fixture_id"] == "modular"
    assert report["snapshot"]["head_sha"] == fixture.head_revision
    assert report["snapshot"]["dirty"] is False
    assert report["safety"]["code_executed"] is False
    assert report["safety"]["analysis_mode"] == "static_read_only"
    assert report["summary"]["tracked_supported_files"] > 0
    assert report["summary"]["changed_files"] > 0
    assert report["summary"]["dependency_nodes"] > 0
    assert report["summary"]["dependency_edges"] >= report["summary"]["dependency_nodes"]


def test_every_curated_github_fixture_preserves_static_analysis_contract(tmp_path: Path):
    manifest = Path(__file__).resolve().parents[1] / "data/code_health/demo_repositories.json"
    fixtures = load_demo_repositories(manifest)
    changed_file_counts = []

    for fixture in fixtures:
        root = checkout_demo_repository(fixture, tmp_path / "cache")
        report = cast(dict[str, Any], analyze_demo_repository(root, fixture, max_file_vectors=2))
        summary = report["summary"]

        assert report["snapshot"]["head_sha"] == fixture.head_revision
        assert report["snapshot"]["dirty"] is False
        assert report["safety"]["code_executed"] is False
        assert summary["changed_files"] >= 0
        changed_file_counts.append(summary["changed_files"])
        assert summary["file_vectors_included"] <= 2
        assert summary["dependency_nodes"] >= 0
        assert summary["dependency_edges"] >= 0
        if summary["tracked_supported_files"]:
            assert summary["feature_count_per_file"] == 60

    assert any(count > 0 for count in changed_file_counts)