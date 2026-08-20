from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from code_provenance.change_context import ChangeIntent, build_change_context
from code_provenance.feature_space import (
    FEATURE_DEFINITIONS,
    FeatureFamily,
    FeatureScope,
    extract_change_features,
    extract_repository_features,
    sample_feature_vector,
)
from code_provenance.features import FEATURE_NAMES
from code_provenance.schema import AuthorshipLabel
from test_code_provenance import sample


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_every_model_feature_has_complete_nonleaking_metadata():
    assert len(FEATURE_NAMES) >= 50
    assert set(FEATURE_DEFINITIONS) == set(FEATURE_NAMES)
    for definition in FEATURE_DEFINITIONS.values():
        assert definition.scope in FeatureScope
        assert definition.family in FeatureFamily
        assert definition.producer
        assert definition.scientific_role
        assert "repository_id" not in definition.name
        assert "author" not in definition.name


def test_sample_vector_preserves_target_metadata_and_finite_values():
    vector = sample_feature_vector(sample(1, AuthorshipLabel.HUMAN))
    assert vector.target_id == "s1-human"
    assert tuple(item.definition.name for item in vector.values) == FEATURE_NAMES
    assert np.isfinite([item.value for item in vector.values]).all()


def test_repository_and_change_extractors_have_distinct_scopes(tmp_path: Path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / "app.py").write_text("def run(x):\n    return x + 1\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    git(tmp_path, "add", "app.py", "test_app.py")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "app.py").write_text("def run(x):\n    if x:\n        return x + 2\n", encoding="utf-8")
    context = build_change_context(tmp_path, intent=ChangeIntent("Handle truthy values", "test"))

    repository = extract_repository_features(tmp_path, context.target)
    change = extract_change_features(context)

    assert repository.scope is FeatureScope.REPOSITORY
    assert repository.as_dict()["repository_test_file_ratio"] == 0.5
    assert change.scope is FeatureScope.CHANGE
    assert change.as_dict()["change_files"] == 1
    assert change.as_dict()["change_churn"] >= 3
