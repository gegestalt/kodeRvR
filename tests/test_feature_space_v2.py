from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from code_provenance.change_context import ChangeIntent, build_change_context
from code_provenance.dependency_context import build_dependency_context
from code_provenance.feature_space import (
    FEATURE_DEFINITIONS,
    FeatureFamily,
    FeatureScope,
    extract_change_features,
    extract_dependency_features,
    extract_repository_features,
    sample_feature_vector,
)
from code_provenance.features import FEATURE_NAMES
from code_provenance.schema import AuthorshipLabel
from code_provenance.symbol_index import build_changed_symbol_index
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


def test_dependency_features_expose_graph_quality_and_changed_symbol_impact(tmp_path: Path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / "app.py").write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    (tmp_path / "consumer.py").write_text(
        "from app import helper\n\ndef use():\n    return helper()\n",
        encoding="utf-8",
    )
    git(tmp_path, "add", "app.py", "consumer.py")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "app.py").write_text(
        "def helper():\n    return 2\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    context = build_change_context(tmp_path, intent=ChangeIntent("Change helper", "test"))
    symbols = build_changed_symbol_index(tmp_path, context)
    dependencies = build_dependency_context(tmp_path, context, symbols)

    vector = extract_dependency_features(dependencies)
    values = vector.as_dict()

    assert vector.scope is FeatureScope.SYMBOL
    assert values["dependency_nodes"] >= 4
    assert values["dependency_resolved_edges"] >= 2
    assert values["dependency_unresolved_ratio"] < 1.0
    assert values["dependency_changed_symbol_count"] == 1
    assert values["dependency_max_transitive_dependents"] >= 1
    assert np.isfinite(list(values.values())).all()


def test_dependency_features_are_finite_for_an_empty_repository(tmp_path: Path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    git(tmp_path, "commit", "--allow-empty", "-qm", "empty")
    context = build_change_context(tmp_path)
    symbols = build_changed_symbol_index(tmp_path, context)
    dependencies = build_dependency_context(tmp_path, context, symbols)

    values = extract_dependency_features(dependencies).as_dict()

    assert values["dependency_nodes"] == 0
    assert values["dependency_edges"] == 0
    assert values["dependency_changed_symbol_count"] == 0
    assert np.isfinite(list(values.values())).all()
