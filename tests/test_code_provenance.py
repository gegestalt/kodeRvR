from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from code_provenance.dataset import load_manifest
from code_provenance.features import FEATURE_NAMES, extract_features
from code_provenance.model import ModelConfig, ProvenanceClassifier
from code_provenance.repository import (
    build_provenance_observations,
    recent_commit_metadata,
    working_tree_samples,
)
from code_provenance.reuse import PublicReuseIndex
from code_provenance.schema import AuthorshipLabel, CodeSample, EvidenceSource
from code_provenance.security import package_risk, python_dependencies, scan_code


def sample(index: int, label: AuthorshipLabel) -> CodeSample:
    human = f"def compact_{index}(x):\n    return x + {index}\n"
    ai = f'''def comprehensively_process_value_{index}(input_value):
    """Process the supplied value using a clear and robust workflow."""
    if input_value is None:
        raise ValueError("input_value must not be None")
    processed_value = input_value + {index}
    return processed_value
'''
    hybrid = human + f"\n# Added validation\nif __name__ == '__main__':\n    print(compact_{index}({index}))\n"
    return CodeSample(
        f"s{index}-{label}", f"repo-{index // 3}", f"group-{index // 3}", "python",
        {AuthorshipLabel.HUMAN: human, AuthorshipLabel.AI: ai, AuthorshipLabel.HYBRID: hybrid}[label],
        label=label, label_source=EvidenceSource.CONTROLLED_GENERATION,
    )


def test_features_are_finite_and_schema_stable():
    values = extract_features(sample(1, AuthorshipLabel.HUMAN))
    assert tuple(values) == FEATURE_NAMES
    assert np.isfinite(list(values.values())).all()


def test_reuse_index_detects_shared_token_regions():
    index = PublicReuseIndex(width=3)
    index.add(["def shared(x):\n return x + 1"])
    assert index.overlap_fraction("def shared(x):\n return x + 1") == 1
    assert index.overlap_fraction("class EntirelyDifferent: pass") < .5


def test_group_safe_model_returns_calibrated_or_abstained_estimate():
    samples = []
    labels = [AuthorshipLabel.HUMAN, AuthorshipLabel.AI, AuthorshipLabel.HYBRID]
    for group in range(9):
        for offset, label in enumerate(labels):
            item = sample(group * 3 + offset, label)
            samples.append(CodeSample(**{**item.__dict__, "group_id": f"group-{group}"}))
    model = ProvenanceClassifier(ModelConfig(folds=3, trees=30, confidence_threshold=.2, ood_threshold=1.0))
    metrics = model.fit(samples)
    query = sample(100, AuthorshipLabel.HUMAN)
    query = CodeSample(**{**query.__dict__, "label": AuthorshipLabel.UNKNOWN})
    estimate = model.predict(query, public_reuse_fraction=.1)
    assert metrics["groups"] == 9
    assert metrics["features"] >= 50
    assert {
        "group_oof_balanced_accuracy", "group_oof_mcc", "group_oof_log_loss",
        "group_oof_multiclass_brier", "group_oof_ece_10bin",
        "group_oof_roc_auc_ovr_macro", "group_oof_pr_auc_macro",
        "class_human_precision", "class_ai_recall", "class_hybrid_f1",
        "confusion_human_as_ai",
    } <= set(metrics)
    assert np.isfinite(list(metrics.values())).all()
    assert abs(sum(estimate.probabilities.values()) - 1) < 1e-6
    assert estimate.organic_fraction is not None
    assert "proof" in estimate.claim
    explanation = model.explain_prediction(query, top_k=5)
    assert len(explanation) == 5
    assert {item["name"] for item in explanation} <= set(FEATURE_NAMES)
    assert all("value" in item and "importance" in item for item in explanation)


def test_manifest_rejects_heuristic_training_labels(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "sample_id": "1", "repository_id": "r", "group_id": "g", "language": "python",
        "path": "a.py", "code": "pass", "label": "ai", "label_source": "heuristic",
        "generator_family": "unknown", "dataset_id": "controlled", "dataset_version": "1",
        "author_group_id": "author-1", "dataset_role": "train",
        "provenance_source": "https://example.test/evidence/1",
        "source_url": "https://github.com/example/repo", "source_revision": "a" * 40,
        "content_hash": __import__("hashlib").sha256(b"pass").hexdigest(),
        "license": "MIT", "acquisition_date": "2026-08-20",
    }]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="heuristic"):
        load_manifest(path)


def test_controlled_manifest_preserves_verified_identity_and_role(tmp_path: Path):
    code = "def answer():\n    return 42\n"
    import hashlib
    path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "sample_id": "sample-1", "repository_id": "repo-1", "group_id": "repo-group-1",
        "author_group_id": "author-1", "language": "python", "path": "answer.py",
        "code": code, "label": "ai", "label_source": "controlled_generation",
        "generator_family": "test-generator", "dataset_id": "controlled-v1",
        "dataset_version": "2026-08-20", "dataset_role": "train",
        "provenance_source": "https://example.test/evidence/sample-1",
        "source_url": "https://github.com/example/repo", "source_revision": "a" * 40,
        "content_hash": hashlib.sha256(code.encode()).hexdigest(), "license": "MIT",
        "acquisition_date": "2026-08-20",
    }]).to_csv(path, index=False)

    records = load_manifest(path)

    assert records[0].dataset_id == "controlled-v1"
    assert records[0].author_group_id == "author-1"
    assert records[0].dataset_role.value == "train"
    assert records[0].source_revision == "a" * 40


def test_structural_github_record_cannot_become_training_label(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "sample_id": "live-1", "repository_id": "repo-1", "group_id": "repo-1",
        "author_group_id": "unknown", "language": "go", "path": "main.go", "code": "package main",
        "label": "human", "label_source": "unlabelled", "generator_family": "unknown",
        "dataset_id": "github-live", "dataset_version": "2026-08-20", "dataset_role": "structural_only",
        "provenance_source": "https://github.com/example/repo", "source_url": "https://github.com/example/repo",
        "source_revision": "a" * 40, "content_hash": __import__("hashlib").sha256(b"package main").hexdigest(),
        "license": "MIT", "acquisition_date": "2026-08-20",
    }]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="structural_only|unlabelled"):
        load_manifest(path)


def test_security_and_package_checks_separate_missing_from_unknown():
    code = "import imaginary_auth\nsecret = '1234567890'\neval(user_input)"
    signals = scan_code(code)
    assert {signal.rule_id for signal in signals} >= {"SECRET-LITERAL", "PY-EVAL"}
    rows = package_risk(python_dependencies(code), lambda _: "missing")
    assert rows[0]["supply_chain_risk"] == "critical"


def test_repository_extraction_is_read_only_and_supported():
    root = Path(__file__).resolve().parents[1]
    samples = working_tree_samples(root)
    commits = recent_commit_metadata(root, limit=3)
    assert any(item.path.endswith(".py") for item in samples)
    assert 1 <= len(commits) <= 3


def test_explicit_provenance_markers_remain_observations_not_labels():
    result = build_provenance_observations([{
        "commit_sha": "abc",
        "commit_message": "Add feature\n\nCo-authored-by: Copilot <bot@example.test>",
    }])
    assert result["explicit_ai_marker_count"] == 1
    assert result["commits"][0]["explicit_ai_marker"] is True
    assert result["claim"] == "observations are not authorship ground truth"
