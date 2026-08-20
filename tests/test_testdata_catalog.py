from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_provenance.testdata import (
    FixtureCategory,
    GroundTruthScope,
    TestFixtureRecord,
    categorize_devgpt_row,
    categorize_swebench_row,
    validate_fixture_catalog,
    verify_fetched_fixtures,
)


def test_swebench_is_correctness_ground_truth_not_authorship_ground_truth():
    record = categorize_swebench_row({
        "instance_id": "django__django-123",
        "repo": "django/django",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the failing behavior.",
        "patch": "diff --git a/x.py b/x.py",
        "test_patch": "diff --git a/test_x.py b/test_x.py",
    })

    assert record.category is FixtureCategory.ISSUE_PATCH_CORRECTNESS
    assert record.ground_truth_scope is GroundTruthScope.PATCH_AND_TEST_OUTCOME
    assert record.authorship_label is None


def test_devgpt_is_ai_association_not_ai_authorship_label():
    record = categorize_devgpt_row({
        "URL": "https://chat.openai.com/share/example",
        "Status": "200",
        "MentionedURL": "https://github.com/o/r/commit/abc",
        "MentionedSource": "commit",
        "MentionedProperty": "message",
        "MentionedAuthor": "developer",
    })

    assert record.category is FixtureCategory.AI_ASSISTED_TRACE
    assert record.ground_truth_scope is GroundTruthScope.AI_LINK_ASSOCIATION
    assert record.authorship_label is None
    assert "not proof" in record.label_limitations.lower()


def test_catalog_rejects_duplicate_records_and_missing_hashes():
    record = TestFixtureRecord(
        dataset_id="fixture",
        record_id="1",
        category=FixtureCategory.SECURITY_ANALYZER_ORACLE,
        ground_truth_scope=GroundTruthScope.ANALYZER_EXPECTATION,
        source_url="https://example.test/source",
        source_revision="abc",
        license="MIT",
        payload_hash="a" * 64,
        local_path="security/one.json",
        label_limitations="Analyzer-specific oracle only.",
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_fixture_catalog((record, record))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_fixture_catalog((TestFixtureRecord(**{**record.__dict__, "payload_hash": "bad"}),))


def test_fetched_catalog_has_all_required_categories(tmp_path: Path):
    catalog = [
        {
            "dataset_id": "swebench-lite",
            "record_id": "swe-1",
            "category": "issue_patch_correctness",
            "ground_truth_scope": "patch_and_test_outcome",
            "source_url": "https://www.swebench.com/",
            "source_revision": "rev",
            "license": "source-repository-specific",
            "payload_hash": "a" * 64,
            "local_path": "swebench/swe-1.json",
            "label_limitations": "No authorship label.",
            "authorship_label": None,
        },
        {
            "dataset_id": "devgpt",
            "record_id": "dev-1",
            "category": "ai_assisted_trace",
            "ground_truth_scope": "ai_link_association",
            "source_url": "https://github.com/NAIST-SE/DevGPT",
            "source_revision": "rev",
            "license": "NOASSERTION",
            "payload_hash": "b" * 64,
            "local_path": "devgpt/dev-1.json",
            "label_limitations": "Association is not proof.",
            "authorship_label": None,
        },
        {
            "dataset_id": "codeql-python-tests",
            "record_id": "sec-1",
            "category": "security_analyzer_oracle",
            "ground_truth_scope": "analyzer_expectation",
            "source_url": "https://github.com/github/codeql",
            "source_revision": "rev",
            "license": "MIT",
            "payload_hash": "c" * 64,
            "local_path": "codeql/sec-1.json",
            "label_limitations": "Rule-specific only.",
            "authorship_label": None,
        },
    ]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    records = validate_fixture_catalog(tuple(TestFixtureRecord.from_dict(row) for row in json.loads(path.read_text())))

    assert {item.category for item in records} == {
        FixtureCategory.ISSUE_PATCH_CORRECTNESS,
        FixtureCategory.AI_ASSISTED_TRACE,
        FixtureCategory.SECURITY_ANALYZER_ORACLE,
    }


def test_cached_payload_hash_mismatch_is_rejected(tmp_path: Path):
    payload = {"instance_id": "repo__repo-1", "base_commit": "a" * 40}
    record = categorize_swebench_row(payload)
    path = tmp_path / record.local_path
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**payload, "base_commit": "b" * 40}), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_fetched_fixtures(tmp_path, (record,))
