from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_provenance.external_evaluation import (
    AcquisitionFailureKind,
    AnalyzerOracleResult,
    ExternalDataError,
    ExternalEvaluationRecord,
    evaluate_codeql_fixture,
    evaluation_json,
    fetch_with_classified_failures,
    load_catalog,
    summarize_oracles,
    validate_devgpt_payload,
    validate_swebench_payload,
    verified_payload,
    write_verified_cache,
)
from code_provenance.testdata import categorize_devgpt_row, categorize_swebench_row, canonical_hash


def test_cached_fixture_tampering_is_rejected(tmp_path: Path):
    expected = canonical_hash({"trusted": True})
    with pytest.raises(ExternalDataError) as caught:
        verified_payload(b'{"trusted":false}', expected)
    assert caught.value.kind is AcquisitionFailureKind.HASH_MISMATCH


def test_verified_cache_is_not_silently_overwritten(tmp_path: Path):
    path = tmp_path / "fixture.json"
    payload = b"verified"
    import hashlib
    digest = hashlib.sha256(payload).hexdigest()
    write_verified_cache(path, payload, digest)
    path.write_bytes(b"tampered")
    with pytest.raises(ExternalDataError, match="hash mismatch"):
        write_verified_cache(path, payload, digest)


def test_network_failure_is_distinct_from_analyzer_failure():
    class Response:
        status_code = 503
        content = b""

    with pytest.raises(ExternalDataError) as caught:
        fetch_with_classified_failures("https://upstream.test", lambda *_args, **_kwargs: Response())
    assert caught.value.kind is AcquisitionFailureKind.UPSTREAM_UNAVAILABLE


def test_rate_limit_is_classified_separately():
    class Response:
        status_code = 429
        content = b""

    with pytest.raises(ExternalDataError) as caught:
        fetch_with_classified_failures("https://upstream.test", lambda *_args, **_kwargs: Response())
    assert caught.value.kind is AcquisitionFailureKind.RATE_LIMIT


def test_unsupported_upstream_schema_fails_explicitly(tmp_path: Path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"unexpected": "object"}), encoding="utf-8")
    with pytest.raises(ExternalDataError) as caught:
        load_catalog(path)
    assert caught.value.kind is AcquisitionFailureKind.SCHEMA_CHANGE


def test_swebench_schema_requires_patch_research_fields():
    with pytest.raises(ExternalDataError, match="problem_statement"):
        validate_swebench_payload({"instance_id": "x", "repo": "o/r", "base_commit": "a", "patch": "d"})


def test_devgpt_schema_requires_trace_and_artifact():
    with pytest.raises(ExternalDataError, match="MentionedURL"):
        validate_devgpt_payload({"URL": "https://chat.openai.com/share/x"})


def test_swebench_record_never_receives_authorship_label():
    row = {"instance_id": "o__r-1", "repo": "o/r", "base_commit": "a" * 40,
           "problem_statement": "p", "patch": "diff", "test_patch": "tests"}
    validate_swebench_payload(row)
    assert categorize_swebench_row(row).authorship_label is None


def test_devgpt_association_never_becomes_authorship_ground_truth():
    row = {"URL": "https://chat.openai.com/share/x", "MentionedURL": "https://github.com/o/r/issues/1"}
    validate_devgpt_payload(row)
    assert categorize_devgpt_row(row).authorship_label is None


def test_codeql_oracle_is_executed_against_security_analyzer():
    result = evaluate_codeql_fixture(
        "sql-format", "CWE-089", 'cursor.execute(f"SELECT * FROM x WHERE id={user}")',
        expected_detection=True, supported_rule_ids=("SQL-FORMAT",),
    )
    assert result.detected is True
    assert result.matched_rule_ids == ("SQL-FORMAT",)


def test_known_codeql_gap_is_reported_as_miss_not_relabelled():
    result = evaluate_codeql_fixture(
        "sql-concat", "CWE-089", 'cursor.execute("SELECT " + user)',
        expected_detection=True, supported_rule_ids=("SQL-FORMAT",),
    )
    assert result.expected_detection is True
    assert result.detected is False
    summary = summarize_oracles((result,))
    assert summary.false_negative == 1
    assert summary.recall == 0.0


def test_precision_is_not_calculated_without_negative_controls():
    result = AnalyzerOracleResult("x", "CWE", False, True, (), False)
    summary = summarize_oracles((result,))
    assert summary.precision is None
    assert summary.recall is None


def test_clean_controls_enable_false_positive_measurement():
    positive = AnalyzerOracleResult("p", "CWE", True, True, ("R",), True)
    negative = AnalyzerOracleResult("n", "clean", False, False, (), True)
    summary = summarize_oracles((positive, negative))
    assert summary.precision == summary.recall == summary.f1 == 1.0


def test_evaluation_expected_and_observed_values_are_separate():
    row = ExternalEvaluationRecord("fixture", {"security_issue": "sql_injection"}, {"detected": False})
    encoded = json.loads(evaluation_json((row,)))[0]
    assert encoded["expected"] == {"security_issue": "sql_injection"}
    assert encoded["observed"] == {"detected": False}
