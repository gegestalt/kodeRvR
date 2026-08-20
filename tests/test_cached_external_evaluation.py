from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_provenance.external_evaluation import AnalyzerEvaluationSummary, assert_no_detection_regression


pytestmark = [pytest.mark.evaluation, pytest.mark.external_fixture]


def test_security_baseline_is_pinned_and_never_self_updates():
    path = Path("data/code_health/evaluation_baselines/security.json")
    before = path.read_bytes()
    baseline = json.loads(before)
    current = AnalyzerEvaluationSummary(3, 3, 0, 2, 1, 0, None, 0.0, None)
    assert_no_detection_regression(baseline, current)
    assert path.read_bytes() == before
    assert len(baseline["revision"]) == 40


def test_security_baseline_rejects_a_detection_regression():
    baseline = {"revision": "a" * 40, "detected": 2}
    current = AnalyzerEvaluationSummary(3, 3, 1, 1, 1, 0, 1.0, 0.5, 2 / 3)
    with pytest.raises(AssertionError, match="regression"):
        assert_no_detection_regression(baseline, current)


def test_cached_real_data_evaluation_is_offline_and_verified():
    cache = Path(".cache/code_health")
    if not (cache / "catalog.json").exists():
        pytest.skip("verified local cache absent; fetch explicitly with scripts/fetch_test_fixtures.py")
    from scripts.evaluate_external_fixtures import evaluate
    result = evaluate(cache)
    assert result["sources"] == {"swebench": 3, "devgpt": 3, "codeql": 3}
    assert result["authorship_labels_inferred"] == 0
    assert result["codeql"]["false_negative"] == 2
