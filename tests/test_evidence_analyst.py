import pytest

from ips.analysis.analyst import EvidenceAnalyst


EVIDENCE = {"detector": {"best_model": "HGB", "pr_auc": .25},
            "policy": {"best_policy": "Rule", "evidence_grade": "SIMULATED"},
            "gates": {"calibration": "FAIL", "schema": "PASS"}}


def test_fallback_analyst_is_grounded_and_deterministic() -> None:
    first = EvidenceAnalyst().explain(EVIDENCE)
    assert first == EvidenceAnalyst().explain(EVIDENCE)
    assert "HGB" in first and "0.2500" in first and "calibration" in first


def test_external_analyst_rejects_invented_number() -> None:
    analyst = EvidenceAnalyst(lambda evidence: "The unsupported result is 73.2.")
    with pytest.raises(ValueError, match="unsupported numbers"):
        analyst.explain(EVIDENCE)
