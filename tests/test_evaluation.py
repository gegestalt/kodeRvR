from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from code_provenance.evaluation import (
    audit_near_duplicate_leakage,
    evaluate_predictions,
    validate_group_disjoint,
)
from code_provenance.schema import AuthorshipLabel, CodeSample, EvidenceSource


def samples() -> list[CodeSample]:
    return [CodeSample(
        sample_id=str(index), repository_id=f"repo-{index // 2}", group_id=f"group-{index // 2}",
        language="python" if index < 3 else "go", code=code,
        label=label, label_source=EvidenceSource.CONTROLLED_GENERATION,
    ) for index, (code, label) in enumerate((
        ("def shared(): return 1", AuthorshipLabel.HUMAN),
        ("def shared(): return 1", AuthorshipLabel.AI),
        ("def other(): return 2", AuthorshipLabel.HYBRID),
        ("func other() int { return 2 }", AuthorshipLabel.HUMAN),
    ))]


def test_evaluation_reports_language_class_calibration_and_selective_metrics():
    report = evaluate_predictions(
        samples(),
        predicted_labels=("human", "human", "hybrid", "ai"),
        probabilities=np.asarray([
            [.8, .1, .1], [.7, .2, .1], [.1, .2, .7], [.1, .8, .1],
        ]),
        class_names=("human", "ai", "hybrid"),
        abstained=(False, False, False, True),
        ood_scores=(.1, .2, .3, .9),
        public_reuse_fractions=(0.0, .5, 0.0, 0.0),
    )

    assert set(report) >= {
        "per_class", "per_language", "confusion_matrix", "ece_10bin",
        "multiclass_brier", "selective_coverage", "selective_risk",
        "human_false_positive_rate", "public_reuse_mean",
    }
    assert set(report["per_language"]) == {"go", "python"}
    assert 0 <= report["ece_10bin"] <= 1
    assert 0 <= report["selective_coverage"] <= 1
    assert np.isfinite(report["multiclass_brier"])


def test_near_duplicate_audit_identifies_cross_group_overlap():
    fixture = samples()
    fixture[1] = replace(fixture[1], group_id="group-1")
    result = audit_near_duplicate_leakage(fixture, width=3)
    assert result["cross_group_pair_count"] >= 1
    assert result["max_overlap"] == 1.0


def test_group_disjoint_validation_rejects_leakage():
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_disjoint(samples()[:2], samples()[:1])