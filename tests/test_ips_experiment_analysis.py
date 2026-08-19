"""Tests for dynamic model-selection analysis."""

import pandas as pd

from ips.experiment_analysis import add_objective_scores, detect_outliers, pareto_mask


def sample_frame():
    return pd.DataFrame(
        {
            "name": ["fast", "balanced", "dominated", "large"],
            "hidden_dim": [16, 32, 32, 64],
            "gamma": [0.95, 0.99, 0.99, 0.99],
            "learning_rate": [1e-3] * 4,
            "batch_size": [16, 32, 32, 64],
            "target_update_steps": [50, 75, 75, 100],
            "seed": [42] * 4,
            "safety_quality": [0.80, 0.95, 0.70, 0.96],
            "runtime_s": [1.0, 2.0, 3.0, 4.0],
            "checkpoint_mb": [0.1, 0.2, 0.2, 0.5],
            "peak_python_memory_mb": [1.0, 2.0, 2.0, 20.0],
            "throughput_steps_s": [100, 80, 60, 40],
            "containment_rate": [0.8, .95, .7, .96],
            "compromise_rate": [0.0] * 4,
            "false_preventions_per_episode": [0.0] * 4,
        }
    )


def test_pareto_rejects_strictly_dominated_configuration():
    mask = pareto_mask(sample_frame())
    assert not mask.iloc[2]
    assert mask.iloc[0]


def test_objective_scores_are_bounded():
    scored = add_objective_scores(sample_frame())
    assert scored["overall_score"].between(0, 1).all()


def test_outlier_analysis_explains_association_without_claiming_causality():
    out = detect_outliers(sample_frame())
    flagged = out[out["is_outlier"]]
    assert not flagged.empty
    assert flagged["outlier_explanation"].str.contains("does not prove causality").all()
