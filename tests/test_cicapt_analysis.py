import pandas as pd

from ips.analysis.cicapt import benchmark_detectors


def test_cicapt_benchmark_never_scores_locked_holdout():
    rows = []
    for role in ("train", "validation", "development_test", "locked_final_holdout"):
        for i in range(30):
            attack = i % 5 == 0
            rows.append({"split_role": role, "x": float(attack) + i / 100,
                         "attack_present": attack, "attack_tactic": "attack" if attack else "0"})
    metrics, families = benchmark_detectors(pd.DataFrame(rows), ["x"], seed=42)
    assert set(metrics.evaluation_role) == {"validation", "development_test"}
    assert "locked_final_holdout" not in set(families.evaluation_role)
    assert {"logistic", "hist_gradient_boosting", "benign_isolation_forest"} <= set(metrics.detector)
