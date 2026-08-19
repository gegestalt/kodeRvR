import numpy as np
import pandas as pd

from ips.analysis.data_intelligence import (
    benign_sampling_fidelity, feature_health, join_tolerance_sensitivity,
    label_alignment, tactic_split_coverage, temporal_feature_drift,
)


def test_health_drift_and_fidelity_are_explicit() -> None:
    frame = pd.DataFrame({"x": [0., 0., 1., 2., 3., 4.], "constant": 1.,
                          "split_role": ["train", "train", "validation", "validation", "development_test", "development_test"],
                          "attack_present": [1, 0, 1, 0, 1, 0], "attack_tactic": ["collection", "normal", "new", "normal", "new", "normal"]})
    health = feature_health(frame, ["x", "constant"])
    assert health.set_index("feature").loc["constant", "status"] == "FAIL"
    drift = temporal_feature_drift(frame, ["x"])
    assert set(drift.candidate_role) == {"validation", "development_test"}
    fidelity = benign_sampling_fidelity(frame.iloc[:3], frame.iloc[3:], ["x"])
    assert {"psi", "ks", "jsd", "wasserstein", "psi_band"} <= set(fidelity)


def test_holdout_is_concealed_and_alignment_is_auditable() -> None:
    events = pd.DataFrame({"split_role": ["train", "locked_final_holdout"], "attack_present": [True, True],
                           "attack_tactic": ["collection", "cleanup"]})
    coverage = tactic_split_coverage(events)
    assert "cleanup" not in set(coverage.attack_tactic)
    assert coverage.locked_final_holdout.eq("LOCKED").all()
    timeline = pd.DataFrame({"campaign_step": [1], "attack_time": pd.to_datetime([100], unit="s", utc=True),
                             "tactic": ["collection"], "technique": ["find"]})
    aligned = label_alignment(timeline, pd.Series([103.]), pd.Series([110.]))
    assert aligned.loc[0, "network_delta_s"] == 3
    sensitivity = join_tolerance_sensitivity(aligned, [2, 5, 15])
    assert sensitivity.network_matched_steps.tolist() == [0, 1, 1]
