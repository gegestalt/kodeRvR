"""Leakage, chronology, and OPE-gate tests for timestamped IPS evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ips.actions import IpsAction
from ips.adapters.cse_temporal import (
    TemporalDetectorConfig,
    build_temporal_detector_events,
    log_shadow_decision,
    require_observed_ope_evidence,
    split_events_by_role,
)


def _day(day: int, family: str) -> pd.DataFrame:
    rows = []
    for hour in range(4):
        for step in range(6):
            attack = step >= 3 and family != "Benign"
            rows.append(
                {
                    "Timestamp": f"{day:02d}/02/2018 {hour:02d}:00:{step:02d}",
                    "Label": family if attack else "Benign",
                    "feature_a": hour + attack * 4 + step / 10,
                    "feature_b": step + attack * 3,
                    "source_day": f"2018-02-{day:02d}",
                }
            )
    return pd.DataFrame(rows)


def test_temporal_detector_uses_oof_train_and_past_only_heldout_scoring():
    frames = [_day(14, "FTP-BruteForce"), _day(15, "DoS-GoldenEye"), _day(16, "DoS-Hulk")]
    events, audit = build_temporal_detector_events(
        frames,
        TemporalDetectorConfig(folds=3, max_iter=10, window_seconds=60),
    )
    assert list(audit.sort_values("role").role.unique()) == ["final_test", "train", "validation"]
    assert set(events.score_origin) == {"train_group_oof", "train_fitted_heldout"}
    assert events.threat_probability.between(0, 1).all()
    assert events.anomaly_score.between(0, 1).all()
    assert events.timestamp.min() > 1_500_000_000
    splits = split_events_by_role(events)
    assert not ({episode.group_id for episode in splits.train} & {episode.group_id for episode in splits.test})


def test_shadow_logger_records_full_valid_distribution_and_executed_propensity():
    mask = np.array([True, True, True, False, False, False, False])
    row = log_shadow_decision(
        episode_id="ep", timestamp=1.0, proposed=IpsAction.RATE_LIMIT,
        executed=IpsAction.RATE_LIMIT, action_mask=mask, epsilon=0.2,
        evidence_kind="counterfactual_dataset_replay",
    )
    probabilities = np.asarray(row["behavior_probabilities"])
    assert np.isclose(probabilities.sum(), 1)
    assert np.all(probabilities[~mask] == 0)
    assert row["behavior_propensity"] == probabilities[IpsAction.RATE_LIMIT]


def test_doubly_robust_gate_refuses_counterfactual_or_missing_outcomes():
    frame = pd.DataFrame(
        [{
            "evidence_kind": "counterfactual_dataset_replay",
            "behavior_propensity": 0.5,
            "observed_reward": 1.0,
            "action": 0,
        }]
    )
    with pytest.raises(ValueError, match="observed shadow deployment"):
        require_observed_ope_evidence(frame)
    frame["evidence_kind"] = "observed_shadow_deployment"
    frame["observed_reward"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        require_observed_ope_evidence(frame)


def test_official_repeated_header_rows_are_removed(tmp_path):
    source = tmp_path / "day.csv"
    source.write_text(
        "Timestamp,Label,x\n"
        "14/02/2018 10:00:00,Benign,1\n"
        "Timestamp,Label,x\n"
        "14/02/2018 11:00:00,FTP-BruteForce,2\n",
        encoding="utf-8",
    )
    from ips.adapters.cse_temporal import read_cse_day_sample

    sampled = read_cse_day_sample(source, benign_rows=1, attack_rows_per_family=1, chunksize=2)
    assert len(sampled) == 2
    assert "Timestamp" not in set(sampled.Timestamp)
