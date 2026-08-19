import json

import numpy as np
import pandas as pd

from ips.analysis.next_phase import (
    InterventionLogger,
    build_locked_holdout_manifest,
    consequence_trace,
    detector_temporal_benchmark,
    source_leakage_curve,
)


def test_source_leakage_curve_reports_single_and_leave_one_out_views():
    frame = pd.DataFrame({
        "source_dataset": ["a"] * 12 + ["b"] * 12,
        "group_id": [f"g{i // 2}" for i in range(24)],
        "x": [0.0] * 12 + [1.0] * 12,
        "noise": np.tile([0.1, 0.2], 12),
    })
    result = source_leakage_curve(frame, ["x", "noise"], folds=3)
    assert {"all", "single_feature", "leave_one_feature_out"} <= set(result.view)
    assert result.balanced_accuracy.between(0, 1).all()
    assert result.query("view == 'single_feature' and feature == 'x'").iloc[0].balanced_accuracy > .9


def test_detector_benchmark_keeps_development_test_separate():
    rows = []
    for role in ("train", "validation", "development_test"):
        for i in range(30):
            attack = i % 3 == 0
            rows.append({"split_role": role, "attack_present": attack,
                         "threat_probability": .8 if attack else .2,
                         "anomaly_score": .7 if attack else .1,
                         "timestamp": float(i)})
    report = detector_temporal_benchmark(pd.DataFrame(rows), seed=42)
    assert {"validation", "development_test"} == set(report.evaluation_role)
    assert {"logistic_fusion", "hist_gradient_boosting", "benign_isolation_forest"} <= set(report.detector)
    assert {"pr_auc", "brier", "ece", "false_alarms_per_hour", "missed_attacks_per_hour"} <= set(report.columns)


def test_holdout_manifest_is_locked_without_reading_unavailable_metrics(tmp_path):
    target = tmp_path / "manifest.json"
    manifest = build_locked_holdout_manifest(
        target, ["Monday-19-02-2018.csv", "Tuesday-20-02-2018.csv"]
    )
    assert manifest["status"] == "LOCKED_UNINSPECTED"
    assert "metrics" not in manifest
    assert json.loads(target.read_text())["files"][0]["sha256"] is None


def test_intervention_logger_validates_and_hash_chains_records(tmp_path):
    logger = InterventionLogger(tmp_path / "interventions.jsonl")
    first = logger.append({"episode_id": "e1", "timestamp": 1.0,
                           "proposed_action": "DROP_FLOW", "executed_action": "MONITOR",
                           "behavior_probabilities": [1/7] * 7, "shield_reason": "blast radius",
                           "attack_continued": True, "attack_succeeded": False,
                           "service_disrupted": False, "legitimate_sessions_affected": 0,
                           "affected_identities": 0, "rollback_performed": False,
                           "recovery_seconds": 0.0, "time_to_containment_seconds": None})
    second = logger.append({**first, "episode_id": "e2", "timestamp": 2.0})
    assert first["record_hash"] != second["record_hash"]
    assert second["previous_hash"] == first["record_hash"]


def test_consequence_trace_exposes_uncontained_truncation():
    events = pd.DataFrame({"episode_id": ["a", "b"], "attack_present": [True, True],
                           "contained": [False, True], "compromised": [False, False],
                           "terminated": [False, True], "truncated": [True, False]})
    trace = consequence_trace(events)
    assert trace.loc[trace.outcome.eq("uncontained_truncated"), "episodes"].iloc[0] == 1
