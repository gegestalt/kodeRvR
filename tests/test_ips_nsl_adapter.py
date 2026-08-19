"""Contract tests for the real NSL-KDD IPS episode adapter."""

from __future__ import annotations

import pandas as pd

from data import COLUMN_NAMES
from ips.nsl_adapter import NslIpsConfig, build_nsl_ips_evidence


def _frame(rows: int, prefix: str) -> pd.DataFrame:
    records = []
    labels = ("normal", "neptune", "satan", "guess_passwd", "buffer_overflow")
    for index in range(rows):
        row = {column: float((index + offset) % 7) for offset, column in enumerate(COLUMN_NAMES[:-2])}
        row.update(
            protocol_type=("tcp", "udp", "icmp")[index % 3],
            service=("http", "smtp", "ftp")[index % 3],
            flag=("SF", "S0")[index % 2],
            label=labels[index % len(labels)],
            difficulty=index % 21,
        )
        records.append(row)
    return pd.DataFrame(records)


def test_nsl_adapter_builds_disjoint_scored_proxy_episodes():
    evidence = build_nsl_ips_evidence(
        _frame(100, "train"),
        _frame(50, "test"),
        NslIpsConfig(max_train_rows=100, max_test_rows=50, episode_size=5, folds=2),
    )
    assert evidence.metadata["sequence_semantics"] == "ordered-row proxy; NSL-KDD has no timestamps"
    assert evidence.metadata["detector_train_scoring"] == "out-of-fold"
    assert evidence.metadata["detector_test_scoring"] == "fit on train only"
    assert set(evidence.train_events["split"]) == {"train_oof"}
    assert set(evidence.test_events["split"]) == {"official_test"}
    assert evidence.train_events["threat_probability"].between(0, 1).all()
    assert evidence.test_events["anomaly_score"].between(0, 1).all()
    train_groups = {episode.group_id for episode in evidence.splits.train + evidence.splits.validation}
    test_groups = {episode.group_id for episode in evidence.splits.test}
    assert train_groups.isdisjoint(test_groups)
    assert set(evidence.test_events["attack_family"]) >= {"normal", "DoS", "Probe", "R2L", "U2R"}
    assert evidence.resources["wall_time_s"] >= 0
    assert evidence.resources["max_rss_mb"] > 0
