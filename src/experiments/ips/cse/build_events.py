"""Build timestamped CSE-CIC-IDS2018 IPS events from three official days.

Run from the repository root:
    .venv/bin/python -m experiments.ips.cse.build_events
"""

from __future__ import annotations

import json
from pathlib import Path

from ips.adapters.cse_temporal import (
    TemporalDetectorConfig,
    build_temporal_detector_events,
    read_cse_day_sample,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "data" / "cse_cic_ids2018" / "processed"
OUTPUT = ROOT / "data" / "ips_events"
FILES = (
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
)


def run() -> dict[str, object]:
    frames = [
        read_cse_day_sample(SOURCE / name, seed=42 + index)
        for index, name in enumerate(FILES)
    ]
    events, audit = build_temporal_detector_events(
        frames, TemporalDetectorConfig(folds=5, max_iter=60, window_seconds=30)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUT / "cse_cic_ids2018_temporal_events.parquet"
    audit_path = OUTPUT / "cse_cic_ids2018_temporal_audit.csv"
    events.to_parquet(events_path, index=False)
    audit.to_csv(audit_path, index=False)
    report = {
        "source": "official CSE-CIC-IDS2018 generated flow CSVs",
        "files": list(FILES),
        "rows": int(len(events)),
        "episodes": int(events.episode_id.nunique()),
        "groups": int(events.group_id.nunique()),
        "days": events.groupby("split_role")["source_day"].unique().map(list).to_dict(),
        "score_origin": events.score_origin.value_counts().to_dict(),
        "families": events.attack_family.value_counts().to_dict(),
        "outcome_evidence": "counterfactual IPS transitions; real detector inputs",
        "ope_status": "blocked until observed shadow deployment outcomes exist",
    }
    (OUTPUT / "cse_cic_ids2018_temporal_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
