"""Run the chronological CICAPT detector benchmark without unlocking day four."""

from __future__ import annotations

import json
import pandas as pd

from ips.analysis.cicapt import benchmark_detectors
from ips.workspace import ProjectPaths


def run() -> dict[str, object]:
    project = ProjectPaths.discover(); output = project.results / "cicapt_iiot2024"
    events = pd.read_parquet(output / "phase2_chronological_events.parquet")
    manifest = json.loads((output / "event_manifest.json").read_text())
    metrics, families = benchmark_detectors(events, manifest["feature_columns"])
    metrics.to_csv(output / "detector_temporal_metrics.csv", index=False)
    families.to_csv(output / "detector_tactic_recall.csv", index=False)
    status = {"train_day":"2023-12-01","validation_day":"2023-12-02","development_test_day":"2023-12-03",
              "locked_final_holdout_day":"2023-12-04","locked_rows_scored":0,
              "models":sorted(metrics.detector.unique()),"primary_metric":"PR-AUC plus tactic recall",
              "policy_training":"BLOCKED until detector finalist is selected on validation/development evidence"}
    (output / "benchmark_status.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    return status


if __name__ == "__main__": print(json.dumps(run(), indent=2))
