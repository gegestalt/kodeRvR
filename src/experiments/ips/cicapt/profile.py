"""Profile CICAPT network phases without loading multi-GB CSVs into memory."""

from __future__ import annotations

import json

from ips.adapters.cicapt import build_campaign_timeline, profile_network_csv
from ips.workspace import ProjectPaths
import pandas as pd


def run() -> dict[str, object]:
    project = ProjectPaths.discover()
    source = project.cicapt_source()
    if source is None:
        raise FileNotFoundError("CICAPT download not found")
    artifacts = project.cicapt_primary_artifacts()
    phase1 = artifacts["phase1_network"]
    phase2 = artifacts["phase2_network"]
    attack_info = artifacts["attack_info"]
    output = project.results / "cicapt_iiot2024"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase1": profile_network_csv(phase1),
        "phase2": profile_network_csv(phase2),
        "campaign": {
            "steps": len(timeline := build_campaign_timeline(pd.read_csv(attack_info))),
            "tactics": timeline.tactic.value_counts().to_dict(),
            "start": timeline.attack_time.min().isoformat(),
            "end": timeline.attack_time.max().isoformat(),
        },
    }
    (output / "network_profile.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timeline.to_parquet(output / "hidden_campaign_timeline.parquet", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
