"""Build leakage-safe, attack-preserving CICAPT phase-2 evidence."""

from __future__ import annotations

import json

import pandas as pd

from ips.adapters.cicapt import build_attack_preserving_sample, build_campaign_timeline
from ips.workspace import ProjectPaths


def run(*, benign_fraction: float = .01, seed: int = 42) -> dict[str, object]:
    project = ProjectPaths.discover()
    source = project.cicapt_source()
    if source is None:
        raise FileNotFoundError("CICAPT download not found")
    output = project.results / "cicapt_iiot2024"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = project.cicapt_primary_artifacts()
    events, manifest = build_attack_preserving_sample(
        artifacts["phase2_network"], benign_fraction=benign_fraction, seed=seed
    )
    timeline = build_campaign_timeline(pd.read_csv(artifacts["attack_info"]))
    attack_times = timeline[["attack_time", "campaign_step", "tactic", "technique"]].copy()
    attack_times["timestamp"] = attack_times.attack_time.astype("int64") / 1e9
    events = pd.merge_asof(
        events.sort_values("timestamp"), attack_times.drop(columns="attack_time").sort_values("timestamp"),
        on="timestamp", direction="nearest", tolerance=15 * 60,
    )
    events["campaign_step"] = events.campaign_step.astype("Int64")
    events["policy_visible_campaign_metadata"] = False
    events.to_parquet(output / "phase2_attack_preserving_events.parquet", index=False)
    timeline.to_parquet(output / "hidden_campaign_timeline.parquet", index=False)
    days = sorted(events.source_day.unique())
    if len(days) != 4:
        raise ValueError(f"expected four CICAPT phase-2 campaign days, found {days}")
    roles = dict(zip(days, ("train", "validation", "development_test", "locked_final_holdout"), strict=True))
    events.assign(split_role=events.source_day.map(roles)).to_parquet(
        output / "phase2_chronological_events.parquet", index=False
    )
    manifest.update({"days": days, "day_roles": roles, "campaign_steps": len(timeline),
                     "join_tolerance_seconds": 900, "evidence_kind": "passive_observation_not_intervention_outcome"})
    (output / "event_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
