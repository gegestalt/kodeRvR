"""Audit locally acquired official CICAPT-IIoT2024 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ips.adapters.cicapt import CicaptPaths, build_campaign_timeline, build_multimodal_manifest, validate_provenance_graph
from ips.workspace import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--attack-info", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/notebook_ips_lab/cicapt_iiot2024"))
    args = parser.parse_args()
    project = ProjectPaths.discover()
    inventory = project.cicapt_inventory()
    primary = project.cicapt_primary_artifacts()
    def resolve(explicit: Path | None, modality: str) -> Path:
        if explicit is not None:
            return explicit
        preferred = {"network_csv": primary["phase2_network"], "provenance": primary["phase2_provenance"], "attack_info": primary["attack_info"]}[modality]
        if preferred.exists():
            return preferred
        candidates = inventory[modality]
        if len(candidates) != 1:
            raise SystemExit(f"expected exactly one {modality} artifact, found {len(candidates)}; pass --{modality.replace('_', '-')} explicitly")
        return candidates[0]
    paths = CicaptPaths(
        resolve(args.network, "network_csv"),
        resolve(args.provenance, "provenance"),
        resolve(args.attack_info, "attack_info"),
    )
    manifest = build_multimodal_manifest(paths)
    provenance = pd.read_csv(paths.provenance, low_memory=False)
    attacks = pd.read_csv(paths.attack_info, low_memory=False)
    graph_audit = validate_provenance_graph(provenance)
    timeline = build_campaign_timeline(attacks)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "multimodal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "provenance_graph_audit.json").write_text(json.dumps(graph_audit, indent=2) + "\n", encoding="utf-8")
    timeline.to_parquet(args.output / "hidden_attack_timeline.parquet", index=False)
    print(json.dumps({"manifest": str(args.output / "multimodal_manifest.json"), "graph": graph_audit,
                      "campaign_steps": len(timeline), "tactics": sorted(timeline.tactic.unique())}, indent=2))


if __name__ == "__main__":
    main()
