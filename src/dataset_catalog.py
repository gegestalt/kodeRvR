"""Dataset catalog and local-availability checks for the lab roadmap.

This is deliberately not a downloader. Its job is to stop vague dataset claims:
each dataset has a role, source URL, expected local layout, and current local
status.

Run:
    .venv/bin/python src/dataset_catalog.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import data as D

DOC_PATH = D.REPO_ROOT / "docs" / "datasets" / "catalog.md"


@dataclass(frozen=True)
class DatasetEntry:
    """Metadata for one dataset track."""

    name: str
    role: str
    source_url: str
    local_dir: Path
    expected_glob: str
    current_status: str
    first_audit: str
    blocked_claim: str


def has_files(local_dir: Path, pattern: str) -> bool:
    """Return whether expected local files exist."""
    return local_dir.exists() and any(local_dir.glob(pattern))


def build_entries() -> list[DatasetEntry]:
    root = D.REPO_ROOT / "data"
    ciciot_csv = has_files(root / "ciciot2023" / "CSV", "*.csv")
    ciciot_dev = has_files(root / "ciciot2023", "*.parquet")
    cse_available = has_files(root / "cse_cic_ids2018", "**/*.csv")
    return [
        DatasetEntry(
            name="NSL-KDD",
            role="Controlled baseline for every new method.",
            source_url="https://www.unb.ca/cic/datasets/nsl.html",
            local_dir=root / "nsl_kdd",
            expected_glob="KDDTrain+.txt",
            current_status="available" if has_files(root / "nsl_kdd", "KDDTrain+.txt") else "missing",
            first_audit="Already complete: preprocessing, supervised baselines, stability.",
            blocked_claim="None for current NSL-KDD baseline; temporal claims remain invalid.",
        ),
        DatasetEntry(
            name="CICIoT2023 dev parquet",
            role="Fast modern IoT dev sample for quality checks and pilot modelling.",
            source_url="https://www.unb.ca/cic/datasets/iotdataset-2023.html",
            local_dir=root / "ciciot2023",
            expected_glob="*.parquet",
            current_status="available" if ciciot_dev else "missing",
            first_audit="Run src/ciciot.py.",
            blocked_claim="Do not treat dev-sample results as full official raw-release results.",
        ),
        DatasetEntry(
            name="CICIoT2023 raw CSV",
            role="Primary modern IoT supervised dataset: binary, 8-category, fine-label.",
            source_url="https://www.unb.ca/cic/datasets/iotdataset-2023.html",
            local_dir=root / "ciciot2023" / "CSV",
            expected_glob="*.csv",
            current_status="available" if ciciot_csv else "blocked_missing_local_files",
            first_audit="Run src/ciciot2023_raw_audit.py.",
            blocked_claim="No full raw CSV modelling or quality claim until CSV files are present.",
        ),
        DatasetEntry(
            name="TON_IoT",
            role="Multimodal SOC/EDR-style track: network, IoT telemetry, Windows/Linux traces.",
            source_url="https://research.unsw.edu.au/projects/toniot-datasets",
            local_dir=root / "ton_iot",
            expected_glob="**/*.csv",
            current_status="available" if has_files(root / "ton_iot", "**/*.csv") else "blocked_missing_local_files",
            first_audit="Create schema inventory for each modality before modelling.",
            blocked_claim="No multimodal fusion, host telemetry, or TON_IoT score is valid yet.",
        ),
        DatasetEntry(
            name="CSE-CIC-IDS2018",
            role="Enterprise-scale/day-based drift and chronological evaluation.",
            source_url="https://www.unb.ca/cic/datasets/ids-2018.html",
            local_dir=root / "cse_cic_ids2018",
            expected_glob="**/*.csv",
            current_status="available" if cse_available else "blocked_missing_local_files",
            first_audit=(
                "Complete for the initial 14–16 February subset; see data/cse_cic_ids2018/SOURCE.md."
                if cse_available else
                "Inventory days/files, strip leakage columns, define chronological split."
            ),
            blocked_claim=(
                "Only the audited three-day subset is supported; full ten-day and causal IPS claims remain blocked."
                if cse_available else
                "No enterprise/day-based drift result is valid yet."
            ),
        ),
        DatasetEntry(
            name="Common NetFlow schema",
            role="Cross-dataset generalization using shared NetFlow-style features.",
            source_url="https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
            local_dir=root / "netflow",
            expected_glob="**/*.csv",
            current_status="available" if has_files(root / "netflow", "**/*.csv") else "blocked_missing_local_files",
            first_audit="Map feature names and labels before any train-one/test-another experiment.",
            blocked_claim="No cross-dataset score is valid without a common feature schema.",
        ),
    ]


def render_catalog(entries: list[DatasetEntry]) -> str:
    lines = [
        "# Dataset Catalog and Local Availability",
        "",
        "This file is the referee ledger for dataset claims. A dataset can be on the "
        "roadmap without being evaluated; the `current_status` column decides what "
        "we are allowed to claim today.",
        "",
        "| Dataset | Role | Source | Local path | Expected files | Status | First audit | Blocked claim |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        local = e.local_dir.relative_to(D.REPO_ROOT)
        lines.append(
            f"| {e.name} | {e.role} | <{e.source_url}> | `{local}` | "
            f"`{e.expected_glob}` | {e.current_status} | {e.first_audit} | {e.blocked_claim} |"
        )
    lines.append("")
    return "\n".join(lines)


def run() -> str:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = render_catalog(build_entries())
    DOC_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    report = run()
    print(report)
    print(f"Wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
