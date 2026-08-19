"""CICAPT-IIoT2024 multimodal adapter and campaign contracts.

Network packets, provenance graph entities, and attack ground truth remain
separate artifacts.  Attack metadata is evaluation-only and may never enter a
policy observation.  Fusion occurs by declared campaign/time/process links.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CicaptPaths:
    network: Path
    provenance: Path
    attack_info: Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    normalized = {str(column).strip().casefold().replace("_", " "): column for column in frame}
    for alias in aliases:
        if alias.casefold().replace("_", " ") in normalized:
            return normalized[alias.casefold().replace("_", " ")]
    raise ValueError(f"CICAPT attack metadata lacks {label}; columns={list(frame)}")


def build_campaign_timeline(attack_info: pd.DataFrame) -> pd.DataFrame:
    """Normalize Caldera-derived attack information as hidden ground truth."""
    time_col = _column(attack_info, ("attack time", "time of attack", "timestamp", "time"), "attack time")
    pid_col = _column(attack_info, ("attack pid", "pid", "process id"), "attack PID")
    category_col = _column(attack_info, ("attack category", "category", "tactic", "tactic name"), "attack category/tactic")
    technique_col = _column(attack_info, ("technique id", "technique", "technique name", "mitre id"), "technique")
    raw_time = attack_info[time_col]
    attack_time = (
        pd.to_datetime(pd.to_numeric(raw_time, errors="raise"), unit="s", errors="raise", utc=True)
        if pd.api.types.is_numeric_dtype(raw_time)
        else pd.to_datetime(raw_time, errors="raise", utc=True)
    )
    result = pd.DataFrame({
        "attack_time": attack_time,
        "process_id": attack_info[pid_col].astype(str),
        "tactic": attack_info[category_col].astype(str).str.strip(),
        "technique": attack_info[technique_col].astype(str).str.strip(),
    }).sort_values("attack_time", kind="stable").reset_index(drop=True)
    result["campaign_step"] = range(1, len(result) + 1)
    result["policy_visible"] = False
    return result


def validate_provenance_graph(frame: pd.DataFrame) -> dict[str, int]:
    """Audit heterogeneous node/edge rows without zero-filling them into flows."""
    type_col = _column(frame, ("type", "entity type", "node type", "event type"), "entity type")
    id_col = _column(frame, ("id", "uuid", "node id"), "entity ID")
    kinds = frame[type_col].astype(str).str.casefold()
    process = kinds.str.contains("process", na=False)
    artifact = kinds.str.contains("artifact|file|directory|socket", regex=True, na=False)
    edge = kinds.str.contains("used|wasgeneratedby|wastriggeredby|wasderivedfrom|edge", regex=True, na=False)
    dangling = 0
    if edge.any():
        source_col = _column(frame, ("from", "source", "subject", "src"), "edge source")
        target_col = _column(frame, ("to", "target", "object", "dst"), "edge target")
        nodes = set(frame.loc[~edge, id_col].dropna().astype(str))
        dangling = int((~frame.loc[edge, source_col].astype(str).isin(nodes) | ~frame.loc[edge, target_col].astype(str).isin(nodes)).sum())
    return {"rows": len(frame), "process_nodes": int(process.sum()), "artifact_nodes": int(artifact.sum()),
            "edges": int(edge.sum()), "dangling_edges": dangling}


def build_multimodal_manifest(paths: CicaptPaths) -> dict[str, object]:
    """Hash official artifacts and declare a leakage-safe late-fusion contract."""
    missing = [str(path) for path in (paths.network, paths.provenance, paths.attack_info) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CICAPT-IIoT2024 artifacts: {missing}")
    def item(path: Path) -> dict[str, object]:
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
    return {
        "dataset": "CICAPT-IIoT2024", "official_source": "UNB Canadian Institute for Cybersecurity",
        "modalities": {"network": item(paths.network), "provenance": item(paths.provenance),
                       "attack_ground_truth": item(paths.attack_info)},
        "join_contract": {"strategy": "late_fusion", "keys": ["campaign_phase", "time_window", "process_id"],
                          "forbidden": ["zero-fill provenance into network rows", "policy access to tactic/technique/attack PID"]},
        "split_contract": "phase/run/campaign-step disjoint; chronological validation; held-out technique evaluation",
        "evidence_limit": "passive observations do not identify counterfactual intervention effects",
    }


def profile_network_csv(path: Path, *, chunksize: int = 250_000) -> dict[str, object]:
    """Stream a multi-GB network CSV and return a bounded schema/label profile."""
    if not path.exists() or chunksize < 1:
        raise FileNotFoundError(path) if not path.exists() else ValueError("chunksize must be positive")
    header = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    required = {"ts", "label", "subLabel", "subLabelCat"}
    if missing := required - set(header):
        raise ValueError(f"CICAPT network CSV lacks columns: {sorted(missing)}")
    counts = {name: {} for name in ("label", "subLabel", "subLabelCat")}
    rows = 0; minimum = float("inf"); maximum = float("-inf")
    for chunk in pd.read_csv(path, usecols=["ts", "label", "subLabel", "subLabelCat"], chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        times = pd.to_numeric(chunk["ts"], errors="coerce")
        if times.notna().any():
            minimum = min(minimum, float(times.min())); maximum = max(maximum, float(times.max()))
        for column in counts:
            for value, count in chunk[column].astype(str).value_counts(dropna=False).items():
                counts[column][value] = counts[column].get(value, 0) + int(count)
    return {"path": str(path), "bytes": path.stat().st_size, "rows": rows, "columns": len(header),
            "timestamp_min": minimum, "timestamp_max": maximum,
            "label_counts": counts["label"], "sublabel_counts": counts["subLabel"],
            "sublabel_category_counts": counts["subLabelCat"]}


DEFAULT_NETWORK_FEATURES = (
    "flow_duration", "Header_Length", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
    "max_duration", "min_duration", "average_duration", "std_duration",
    "TCP", "UDP", "ARP", "ICMP", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Variance", "Weight",
    "flow_idle_time", "flow_active_time",
)


def build_attack_preserving_sample(
    path: Path,
    *,
    benign_fraction: float = .01,
    seed: int = 42,
    chunksize: int = 250_000,
    feature_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep every attack row and sample benign background deterministically.

    Identity and ground-truth columns are retained only as explicit hidden
    evaluation fields; they are never returned as detector features.
    """
    if not 0 < benign_fraction <= 1:
        raise ValueError("benign_fraction must be in (0, 1]")
    header = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    features = list(feature_columns or DEFAULT_NETWORK_FEATURES)
    features = [column for column in features if column in header]
    required = ["ts", "label", "subLabel", "subLabelCat"]
    if missing := set(required) - set(header):
        raise ValueError(f"missing CICAPT sample columns: {sorted(missing)}")
    usecols = required + features
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    source_rows = source_attacks = kept_benign = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        labels = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int)
        attack = labels.ne(0)
        benign_keep = (~attack) & (rng.random(len(chunk)) < benign_fraction)
        selected = chunk.loc[attack | benign_keep].copy()
        selected["attack_present"] = labels.loc[selected.index].ne(0).to_numpy()
        selected["attack_tactic"] = selected.pop("subLabel").astype(str)
        selected["attack_technique"] = selected.pop("subLabelCat").astype(str)
        selected = selected.drop(columns="label")
        selected["timestamp"] = pd.to_numeric(selected.pop("ts"), errors="raise")
        selected["source_day"] = pd.to_datetime(selected.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        parts.append(selected)
        source_rows += len(chunk); source_attacks += int(attack.sum()); kept_benign += int(benign_keep.sum())
    result = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    result["group_id"] = result.source_day + "|hour=" + pd.to_datetime(result.timestamp, unit="s", utc=True).dt.hour.astype(str)
    manifest = {"source": str(path), "source_rows": source_rows, "source_attack_rows": source_attacks,
                "sample_rows": len(result), "sample_attack_rows": int(result.attack_present.sum()),
                "sample_benign_rows": kept_benign, "benign_fraction": benign_fraction,
                "feature_columns": features, "forbidden_detector_columns": ["attack_present", "attack_tactic", "attack_technique", "timestamp", "source_day", "group_id"]}
    return result, manifest
