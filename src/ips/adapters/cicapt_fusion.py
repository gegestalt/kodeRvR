"""Window-aligned network/provenance late-fusion adapter for CICAPT."""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE_MAP = {
    "process":"process", "artifact":"artifact", "used":"used",
    "wasgeneratedby":"generated_by", "wastriggeredby":"triggered_by",
    "wasderivedfrom":"derived_from",
}


def _normalize_tactic(value: object) -> str:
    text = str(value).strip().casefold().replace(" ", "").replace("_", "")
    aliases = {"0":"normal","nan":"normal","candc":"command_and_control","commandandcontrol":"command_and_control",
               "credentialaccess":"credential_access","defenceevasion":"defence_evasion",
               "lateralmovement":"lateral_movement","initialaccess":"initial_access"}
    return aliases.get(text, text)


def build_multimodal_windows(
    network: pd.DataFrame,
    provenance: pd.DataFrame,
    network_features: list[str],
    *,
    window_seconds: int = 300,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate modalities independently, then join on a common time window."""
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    required_network={"timestamp","attack_present","attack_tactic","split_role",*network_features}
    if missing:=required_network-set(network): raise ValueError(f"missing network columns: {sorted(missing)}")
    required_prov={"type","label","subLabel","time","seen time","start time"}
    if missing:=required_prov-set(provenance): raise ValueError(f"missing provenance columns: {sorted(missing)}")
    net=network.copy(); net["window"]=(pd.to_numeric(net.timestamp,errors="raise")//window_seconds).astype("int64")
    aggregations={feature:["mean","std","max"] for feature in network_features}
    net_features=net.groupby("window").agg(aggregations)
    net_features.columns=[f"net_{feature}_{stat}" for feature,stat in net_features.columns]
    net_truth=net.groupby("window").agg(attack_present=("attack_present","max"),
        attack_tactic=("attack_tactic",lambda values: next((_normalize_tactic(v) for v in values if _normalize_tactic(v)!="normal"),"normal")),
        split_role=("split_role","first"))

    prov=provenance.copy()
    timestamp=prov["time"].combine_first(prov["seen time"]).combine_first(prov["start time"])
    prov=prov.loc[timestamp.notna()].copy(); prov["window"]=(pd.to_numeric(timestamp[timestamp.notna()],errors="raise")//window_seconds).astype("int64")
    kinds=prov["type"].astype(str).str.casefold().map(TYPE_MAP).fillna("other")
    type_counts=pd.crosstab(prov.window,kinds).add_prefix("prov_type_")
    prov_features=pd.DataFrame(index=type_counts.index).join(type_counts)
    prov_features["prov_unique_pid"]=prov.groupby("window")["pid"].nunique() if "pid" in prov else 0
    prov_features["prov_unique_exe"]=prov.groupby("window")["exe"].nunique() if "exe" in prov else 0
    prov_features["prov_unique_operation"]=prov.groupby("window")["operation"].nunique() if "operation" in prov else 0
    prov_features["prov_rows"]=prov.groupby("window").size()
    prov_truth=prov.groupby("window").agg(prov_attack=("label","max"),
        prov_tactic=("subLabel",lambda values: next((_normalize_tactic(v) for v in values if _normalize_tactic(v)!="normal"),"normal")))

    windows=net_features.join(prov_features,how="outer").join(net_truth,how="left").join(prov_truth,how="left")
    windows["attack_present"]=windows.attack_present.fillna(False)|windows.prov_attack.fillna(0).astype(bool)
    windows["attack_tactic"]=np.where(windows.attack_tactic.fillna("normal").eq("normal"),windows.prov_tactic.fillna("normal"),windows.attack_tactic)
    windows["timestamp"]=windows.index.to_numpy(dtype=float)*window_seconds
    windows=windows.drop(columns=["prov_attack","prov_tactic"]).reset_index(drop=True)
    feature_columns=[column for column in windows if column.startswith(("net_","prov_"))]
    windows[feature_columns]=windows[feature_columns].replace([np.inf,-np.inf],np.nan).fillna(0)
    manifest={"window_seconds":window_seconds,"rows":len(windows),"feature_columns":feature_columns,
              "network_features":[x for x in feature_columns if x.startswith("net_")],
              "provenance_features":[x for x in feature_columns if x.startswith("prov_")],
              "hidden_columns":["attack_present","attack_tactic","split_role","timestamp"]}
    return windows.sort_values("timestamp").reset_index(drop=True),manifest
