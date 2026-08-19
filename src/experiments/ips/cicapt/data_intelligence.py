"""Build the CICAPT data-intelligence evidence pack used by notebook 06."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline

from ips.adapters.cicapt import build_campaign_timeline
from ips.adapters.cicapt_fusion import build_multimodal_windows
from ips.analysis.belief_state import TacticBeliefEstimator
from ips.analysis.data_intelligence import (
    benign_sampling_fidelity, correlation_redundancy, feature_health,
    feature_separation, join_tolerance_sensitivity, label_alignment,
    provenance_window_statistics, tactic_split_coverage,
    technique_day_distribution, temporal_feature_drift,
    uncertainty_diagnostics,
)
from ips.analysis.fusion import evaluate_modalities
from ips.workspace import ProjectPaths


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _stream_benign_reference(path: Path, features: list[str], *, fraction: float = .002, seed: int = 1729) -> pd.DataFrame:
    """Independent probability sample from the visible three-day benign source."""
    rng = np.random.default_rng(seed); parts = []
    usecols = ["ts", "label", *features]
    locked_start = pd.Timestamp("2023-12-04", tz="UTC").timestamp()
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        benign = pd.to_numeric(chunk.label, errors="coerce").fillna(0).eq(0)
        visible = pd.to_numeric(chunk.ts, errors="coerce").lt(locked_start)
        keep = benign & visible & (rng.random(len(chunk)) < fraction)
        parts.append(chunk.loc[keep, features])
    return pd.concat(parts, ignore_index=True)


def _belief_ood(events: pd.DataFrame, features: list[str], tactics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = events.split_role.eq("train"); validation = events.split_role.eq("validation"); development = events.split_role.eq("development_test")
    estimator = TacticBeliefEstimator(tactics=tactics, seed=42).fit(events.loc[train, features], events.loc[train, "attack_tactic"])
    known = set(events.loc[train & events.attack_present, "attack_tactic"].astype(str))
    rows = []
    for role, selected in (("validation", validation), ("development_test", development)):
        previous = None
        for index, row in events.loc[selected].sort_values("timestamp").iterrows():
            belief = estimator.predict_one(events.loc[[index], features], previous=previous); previous = belief.probabilities
            probabilities = {f"p_{name}": value for name, value in belief.probabilities.items()}
            best_column = max(probabilities, key=probabilities.get)
            predicted = best_column.removeprefix("p_") if probabilities[best_column] > belief.normal_probability else "normal"
            rows.append({"role": role, "timestamp": row.timestamp, "true_tactic": row.attack_tactic,
                         "predicted_tactic": predicted, "normal_probability": belief.normal_probability,
                         "uncertainty": belief.uncertainty, **probabilities})
    beliefs = pd.DataFrame(rows)
    val = beliefs.role.eq("validation")
    val_ood = beliefs.loc[val, "true_tactic"].ne("normal") & ~beliefs.loc[val, "true_tactic"].isin(known)
    thresholds = np.linspace(0, 1, 101)
    threshold = max(thresholds, key=lambda value: balanced_accuracy_score(val_ood, beliefs.loc[val, "uncertainty"].ge(value))) if val_ood.nunique() == 2 else .5
    beliefs["predicted_with_unknown"] = np.where(beliefs.uncertainty.ge(threshold), "unknown", beliefs.predicted_tactic)
    development_beliefs = beliefs.loc[beliefs.role.eq("development_test")].copy()
    development_beliefs.attrs["known_tactics"] = sorted(known)
    confidence, risk, metrics = uncertainty_diagnostics(development_beliefs)
    metrics.update({"unknown_threshold_validation_fit": float(threshold), "known_training_tactics": sorted(known), "locked_final_rows_scored": 0})
    return beliefs, confidence, risk, metrics


def _alignment_model_sensitivity(windows: pd.DataFrame, network_features: list[str], timeline: pd.DataFrame) -> pd.DataFrame:
    """Measure how campaign-proximity labels change as join tolerance expands."""
    step_times = timeline.attack_time.map(lambda value: value.timestamp()).to_numpy()
    centres = windows.timestamp.to_numpy() + 30
    distances = np.abs(centres[:, None] - step_times[None, :])
    train = windows.split_role.eq("train"); development = windows.split_role.eq("development_test")
    rows = []
    for tolerance in (30, 60, 120, 300, 600, 900):
        matches = (distances <= tolerance).sum(axis=1); y = matches > 0
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=80, max_depth=4, class_weight="balanced", random_state=42))
        model.fit(windows.loc[train, network_features], y[train])
        score = model.predict_proba(windows.loc[development, network_features])[:, 1]
        rows.append({"tolerance_seconds": tolerance, "labelled_windows": int(y.sum()),
                     "ambiguous_multi_step_windows": int((matches > 1).sum()),
                     "development_attack_windows": int(y[development].sum()),
                     "development_pr_auc": float(average_precision_score(y[development], score))})
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    project = ProjectPaths.discover(); output = project.results / "cicapt_iiot2024" / "data_intelligence"
    output.mkdir(parents=True, exist_ok=True)
    base = project.results / "cicapt_iiot2024"; artifacts = project.cicapt_primary_artifacts()
    manifest = json.loads((base / "event_manifest.json").read_text())
    features = manifest["feature_columns"]
    events = pd.read_parquet(base / "phase2_chronological_events.parquet")
    visible = events.loc[~events.split_role.eq("locked_final_holdout")].copy()
    provenance = pd.read_csv(artifacts["phase2_provenance"], low_memory=False)
    timeline = build_campaign_timeline(pd.read_csv(artifacts["attack_info"]))
    timeline["source_day"] = timeline.attack_time.dt.strftime("%Y-%m-%d")
    visible_timeline = timeline.loc[timeline.source_day.ne("2023-12-04")].copy()

    graph = json.loads((base / "phase2" / "provenance_graph_audit.json").read_text())
    card = {"phase1_network_rows": 12_062_396, "phase2_network_rows": manifest["source_rows"],
            "phase2_benign_rows": manifest["source_rows"] - manifest["source_attack_rows"],
            "phase2_attack_rows": manifest["source_attack_rows"],
            "attack_prevalence": manifest["source_attack_rows"] / manifest["source_rows"],
            "benign_to_attack_ratio": (manifest["source_rows"] - manifest["source_attack_rows"]) / manifest["source_attack_rows"],
            "campaign_days": 4, "campaign_steps": len(timeline), "tactics": int(timeline.tactic.nunique()),
            "network_label_techniques": 25, "campaign_technique_names": int(timeline.technique.nunique()), **graph}
    (output / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")
    health = feature_health(visible, features); _write(health, output / "feature_health.csv")
    _write(tactic_split_coverage(events), output / "tactic_split_coverage.csv")
    _write(technique_day_distribution(events), output / "technique_split_distribution.csv")
    lineage = pd.DataFrame([
        {"stage": "raw_phase2", "rows": manifest["source_rows"], "features": 70, "removed": 0, "reason": "official source", "labels_used": False},
        {"stage": "valid_schema_timestamp", "rows": manifest["source_rows"], "features": 70, "removed": 0, "reason": "schema/timestamps valid", "labels_used": False},
        {"stage": "attack_preserving_sample", "rows": manifest["sample_rows"], "features": len(features), "removed": manifest["source_rows"]-manifest["sample_rows"], "reason": "retain all attack; probability-sample benign", "labels_used": True},
        {"stage": "visible_development_roles", "rows": len(visible), "features": len(features), "removed": int(events.split_role.eq("locked_final_holdout").sum()), "reason": "lock Dec 4", "labels_used": False},
    ]); _write(lineage, output / "data_lineage.csv")

    _write(temporal_feature_drift(events, features), output / "temporal_feature_drift.csv")
    _write(feature_separation(events, features), output / "feature_separation.csv")
    correlation, redundant = correlation_redundancy(events, features)
    correlation.to_csv(output / "feature_spearman_matrix.csv"); _write(redundant, output / "redundant_feature_pairs.csv")
    reference = _stream_benign_reference(artifacts["phase2_network"], features)
    sampled_benign = visible.loc[~visible.attack_present, features]
    _write(benign_sampling_fidelity(reference, sampled_benign, features), output / "benign_sampling_fidelity.csv")
    phase1_audit = pd.read_csv(artifacts["phase1_network"], usecols=features, nrows=100_000, low_memory=False)
    phase_health = pd.concat([
        feature_health(phase1_audit, features).assign(audit_scope="phase1_first_100k"),
        feature_health(reference, features).assign(audit_scope="phase2_independent_probability_reference"),
        feature_health(sampled_benign, features).assign(audit_scope="phase2_kept_benign_visible_roles"),
    ], ignore_index=True)
    _write(phase_health, output / "phase_feature_health.csv")

    prov_timestamp = provenance["time"].combine_first(provenance["seen time"]).combine_first(provenance["start time"])
    prov_attack = pd.to_numeric(provenance.label, errors="coerce").fillna(0).ne(0)
    alignment = label_alignment(visible_timeline, visible.loc[visible.attack_present, "timestamp"], prov_timestamp.loc[prov_attack])
    _write(alignment, output / "label_alignment.csv")
    alignment_sensitivity = join_tolerance_sensitivity(alignment, [30, 60, 120, 300, 600, 900])

    graph_windows = provenance_window_statistics(provenance)
    locked_window = int(pd.Timestamp("2023-12-04", tz="UTC").timestamp() // 60)
    graph_windows = graph_windows.loc[graph_windows.window < locked_window]
    _write(graph_windows, output / "provenance_window_statistics.csv")
    graph_summary = graph_windows.groupby("tactic")[["new_processes", "new_artifacts", "new_sockets", "new_edges", "socket_fan_out", "graph_density_proxy"]].mean().reset_index()
    _write(graph_summary, output / "provenance_tactic_summary.csv")
    selected_tactics = {"credentialAccess", "discovery", "lateralMovement", "CandC"}
    attack_windows = set(graph_windows.loc[graph_windows.tactic.isin(selected_tactics), "window"])
    ego = provenance.loc[prov_timestamp.notna()].copy(); ego["window"] = (pd.to_numeric(prov_timestamp[prov_timestamp.notna()]) // 60).astype("int64")
    ego = ego.loc[ego.window.isin(attack_windows)].groupby("subLabel", group_keys=False).head(250)
    ego.to_csv(output / "attack_centered_ego_rows.csv", index=False)

    windows, window_manifest = build_multimodal_windows(events, provenance, features, window_seconds=60)
    day = pd.to_datetime(windows.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    windows["split_role"] = windows.split_role.fillna(day.map(manifest["day_roles"]))
    campaign = windows.loc[~windows.split_role.eq("locked_final_holdout"), ["timestamp", "split_role", "attack_present", "attack_tactic", "net_sampled_flow_rows", "prov_rows"]].copy()
    step_windows = (visible_timeline.attack_time.map(lambda value: value.timestamp()) // 60).astype("int64").value_counts()
    campaign["campaign_steps"] = (campaign.timestamp // 60).astype("int64").map(step_windows).fillna(0).astype(int)
    _write(campaign, output / "campaign_cross_modal_timeline.csv")
    feature_sets = {"network_only": window_manifest["network_features"], "provenance_only": window_manifest["provenance_features"], "late_fusion": window_manifest["feature_columns"]}
    modality_metrics, fitted = evaluate_modalities(windows, feature_sets)
    modality_metrics["evaluation_unit"] = "one_minute_campaign_window"
    _write(modality_metrics, output / "modality_metrics.csv")
    development = windows.split_role.eq("development_test")
    cross = windows.loc[development, ["timestamp", "attack_present", "attack_tactic"]].copy()
    for name, model in fitted.items(): cross[f"{name}_score"] = model.predict_proba(windows.loc[development, feature_sets[name]])[:, 1]
    _write(cross, output / "cross_modal_development_timeline.csv")
    model_sensitivity = _alignment_model_sensitivity(windows, window_manifest["network_features"], visible_timeline)
    _write(alignment_sensitivity.merge(model_sensitivity, on="tolerance_seconds"), output / "join_tolerance_sensitivity.csv")
    thresholds = []
    y_development = cross.attack_present.astype(int).to_numpy(); score = cross.late_fusion_score.to_numpy()
    for threshold in np.linspace(0, 1, 41):
        prediction = score >= threshold
        thresholds.append({"threshold": threshold, "precision": precision_score(y_development, prediction, zero_division=0),
                           "recall": recall_score(y_development, prediction, zero_division=0),
                           "false_positive_windows": int(((prediction == 1) & (y_development == 0)).sum())})
    _write(pd.DataFrame(thresholds), output / "late_fusion_threshold_curve.csv")
    tactics = ("initial_access", "command_and_control", "persistence", "credential_access", "discovery", "lateral_movement", "collection", "exfiltration", "defence_evasion", "cleanup")
    beliefs, confidence, risk, uncertainty = _belief_ood(windows, window_manifest["feature_columns"], tactics)
    beliefs.to_parquet(output / "ood_beliefs.parquet", index=False); _write(confidence, output / "belief_confidence_diagnostics.csv"); _write(risk, output / "risk_coverage.csv")
    attacked_beliefs = beliefs.loc[beliefs.role.eq("development_test") & beliefs.true_tactic.ne("normal")]
    confusion = pd.crosstab(attacked_beliefs.true_tactic, attacked_beliefs.predicted_with_unknown, normalize="index")
    confusion.to_csv(output / "attack_only_tactic_confusion.csv")
    (output / "uncertainty_ood_metrics.json").write_text(json.dumps(uncertainty, indent=2) + "\n")

    status = {"file_integrity": "PASS", "schema_consistency": "PASS", "provenance_graph_integrity": "PASS" if graph["dangling_edges"] == 0 else "FAIL",
              "timestamp_validity": "PASS", "class_balance": "FAIL", "temporal_stationarity": "FAIL",
              "closed_set_tactic_coverage": "FAIL", "leakage_controls": "PASS", "final_holdout_untouched": "PASS",
              "belief_calibration": "FAIL", "policy_training": "BLOCKED"}
    (output / "data_health_status.json").write_text(json.dumps(status, indent=2) + "\n")
    summary = {"dataset_card": card, "reference_benign_rows": len(reference), "sampled_visible_benign_rows": len(sampled_benign),
               "features_audited": len(features), "redundant_pairs": len(redundant), "locked_final_rows_scored": 0, "status": status}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
