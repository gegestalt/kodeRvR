"""Generate the IPS scientific claim-control report used by the full notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
from ips.belief import observable_belief
from ips.analysis.controls import (
    InterventionContext,
    calibration_metrics,
    dataset_provenance_probe,
    dataset_registry,
    eligibility_audit,
    feature_semantics_registry,
    generalization_manifests,
    intervention_utility,
    observability_registry,
    operational_rates,
    source_kind_policy,
)
from ips.adapters.nsl_kdd import NslIpsConfig, build_nsl_ips_evidence


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "notebook_ips_lab" / "claim_control"
CSE_EVENTS = ROOT / "data" / "ips_events" / "cse_cic_ids2018_temporal_events.parquet"


def _policy_representation(events: pd.DataFrame, dataset: str, limit: int, seed: int) -> pd.DataFrame:
    sampled = events.sample(min(limit, len(events)), random_state=seed).copy()
    beliefs = [
        observable_belief(float(row.threat_probability), float(row.anomaly_score), [float(row.threat_probability)])
        for row in sampled.itertuples()
    ]
    sampled["estimated_stage"] = [item[0] for item in beliefs]
    sampled["estimated_compromise"] = [item[1] for item in beliefs]
    sampled["recent_score_rate"] = [item[2] for item in beliefs]
    sampled["source_dataset"] = dataset
    return sampled


def run() -> dict[str, object]:
    if not CSE_EVENTS.exists():
        raise FileNotFoundError("run python -m experiments.ips.cse.build_events before this report")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = dataset_registry()
    registry.assign(
        generalization_axis=registry.generalization_axis.map(json.dumps),
        task_types=registry.task_types.map(json.dumps),
        group_keys=registry.group_keys.map(json.dumps),
        leakage_keys=registry.leakage_keys.map(json.dumps),
        feature_leakage_candidates=registry.feature_leakage_candidates.map(json.dumps),
    ).to_csv(OUTPUT / "dataset_evidence_registry.csv", index=False)
    observability_registry().to_csv(OUTPUT / "state_observability_audit.csv", index=False)
    source_kind_policy().to_csv(OUTPUT / "source_kind_policy.csv", index=False)
    feature_semantics_registry().to_csv(OUTPUT / "feature_semantics_registry.csv", index=False)

    cse = pd.read_parquet(CSE_EVENTS)
    cse_calibration = []
    for role, rows in cse.groupby("split_role"):
        metrics = calibration_metrics(rows.attack_present.astype(int).to_numpy(), rows.threat_probability.to_numpy())
        cse_calibration.append({"dataset": "CSE-CIC-IDS2018", "split_role": role, "rows": len(rows), **metrics})
    calibration_frame = pd.DataFrame(cse_calibration)
    calibration_frame.to_csv(OUTPUT / "calibration_by_temporal_role.csv", index=False)

    disruptive = cse.threat_probability.ge(.70).to_numpy()
    operational = operational_rates(
        attack_present=cse.attack_present.to_numpy(), disruptive_action=disruptive,
        timestamps=cse.timestamp.to_numpy(),
    )
    pd.DataFrame([{"dataset": "CSE-CIC-IDS2018", "decision_rule": "p_attack>=0.70 diagnostic", **operational}]).to_csv(OUTPUT / "operational_rates.csv", index=False)

    train = cse[cse.split_role.eq("train")]
    test = cse[cse.split_role.eq("final_test")]
    eligibility = eligibility_audit(
        train, test, group_keys=["source_day", "group_id"],
        feature_columns=["threat_probability", "anomaly_score", "critical_service"],
        feature_leakage_candidates=["source_day", "group_id", "timestamp", "attack_family", "attack_present"],
    )
    (OUTPUT / "cse_eligibility.json").write_text(json.dumps({
        "eligible": eligibility.eligible,
        "group_leakage": list(eligibility.group_leakage),
        "forbidden_features": list(eligibility.forbidden_features),
        "reasons": list(eligibility.reasons),
    }, indent=2) + "\n", encoding="utf-8")

    nsl = build_nsl_ips_evidence(
        D.load_nsl_kdd("train"), D.load_nsl_kdd("test"),
        NslIpsConfig(max_train_rows=6_000, max_test_rows=3_000, episode_size=12, folds=3),
    )
    cse_repr = _policy_representation(cse, "CSE-CIC-IDS2018", 3_000, 42)
    nsl_repr = _policy_representation(nsl.test_events, "NSL-KDD", 3_000, 43)
    combined = pd.concat([cse_repr, nsl_repr], ignore_index=True)
    representation_columns = [
        "threat_probability", "anomaly_score", "estimated_stage",
        "estimated_compromise", "recent_score_rate", "critical_service",
    ]
    provenance = dataset_provenance_probe(
        combined[representation_columns], combined.source_dataset,
        combined.group_id.astype(str), folds=5, seed=42,
    )
    pd.DataFrame([{"representation": "policy-visible evidence vector", **provenance}]).to_csv(OUTPUT / "dataset_provenance_probe.csv", index=False)

    manifests = []
    for item in generalization_manifests(combined, holdout="source_dataset"):
        manifests.append({
            "protocol": item.protocol, "holdout": item.holdout_value,
            "train_rows": len(item.train_indices), "test_rows": len(item.test_indices),
            "status": "MANIFEST_ONLY_DIAGNOSTIC",
            "reason": "NSL has ordered-row proxy timing and only policy-evidence features are harmonized",
        })
    for item in generalization_manifests(cse, holdout="attack_family"):
        manifests.append({
            "protocol": item.protocol, "holdout": item.holdout_value,
            "train_rows": len(item.train_indices), "test_rows": len(item.test_indices),
            "status": "BLOCKED_DETECTOR_RETRAIN_REQUIRED",
            "reason": "OOF detector currently saw held-out family labels during detector development",
        })
    pd.DataFrame(manifests).to_csv(OUTPUT / "generalization_protocol_manifests.csv", index=False)

    contexts = {
        "single_confident_identity": InterventionContext(.5, .95, .05, 1, .1, .2, .1, .2),
        "shared_nat_critical_service": InterventionContext(1.0, .4, .95, 500, .9, 1.0, .8, 1.0),
    }
    utility_rows = []
    from ips.actions import IpsAction
    for name, context in contexts.items():
        for action in IpsAction:
            utility_rows.append({"context": name, "action": action.name, **intervention_utility(action, .95, context)})
    pd.DataFrame(utility_rows).to_csv(OUTPUT / "blast_radius_utility.csv", index=False)

    factorial = pd.MultiIndex.from_product(
        [
            ["HistGradientBoosting", "benign-only IsolationForest", "temporal/SSL future"],
            ["none", "Platt", "isotonic"],
            ["rule", "optimal_stopping", "LinUCB", "DQN", "constrained_PPO"],
            ["in_domain", "future_time", "unknown_family", "new_dataset", "adversarial"],
        ],
        names=["detector", "calibration", "policy", "distribution"],
    ).to_frame(index=False)
    factorial["status"] = np.where(
        factorial.detector.eq("HistGradientBoosting")
        & factorial.calibration.isin(["none", "Platt", "isotonic"])
        & factorial.distribution.isin(["in_domain", "future_time"]),
        "CURRENTLY_ELIGIBLE_OR_PARTIAL", "PLANNED_REQUIRES_NEW_EVIDENCE",
    )
    factorial.to_csv(OUTPUT / "detector_calibration_policy_distribution_grid.csv", index=False)
    pd.DataFrame([
        ("random_label", "performance must collapse to chance", "detect split/preprocessing leakage"),
        ("dataset_identity_only", "must not predict attack reliably", "detect source shortcut"),
        ("forbidden_identifier_only", "must fail eligibility gate", "feature leakage sentinel"),
        ("shuffled_time", "must not outperform true future holdout", "temporal leakage sentinel"),
        ("allow_only_policy", "zero prevention benefit", "policy necessity baseline"),
        ("contextual_bandit", "tie implies sequential RL not justified", "delayed-credit baseline"),
    ], columns=["negative_control", "expected_behavior", "purpose"]).to_csv(OUTPUT / "negative_controls.csv", index=False)
    pd.DataFrame([
        ("input", "P(X)", "feature PSI/KS/Wasserstein", "PARTIAL"),
        ("representation", "P(Z)", "embedding MMD/source probe", "SOURCE PROBE IMPLEMENTED"),
        ("score", "P(prediction)", "score PSI/KS", "IMPLEMENTED EARLIER"),
        ("calibration", "P(Y|score)", "ECE/Brier/log-loss by time", "IMPLEMENTED"),
        ("performance", "FPR/FNR/precision", "rolling labelled outcomes", "PARTIAL"),
        ("policy", "P(action|belief)", "action distribution drift", "IMPLEMENTED EARLIER"),
        ("outcome", "P(success|belief,action)", "intervention success drift", "BLOCKED OBSERVED OUTCOMES"),
    ], columns=["drift_layer", "distribution", "metric", "status"]).to_csv(OUTPUT / "drift_layers.csv", index=False)
    pd.DataFrame([
        ("transition_id", "string", True), ("timestamp_issued", "UTC datetime", True),
        ("telemetry_window_id", "string", True), ("belief_state", "versioned vector", True),
        ("proposed_action", "enum", True), ("executed_action", "enum", True),
        ("behavior_probability_vector", "float[action]", True), ("action_scope", "entity/scope", True),
        ("shield_reason", "string/null", True), ("attack_continued", "bool", True),
        ("attack_succeeded", "bool", True), ("connection_terminated", "bool", True),
        ("service_disrupted", "bool", True), ("legitimate_sessions_affected", "integer", True),
        ("affected_identities", "integer", True), ("recovery_required", "bool", True),
        ("time_to_containment_ms", "float/null", True), ("rollback_count", "integer", True),
        ("outcome_observation_horizon_s", "float", True), ("evidence_kind", "observed_shadow_deployment", True),
    ], columns=["field", "type", "required"]).to_csv(OUTPUT / "intervention_log_contract.csv", index=False)

    status = {
        "research_frame": "Safe Adaptive Intrusion Prevention Under Distribution Shift",
        "rq1_generalization": "PARTIAL: CSE future-day evidence; full raw-feature LODO blocked",
        "rq2_sequential_control": "PARTIAL: simulator/counterfactual outcomes only",
        "rq3_safe_uncertainty": "IMPLEMENTED CONTRACTS: abstention, shield, blast radius; external OOD validation pending",
        "rq4_causal_prevention": "BLOCKED: no observed intervention transitions",
        "pomdp_observability": "IMPLEMENTED: policy receives detector-history beliefs, hidden labels remain reward-only",
        "constrained_mdp": "CONTRACT IMPLEMENTED; empirical audit blocked until per-transition collateral/outage logs exist",
        "realisable_traffic_evasion": "BLOCKED: packet mutation + re-extraction pipeline not yet available",
        "cyber_range": "BLOCKED: target network/enforcement/telemetry infrastructure not configured",
    }
    (OUTPUT / "claim_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {**status, "dataset_identifiability_score": provenance["balanced_accuracy"]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
