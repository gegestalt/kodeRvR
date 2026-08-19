"""Claim-control framework for safe adaptive IPS experiments.

This module separates dataset availability from scientific eligibility.  It
provides executable gates for provenance, leakage, generalization, calibration,
operational burden, abstention, observability, and safety-shield behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from ips.actions import IpsAction


@dataclass(frozen=True)
class EligibilityAudit:
    eligible: bool
    group_leakage: tuple[str, ...]
    forbidden_features: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeneralizationManifest:
    protocol: str
    holdout_value: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_values: tuple[str, ...]
    test_values: tuple[str, ...]


@dataclass(frozen=True)
class ResponseDecision:
    disposition: str
    executed_action: IpsAction
    reason: str


@dataclass(frozen=True)
class InterventionContext:
    asset_criticality: float
    identity_confidence: float
    shared_ip_probability: float
    estimated_users_behind_identity: int
    service_dependency_score: float
    action_blast_radius: float
    rollback_cost: float
    business_impact: float


@dataclass(frozen=True)
class ConstraintBudget:
    collateral_budget: float
    critical_outage_probability: float
    false_block_rate: float
    latency_p95_ms: float


def dataset_registry() -> pd.DataFrame:
    """Return the evidence registry; list fields are intentional audit fields."""
    common = {
        "dataset_version": "verify at acquisition",
        "sha256_manifest": "required before eligibility",
        "label_mapping_version": "ips-family-v1",
        "allowed_train_use": "only after eligibility gate",
        "allowed_test_use": "only under declared generalization protocol",
    }
    rows = [
        dict(dataset="CSE-CIC-IDS2018", official_source="UNB CIC/AWS", raw_or_processed="official/processed", scientific_role="active enterprise temporal evidence", generalization_axis=["time", "scenario", "attack_family"], task_types=["binary_detection", "open_set_detection"], group_keys=["source_day", "hour_group"], leakage_keys=["source_day", "capture_file", "timestamp_window"], feature_leakage_candidates=["src_ip", "dst_ip", "flow_id", "timestamp", "capture_filename"], raw_format="daily CICFlowMeter CSV", feature_extractor="CICFlowMeter-V3", feature_extractor_version="UNB release", flow_timeout="release-defined", modality="network flow", feature_space="CSE flow features", required_split="complete-day chronological", attack_value="BruteForce and DoS in active subset", unknown_attack_possible=True, risk_or_gap="processed features and counterfactual IPS outcomes", quarantine_status="eligible subset after audit", ingestion_priority=0, evidence_priority=1, next_adapter="add later days without retuning final test"),
        dict(dataset="CICIoT2023", official_source="UNB CIC", raw_or_processed="official/processed", scientific_role="large-scale IoT detector development", generalization_axis=["device", "capture", "time", "attack_family"], task_types=["binary_detection", "closed_set_multiclass", "open_set_detection"], group_keys=["capture_id", "device_id", "attack_run_id"], leakage_keys=["capture_id", "device_id", "scenario_id", "timestamp_window"], feature_leakage_candidates=["src_ip", "dst_ip", "MAC", "device_id", "capture_filename"], raw_format="PCAP + CSV", feature_extractor="release extractor", feature_extractor_version="verify", flow_timeout="verify", modality="network", feature_space="CICIoT tabular", required_split="capture/device/time-group disjoint", attack_value="33 attacks / seven categories", unknown_attack_possible=True, risk_or_gap="mirrors may expose random row splits", quarantine_status="audit-required", ingestion_priority=1, evidence_priority=2, next_adapter="official provenance + capture/device manifest"),
        dict(dataset="TON_IoT", official_source="UNSW Canberra", raw_or_processed="official/processed", scientific_role="cross-domain and telemetry shift", generalization_axis=["source", "time", "network", "modality"], task_types=["binary_detection", "ood_detection"], group_keys=["source", "scenario", "time_window"], leakage_keys=["source", "scenario", "host", "time_window"], feature_leakage_candidates=["src_ip", "dst_ip", "host_id", "timestamp", "source_file"], raw_format="network + telemetry CSV", feature_extractor="modality-specific", feature_extractor_version="verify", flow_timeout="verify", modality="multi-modal", feature_space="modality-isolated", required_split="source/time/scenario disjoint", attack_value="heterogeneous IoT/IIoT attacks", unknown_attack_possible=True, risk_or_gap="modalities must not be silently pooled", quarantine_status="modality isolation required", ingestion_priority=2, evidence_priority=1, next_adapter="network-flow modality first"),
        dict(dataset="IoT-23", official_source="Stratosphere Laboratory", raw_or_processed="raw capture re-derived", scientific_role="real-malware external validation", generalization_axis=["capture", "scenario", "malware_family"], task_types=["binary_detection", "open_set_detection", "ood_detection"], group_keys=["capture_id", "scenario_id"], leakage_keys=["capture_id", "scenario_id", "infected_host"], feature_leakage_candidates=["src_ip", "dst_ip", "capture_id", "scenario_id"], raw_format="PCAP + Zeek conn.log", feature_extractor="Zeek", feature_extractor_version="manifest required", flow_timeout="Zeek default/versioned", modality="Zeek network flows", feature_space="Zeek conn", required_split="entire capture/scenario disjoint", attack_value="C&C, DDoS, scan, malware behavior", unknown_attack_possible=True, risk_or_gap="extreme imbalance", quarantine_status="none after audit", ingestion_priority=3, evidence_priority=1, next_adapter="labelled conn.log parser"),
        dict(dataset="CICAPT-IIoT", official_source="UNB CIC", raw_or_processed="official/raw+processed", scientific_role="multi-stage APT/provenance validation", generalization_axis=["campaign", "scenario", "technique"], task_types=["open_set_detection", "ood_detection"], group_keys=["campaign", "run", "phase"], leakage_keys=["campaign", "run", "process_id", "attack_time"], feature_leakage_candidates=["node_id", "process_id", "attack_script_id", "scenario_id"], raw_format="provenance graph + PCAP", feature_extractor="graph/network separate", feature_extractor_version="verify", flow_timeout="not shared across modalities", modality="provenance graph", feature_space="nodes/edges; never zero-filled into flow table", required_split="campaign/run disjoint", attack_value=">20 techniques / eight tactics", unknown_attack_possible=True, risk_or_gap="not directly comparable with flow tables", quarantine_status="separate experimental track", ingestion_priority=4, evidence_priority=1, next_adapter="provenance-graph track"),
        dict(dataset="X-IIoTID", official_source="authors / IEEE IoT Journal", raw_or_processed="official/processed", scientific_role="device/connectivity-independent IIoT validation", generalization_axis=["device", "connectivity", "scenario"], task_types=["binary_detection", "ood_detection"], group_keys=["device", "scenario", "source_group"], leakage_keys=["device", "scenario", "source_group"], feature_leakage_candidates=["device_id", "IP", "timestamp", "scenario_id"], raw_format="IIoT telemetry table", feature_extractor="authors", feature_extractor_version="verify", flow_timeout="verify", modality="IIoT security telemetry", feature_space="network-observable subset first", required_split="device/scenario/source-group disjoint", attack_value="heterogeneous IIoT behavior", unknown_attack_possible=True, risk_or_gap="identifier and environment-feature audit required", quarantine_status="identifier audit required", ingestion_priority=5, evidence_priority=2, next_adapter="schema + network-observable subset"),
        dict(dataset="MQTT-IoT-IDS2020", official_source="University of Strathclyde / IEEE DataPort", raw_or_processed="official/raw+processed", scientific_role="protocol-specific external generalization", generalization_axis=["protocol", "capture", "scenario"], task_types=["binary_detection", "open_set_detection"], group_keys=["capture", "scenario", "time_window"], leakage_keys=["capture", "scenario", "time_window"], feature_leakage_candidates=["IP", "capture_filename", "scenario_id"], raw_format="PCAP + packet/uniflow/biflow CSV", feature_extractor="authors at three abstraction levels", feature_extractor_version="verify", flow_timeout="verify", modality="MQTT network", feature_space="MQTT + common-feature projection", required_split="capture/scenario/time disjoint", attack_value="MQTT brute force, scans, SSH brute force", unknown_attack_possible=True, risk_or_gap="protocol-specific features differ from generic flow schema", quarantine_status="representation isolation required", ingestion_priority=6, evidence_priority=2, next_adapter="MQTT flow adapter + semantic intersection"),
        dict(dataset="MedBIoT", official_source="Tallinn University of Technology", raw_or_processed="official/raw", scientific_role="independent botnet transfer validation", generalization_axis=["device", "capture", "botnet_run"], task_types=["binary_detection", "open_set_detection"], group_keys=["device", "capture", "botnet_run"], leakage_keys=["device", "capture", "botnet_run"], feature_leakage_candidates=["IP", "MAC", "device_id", "capture_filename"], raw_format="PCAP", feature_extractor="project parser required", feature_extractor_version="not selected", flow_timeout="not selected", modality="network", feature_space="botnet common-feature projection", required_split="device/capture/botnet-run disjoint", attack_value="botnet propagation", unknown_attack_possible=True, risk_or_gap="narrow threat domain", quarantine_status="none after audit", ingestion_priority=7, evidence_priority=3, next_adapter="botnet-only external benchmark"),
        dict(dataset="Edge-IIoTset", official_source="authors / original publication", raw_or_processed="known/reported-tainted derivative", scientific_role="leakage negative control; future raw-backed IIoT validation", generalization_axis=["device", "capture", "scenario"], task_types=["binary_detection", "ood_detection"], group_keys=["device", "capture", "scenario"], leakage_keys=["serialization_branch", "capture", "device", "scenario"], feature_leakage_candidates=["categorical_placeholder", "IP", "timestamp", "source_file"], raw_format="curated CSV + raw PCAP", feature_extractor="raw reparse required", feature_extractor_version="forbidden curated representation", flow_timeout="must be versioned", modality="IoT/IIoT network", feature_space="raw-backed only", required_split="device/capture/scenario disjoint", attack_value="IoT + IIoT diversity", unknown_attack_possible=True, risk_or_gap="reported 0 vs 0.0 serialization/provenance leakage", quarantine_status="QUARANTINED — raw PCAP reparse + leakage sentinel required", ingestion_priority=8, evidence_priority=3, next_adapter="raw reconstruction; curated data negative-control only"),
        dict(dataset="Bot-IoT", official_source="UNSW Canberra", raw_or_processed="official/processed", scientific_role="botnet prevalence stress test", generalization_axis=["capture", "time", "attack_family"], task_types=["binary_detection"], group_keys=["capture", "attack_run"], leakage_keys=["capture", "attack_run", "time_window"], feature_leakage_candidates=["IP", "port", "sequence", "source_file"], raw_format="PCAP + flow CSV", feature_extractor="UNSW release", feature_extractor_version="verify", flow_timeout="verify", modality="network flow", feature_space="botnet-centric", required_split="capture/time-group disjoint", attack_value="botnet traffic", unknown_attack_possible=True, risk_or_gap="lab prevalence differs from deployment", quarantine_status="audit-required", ingestion_priority=9, evidence_priority=3, next_adapter="schema and label audit"),
        dict(dataset="N-BaIoT", official_source="UCI repository/paper", raw_or_processed="official/processed", scientific_role="device-held-out benign-only anomaly validation", generalization_axis=["device", "attack_run"], task_types=["open_set_detection", "ood_detection"], group_keys=["device", "attack_run"], leakage_keys=["device", "attack_run"], feature_leakage_candidates=["device_id", "source_file"], raw_format="aggregated statistics CSV", feature_extractor="authors", feature_extractor_version="verify", flow_timeout="multi-window aggregation", modality="network statistics", feature_space="aggregated device behavior", required_split="device and attack-run disjoint", attack_value="Mirai/BASHLITE", unknown_attack_possible=True, risk_or_gap="limited intervention timing", quarantine_status="benign-only training protocol", ingestion_priority=10, evidence_priority=3, next_adapter="device-held-out anomaly benchmark"),
    ]
    return pd.DataFrame([{**common, **row} for row in rows])


def eligibility_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    feature_columns: Sequence[str],
    feature_leakage_candidates: Sequence[str],
) -> EligibilityAudit:
    group_leakage = []
    reasons = []
    for key in group_keys:
        if key not in train or key not in test:
            reasons.append(f"missing group key: {key}")
        elif set(train[key].dropna().astype(str)) & set(test[key].dropna().astype(str)):
            group_leakage.append(key)
    forbidden = sorted(set(feature_columns) & set(feature_leakage_candidates))
    if group_leakage:
        reasons.append("group overlap")
    if forbidden:
        reasons.append("feature leakage candidates selected")
    return EligibilityAudit(not reasons, tuple(group_leakage), tuple(forbidden), tuple(reasons))


def dataset_provenance_probe(
    features: pd.DataFrame,
    source_dataset: Sequence[str],
    groups: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Measure how easily representation reveals source dataset identity."""
    X = pd.DataFrame(features).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = np.asarray(source_dataset)
    groups = np.asarray(groups)
    if len(np.unique(y)) < 2:
        raise ValueError("provenance probe requires at least two source datasets")
    if len(np.unique(groups)) < folds:
        raise ValueError("provenance probe requires at least one group per fold")
    predictions = np.empty(len(y), dtype=object)
    for train_idx, test_idx in GroupKFold(folds).split(X, y, groups):
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1)
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "rows": float(len(y)),
        "datasets": float(len(np.unique(y))),
    }


def generalization_manifests(frame: pd.DataFrame, *, holdout: str) -> list[GeneralizationManifest]:
    if holdout not in frame:
        raise ValueError(f"missing holdout column: {holdout}")
    values = sorted(frame[holdout].dropna().astype(str).unique())
    if len(values) < 2:
        raise ValueError("generalization holdout requires at least two values")
    manifests = []
    as_text = frame[holdout].astype(str)
    for value in values:
        test = np.flatnonzero(as_text.eq(value).to_numpy())
        train = np.flatnonzero(as_text.ne(value).to_numpy())
        manifests.append(GeneralizationManifest(
            protocol=f"leave_one_{holdout}_out", holdout_value=value,
            train_indices=tuple(map(int, train)), test_indices=tuple(map(int, test)),
            train_values=tuple(sorted(as_text.iloc[train].unique())), test_values=(value,),
        ))
    return manifests


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or not y.size or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid binary labels or probabilities")
    edges = np.linspace(0, 1, bins + 1)
    membership = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = membership == index
        if selected.any():
            ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": float(ece),
        "ece": float(ece),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
    }


def operational_rates(*, attack_present: np.ndarray, disruptive_action: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    attack = np.asarray(attack_present, dtype=bool)
    disruptive = np.asarray(disruptive_action, dtype=bool)
    times = np.asarray(timestamps, dtype=float)
    if not (attack.shape == disruptive.shape == times.shape) or not attack.size:
        raise ValueError("operational arrays must have equal non-zero shape")
    hours = max((times.max() - times.min()) / 3600, 1 / 3600)
    false_blocks = int((~attack & disruptive).sum())
    missed = int((attack & ~disruptive).sum())
    return {
        "false_blocks_per_hour": false_blocks / hours,
        "false_blocks_per_million_flows": false_blocks / len(attack) * 1_000_000,
        "missed_attacks_per_hour": missed / hours,
        "flows_per_hour_observed": len(attack) / hours,
    }


def abstaining_response(*, attack_probability: float, ood_score: float, calibration_ok: bool, block_threshold: float = .99, ood_threshold: float = .70) -> ResponseDecision:
    if not all(0 <= value <= 1 for value in (attack_probability, ood_score, block_threshold, ood_threshold)):
        raise ValueError("scores and thresholds must be in [0, 1]")
    if ood_score >= ood_threshold or not calibration_ok:
        return ResponseDecision("ABSTAIN_ESCALATE", IpsAction.MONITOR, "OOD or calibration gate")
    if attack_probability >= block_threshold:
        return ResponseDecision("TEMP_BLOCK", IpsAction.TEMP_BLOCK_SOURCE, "high calibrated risk")
    if attack_probability >= .70:
        return ResponseDecision("RATE_LIMIT", IpsAction.RATE_LIMIT, "moderate calibrated risk")
    return ResponseDecision("ALLOW_OBSERVE", IpsAction.MONITOR, "insufficient prevention evidence")


def shield_metrics(proposed: Iterable[IpsAction], executed: Iterable[IpsAction]) -> dict[str, float]:
    raw = [IpsAction(item) for item in proposed]
    safe = [IpsAction(item) for item in executed]
    if len(raw) != len(safe) or not raw:
        raise ValueError("shield sequences must have equal non-zero length")
    severe = {IpsAction.TEMP_BLOCK_SOURCE, IpsAction.BLOCK_DESTINATION_PORT, IpsAction.ISOLATE_HOST}
    severity = {action: int(action) for action in IpsAction}
    return {
        "raw_unsafe_proposal_rate": sum(action in severe for action in raw) / len(raw),
        "shield_intervention_rate": sum(a != b for a, b in zip(raw, safe)) / len(raw),
        "executed_unsafe_action_rate": sum(action in severe for action in safe) / len(safe),
        "mean_severity_downgrade": float(np.mean([max(0, severity[a] - severity[b]) for a, b in zip(raw, safe)])),
    }


def intervention_utility(
    action: IpsAction,
    attack_probability: float,
    context: InterventionContext,
) -> dict[str, float]:
    """Estimate security benefit minus collateral damage and operational cost."""
    bounded = (
        attack_probability, context.asset_criticality, context.identity_confidence,
        context.shared_ip_probability, context.service_dependency_score,
        context.action_blast_radius, context.rollback_cost, context.business_impact,
    )
    if not all(0 <= value <= 1 for value in bounded) or context.estimated_users_behind_identity < 1:
        raise ValueError("blast-radius probabilities/costs must be bounded and users positive")
    effectiveness = {
        IpsAction.ALLOW: 0.0, IpsAction.MONITOR: .05, IpsAction.RATE_LIMIT: .35,
        IpsAction.DROP_FLOW: .55, IpsAction.TEMP_BLOCK_SOURCE: .75,
        IpsAction.BLOCK_DESTINATION_PORT: .85, IpsAction.ISOLATE_HOST: .98,
    }[IpsAction(action)]
    severity = int(action) / max(int(IpsAction.ISOLATE_HOST), 1)
    security_benefit = attack_probability * effectiveness * (0.5 + 0.5 * context.asset_criticality)
    identity_uncertainty = 1 - context.identity_confidence
    affected_scale = np.log1p(context.estimated_users_behind_identity) / np.log(1001)
    collateral = severity * context.action_blast_radius * context.business_impact * (
        .35 * context.shared_ip_probability + .25 * identity_uncertainty
        + .20 * context.service_dependency_score + .20 * affected_scale
    )
    operational = severity * context.rollback_cost * (.5 + .5 * context.asset_criticality)
    return {
        "security_benefit": float(security_benefit),
        "collateral_damage": float(collateral),
        "operational_cost": float(operational),
        "utility": float(security_benefit - collateral - operational),
    }


def constrained_mdp_audit(
    transitions: pd.DataFrame,
    budget: ConstraintBudget,
    *,
    gamma: float = .99,
) -> dict[str, object]:
    """Evaluate the same operational constraints used to judge a policy."""
    required = {"security_reward", "collateral_cost", "critical_outage", "false_block", "latency_ms"}
    missing = required - set(transitions)
    if missing or transitions.empty:
        raise ValueError(f"missing constrained-MDP transition fields: {sorted(missing)}")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    discount = gamma ** np.arange(len(transitions))
    security_return = float(np.sum(discount * transitions.security_reward.to_numpy(float)))
    collateral_return = float(np.sum(discount * transitions.collateral_cost.to_numpy(float)))
    outage_probability = float(transitions.critical_outage.astype(bool).mean())
    false_block_rate = float(transitions.false_block.astype(bool).mean())
    latency_p95 = float(np.percentile(transitions.latency_ms.to_numpy(float), 95))
    constraints = {
        "collateral_budget": collateral_return <= budget.collateral_budget,
        "critical_outage_probability": outage_probability <= budget.critical_outage_probability,
        "false_block_rate": false_block_rate <= budget.false_block_rate,
        "latency_p95_ms": latency_p95 <= budget.latency_p95_ms,
    }
    return {
        "discounted_security_return": security_return,
        "discounted_collateral_cost": collateral_return,
        "critical_outage_probability": outage_probability,
        "false_block_rate": false_block_rate,
        "latency_p95_ms": latency_p95,
        "feasible": all(constraints.values()),
        "constraint_pass": constraints,
    }


def observability_registry() -> pd.DataFrame:
    rows = [
        ("threat_probability", "model_estimated", True, "OOF/calibrated detector output"),
        ("anomaly_score", "model_estimated", True, "benign-only anomaly model"),
        ("estimated_attack_stage", "model_estimated", True, "derived from observable score history"),
        ("estimated_host_compromise", "model_estimated", True, "belief estimate; never ground truth"),
        ("critical_service", "configuration_known", True, "asset inventory"),
        ("recent_score_rate", "sensor_observable", True, "history of detector scores"),
        ("response_budget", "configuration_known", True, "policy budget"),
        ("previous_action", "configuration_known", False, "future recurrent-state extension"),
        ("attack_present", "hidden_ground_truth", False, "reward/evaluation only"),
        ("attack_family", "hidden_ground_truth", False, "stratified reporting only"),
        ("true_attack_stage", "hidden_ground_truth", False, "transition/reward only"),
        ("true_host_compromise", "hidden_ground_truth", False, "transition/reward only"),
        ("future_outcome", "future_information", False, "never available at decision time"),
    ]
    return pd.DataFrame(rows, columns=["variable", "status", "policy_visible", "source_or_rule"])


def source_kind_policy() -> pd.DataFrame:
    return pd.DataFrame([
        ("official/raw", "primary after deterministic parser and leakage audit", "raw-backed evidence"),
        ("official/processed", "only after feature provenance and leakage audit", "processed evidence; raw equivalence not assumed"),
        ("raw capture re-derived", "preferred after parser/hash/split manifest", "raw-backed behavioral evidence"),
        ("Hugging Face mirror", "development after hash/provenance verification", "mirror-specific until verified"),
        ("Kaggle derivative", "sensitivity only", "derivative-only evidence"),
        ("known/reported-tainted derivative", "negative control / leakage demonstration only", "no detector-performance claim"),
        ("survey", "method and hypothesis discovery", "no local empirical claim"),
    ], columns=["source_kind", "allowed_use", "claim_level"])


def feature_semantics_registry() -> pd.DataFrame:
    return pd.DataFrame([
        ("flow_duration", "EXACT", "use when units and timeout match"),
        ("packet_rate", "DERIVED_EQUIVALENT", "record derivation and time unit"),
        ("iat_mean", "APPROXIMATE", "sensitivity analysis; extractor-dependent"),
        ("tcp_flags", "MISSING", "use semantic intersection or presence mask; never silent zero-fill"),
        ("src_ip", "FORBIDDEN", "identity/provenance leakage candidate"),
        ("capture_filename", "FORBIDDEN", "dataset identity shortcut"),
    ], columns=["canonical_feature", "mapping_status", "rule"])


def registry_records() -> list[dict[str, object]]:
    """JSON-serializable registry helper."""
    return [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in dataset_registry().to_dict("records")]
