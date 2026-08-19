"""Tests for the IPS claim-control and generalization framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ips.actions import IpsAction
from ips.dataset import IpsEpisode, IpsEvent
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.analysis.controls import (  # pyright: ignore[reportMissingImports]
    ConstraintBudget,
    InterventionContext,
    abstaining_response,
    calibration_metrics,
    constrained_mdp_audit,
    dataset_provenance_probe,
    dataset_registry,
    eligibility_audit,
    generalization_manifests,
    intervention_utility,
    observability_registry,
    operational_rates,
    shield_metrics,
)


def test_registry_separates_priorities_and_quarantines_tainted_derivative():
    registry = dataset_registry()
    assert {"ingestion_priority", "evidence_priority", "generalization_axis", "leakage_keys"} <= set(registry)
    edge = registry.set_index("dataset").loc["Edge-IIoTset"]
    assert edge.quarantine_status.startswith("QUARANTINED")
    assert registry.set_index("dataset").loc["CICAPT-IIoT"].modality == "provenance graph"


def test_eligibility_gate_detects_group_and_feature_leakage():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "src_ip": [1, 1, 2, 2], "x": [0, 1, 2, 3]})
    failed = eligibility_audit(frame.iloc[:3], frame.iloc[2:], group_keys=["group"], feature_columns=["src_ip", "x"], feature_leakage_candidates=["src_ip"])
    assert not failed.eligible
    assert "group" in failed.group_leakage
    passed = eligibility_audit(frame.iloc[:2], frame.iloc[2:], group_keys=["group"], feature_columns=["x"], feature_leakage_candidates=["src_ip"])
    assert passed.eligible


def test_provenance_probe_and_generalization_manifests_are_explicit():
    X = pd.DataFrame({"x": np.r_[np.zeros(20), np.ones(20)], "noise": np.arange(40)})
    source = np.array(["A"] * 20 + ["B"] * 20)
    groups = np.array([f"g{i//2}" for i in range(40)])
    probe = dataset_provenance_probe(X, source, groups, folds=3, seed=42)
    assert probe["balanced_accuracy"] > 0.9
    frame = pd.DataFrame({"dataset": source, "family": ["normal", "attack"] * 20})
    lodo = generalization_manifests(frame, holdout="dataset")
    loafo = generalization_manifests(frame, holdout="family")
    assert len(lodo) == len(loafo) == 2
    assert all(set(item.train_values).isdisjoint(item.test_values) for item in lodo)


def test_calibration_operational_abstention_and_shield_metrics():
    calibration = calibration_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), bins=2)
    assert 0 <= calibration["ece"] <= 1
    rates = operational_rates(
        attack_present=np.array([0, 0, 1, 1]),
        disruptive_action=np.array([1, 0, 0, 1]),
        timestamps=np.array([0, 1200, 2400, 3600]),
    )
    assert rates["false_blocks_per_hour"] == pytest.approx(1.0)
    abstained = abstaining_response(attack_probability=.95, ood_score=.9, calibration_ok=True)
    assert abstained.disposition == "ABSTAIN_ESCALATE"
    assert abstained.executed_action == IpsAction.MONITOR
    shield = shield_metrics(
        [IpsAction.ISOLATE_HOST, IpsAction.ALLOW],
        [IpsAction.DROP_FLOW, IpsAction.ALLOW],
    )
    assert shield["shield_intervention_rate"] == 0.5
    assert shield["executed_unsafe_action_rate"] == 0.0


def test_observability_registry_keeps_truth_out_of_policy_state():
    registry = observability_registry()
    policy = registry[registry.policy_visible]
    assert not policy.variable.isin(["attack_present", "attack_family", "true_attack_stage", "true_host_compromise"]).any()
    assert set(registry.status) >= {"sensor_observable", "model_estimated", "hidden_ground_truth"}


def test_blast_radius_utility_and_constrained_mdp_are_operationally_aligned():
    low_blast = InterventionContext(.8, .9, .1, 2, .1, .2, .1, .2)
    high_blast = InterventionContext(.8, .4, .9, 500, .9, 1.0, .8, 1.0)
    assert intervention_utility(IpsAction.TEMP_BLOCK_SOURCE, .95, low_blast)["utility"] > intervention_utility(IpsAction.TEMP_BLOCK_SOURCE, .95, high_blast)["utility"]
    transitions = pd.DataFrame({
        "security_reward": [2.0, 1.0], "collateral_cost": [.1, .2],
        "critical_outage": [0, 0], "false_block": [0, 0], "latency_ms": [1, 2],
    })
    audit = constrained_mdp_audit(
        transitions,
        ConstraintBudget(collateral_budget=1, critical_outage_probability=.01, false_block_rate=.1, latency_p95_ms=5),
    )
    assert audit["feasible"]
    assert audit["discounted_security_return"] > 0


def test_policy_observation_does_not_reveal_label_or_true_stage():
    def episode(attack: bool, stage: float) -> IpsEpisode:
        return IpsEpisode("ep", "group", (
            IpsEvent(1.0, .7, .6, attack, stage, False, "DoS" if attack else "normal"),
        ))
    benign_obs, benign_info = DatasetBackedIpsEnv(episode(False, 0.0)).reset()
    attack_obs, attack_info = DatasetBackedIpsEnv(episode(True, 1.0)).reset()
    assert np.array_equal(benign_obs.as_array(), attack_obs.as_array())
    assert np.array_equal(benign_info["action_mask"], attack_info["action_mask"])
