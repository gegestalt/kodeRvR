"""Tests for real-data adaptation and advanced IPS evidence/policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ips.actions import IpsAction
from ips.advanced_policies import PpoConfig, actor_policy, train_bandit, train_ppo
from ips.dataset import build_episodes
from ips.analysis.evidence import (
    adversarial_noise_sweep,
    evaluate_detailed,
    profile_policy,
    reward_sensitivity,
)
from ips.analysis.experiments import factorial_candidates
from ips.adapters.real_events import AdapterConfig, build_real_events


def raw_frame(groups: int = 6) -> pd.DataFrame:
    rows = []
    labels = ["BENIGN", "DoS Hulk", "PortScan", "Brute Force", "Exploits", "R2L"]
    for group in range(groups):
        for step in range(4):
            attack = group > 0
            rows.append(
                {
                    "timestamp": f"2026-01-01 00:{group:02d}:{step:02d}",
                    "src": f"host-{group}",
                    "dst": "server",
                    "label": labels[group],
                    "bytes": 100 + group * 50 + step,
                    "packets": 5 + group + step,
                    "critical": group == 1,
                }
            )
    return pd.DataFrame(rows)


def event_episodes():
    rows = []
    for group, family in enumerate(("normal", "DoS", "Probe", "BruteForce", "Exploitation", "R2L")):
        for step in range(3):
            attack = family != "normal"
            rows.append(
                {
                    "episode_id": f"ep-{group}", "group_id": f"g-{group}",
                    "timestamp": float(step), "threat_probability": .8 if attack else .1,
                    "anomaly_score": .75 if attack else .1, "attack_present": attack,
                    "attack_stage": (step+1)/3 if attack else 0.0,
                    "critical_service": False, "attack_family": family,
                }
            )
    return build_episodes(pd.DataFrame(rows))


def test_real_adapter_builds_group_safe_oof_scores():
    events = build_real_events(
        raw_frame(),
        AdapterConfig("timestamp", "label", ("src", "dst"), critical_col="critical", folds=3),
    )
    assert events.threat_probability.between(0, 1).all()
    assert events.anomaly_score.between(0, 1).all()
    assert {"DoS", "Probe", "BruteForce", "Exploitation", "R2L"} <= set(events.attack_family)


def test_detailed_evaluation_reports_families_and_all_actions():
    policy = lambda state, mask: IpsAction.DROP_FLOW if mask[IpsAction.DROP_FLOW] else IpsAction.MONITOR
    families, actions, summary = evaluate_detailed(policy, event_episodes(), seed=2)
    assert {"DoS", "Probe", "BruteForce", "Exploitation", "R2L"} <= set(families.attack_family)
    assert set(actions.action) == {action.name for action in IpsAction}
    assert 0 <= summary["containment_rate"] <= 1


def test_reward_and_noise_sweeps_have_expected_scenarios():
    policy = lambda state, mask: IpsAction.ALLOW
    rewards = reward_sensitivity(policy, event_episodes())
    noise = adversarial_noise_sweep(policy, event_episodes())
    assert len(rewards) == 12
    assert "combined_stress" in set(noise.scenario)


def test_factorial_grid_has_all_nine_interactions():
    candidates = factorial_candidates(episodes=5)
    assert len(candidates) == 9
    assert len({(c.hidden_dim, c.batch_size) for c in candidates}) == 9


def test_bandit_and_ppo_train_and_respect_action_interface():
    episodes = event_episodes()
    bandit = train_bandit(episodes, training_episodes=3, seed=1)
    action = bandit.choose(np.zeros(7), np.ones(7, dtype=bool))
    assert isinstance(action, IpsAction)
    ppo, history = train_ppo(
        episodes,
        config=PpoConfig(training_episodes=2, hidden_dim=8, update_epochs=1),
        seed=1,
    )
    constrained, constrained_history = train_ppo(
        episodes,
        config=PpoConfig(training_episodes=2, hidden_dim=8, update_epochs=1, constrained=True),
        seed=1,
    )
    assert len(history) == len(constrained_history) == 2
    assert isinstance(actor_policy(ppo)(np.zeros(7), np.ones(7, dtype=bool)), IpsAction)
    assert constrained is not None


def test_resource_profile_reports_latency_and_rss():
    profile = profile_policy(
        lambda state, mask: IpsAction.ALLOW,
        np.zeros((3, 7)),
        np.ones((3, 7), dtype=bool),
        repeats=2,
    )
    assert profile["latency_p95_ms"] >= 0
    assert profile["max_rss_mb"] > 0
    assert profile["rss_scope"] == "shared_process_peak_not_policy_attributable"


def test_repeated_resource_profile_reports_variability():
    from ips.analysis.evidence import profile_policy_trials

    states = np.zeros((2, 7), dtype=np.float32)
    masks = np.ones((2, 7), dtype=bool)
    policy = lambda state, mask: IpsAction.ALLOW
    trials, summary = profile_policy_trials(
        policy, states, masks, trials=3, repeats_per_trial=2
    )
    assert len(trials) == 3
    assert summary["profile_trials"] == 3
    assert summary["latency_p95_std_ms"] >= 0
