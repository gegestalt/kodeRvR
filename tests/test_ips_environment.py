"""Safety and reproducibility tests for the adaptive IPS environment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ips.actions import IpsAction, valid_action_mask
from ips.dqn import DqnConfig, QNetwork, ReplayBuffer, Transition, epsilon_at_step, select_action
from ips.environment import AdaptiveIpsEnv
from ips.evaluate import evaluate_policy
from ips.policies import allow_policy, aggressive_policy, rule_based_policy
from ips.reward import RewardConfig, calculate_reward


def test_critical_host_isolation_requires_strong_evidence():
    mask = valid_action_mask(0.85, critical_service=True, host_compromise=0.85)
    assert not mask[IpsAction.ISOLATE_HOST]
    assert not mask[IpsAction.BLOCK_DESTINATION_PORT]
    assert mask[IpsAction.DROP_FLOW]


def test_unsafe_action_is_masked_to_monitor():
    env = AdaptiveIpsEnv(seed=3)
    _, _ = env.reset(seed=3, attack_present=False, critical_service=True)
    result = env.step(IpsAction.ISOLATE_HOST)
    assert result.info["action_was_masked"] is True
    assert result.info["executed_action"] == "MONITOR"


def test_reset_and_transitions_are_seed_reproducible():
    first = AdaptiveIpsEnv(seed=17)
    second = AdaptiveIpsEnv(seed=17)
    obs_a, _ = first.reset(attack_present=True, critical_service=False)
    obs_b, _ = second.reset(attack_present=True, critical_service=False)
    assert np.array_equal(obs_a.as_array(), obs_b.as_array())

    step_a = first.step(IpsAction.DROP_FLOW)
    step_b = second.step(IpsAction.DROP_FLOW)
    assert step_a.reward == step_b.reward
    assert step_a.info["contained"] == step_b.info["contained"]
    assert np.array_equal(step_a.observation.as_array(), step_b.observation.as_array())


def test_false_prevention_costs_more_on_critical_service():
    cfg = RewardConfig()
    normal = calculate_reward(
        action=IpsAction.ISOLATE_HOST,
        attack_present=False,
        contained=False,
        compromised=False,
        critical_service=False,
        config=cfg,
    )
    critical = calculate_reward(
        action=IpsAction.ISOLATE_HOST,
        attack_present=False,
        contained=False,
        compromised=False,
        critical_service=True,
        config=cfg,
    )
    assert critical < normal


def test_containment_is_better_than_compromise():
    cfg = RewardConfig()
    contained = calculate_reward(
        action=IpsAction.DROP_FLOW,
        attack_present=True,
        contained=True,
        compromised=False,
        critical_service=False,
        config=cfg,
    )
    compromised = calculate_reward(
        action=IpsAction.ALLOW,
        attack_present=True,
        contained=False,
        compromised=True,
        critical_service=False,
        config=cfg,
    )
    assert contained > compromised


def test_rule_policy_returns_a_valid_action():
    env = AdaptiveIpsEnv(seed=42)
    obs, info = env.reset(attack_present=True, critical_service=False)
    action = rule_based_policy(obs, info["action_mask"])
    assert info["action_mask"][int(action)]


def test_finished_episode_requires_reset():
    env = AdaptiveIpsEnv(max_steps=1)
    env.reset(attack_present=False)
    result = env.step(IpsAction.ALLOW)
    assert result.truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(IpsAction.ALLOW)


def test_policy_evaluation_is_reproducible():
    first = evaluate_policy(rule_based_policy, episodes=20, seed=11)
    second = evaluate_policy(rule_based_policy, episodes=20, seed=11)
    assert first == second


def test_policy_metrics_remain_bounded():
    metrics = evaluate_policy(allow_policy, episodes=20, seed=5)
    assert 0.0 <= metrics.containment_rate <= 1.0
    assert 0.0 <= metrics.compromise_rate <= 1.0
    assert metrics.false_preventions_per_episode == 0.0


def test_aggressive_policy_exposes_more_disruption_than_allow():
    allow = evaluate_policy(allow_policy, episodes=30, seed=9)
    aggressive = evaluate_policy(aggressive_policy, episodes=30, seed=9)
    assert aggressive.disruptive_actions_per_episode > allow.disruptive_actions_per_episode


def test_dqn_output_matches_action_space():
    network = QNetwork(state_dim=7, action_dim=len(IpsAction), hidden_dim=16)
    output = network(torch.zeros((3, 7)))
    assert output.shape == (3, len(IpsAction))


def test_dqn_selection_never_chooses_a_masked_action():
    network = QNetwork(state_dim=7, action_dim=len(IpsAction), hidden_dim=8)
    mask = np.array([True, True, False, False, False, False, False])
    rng = np.random.default_rng(4)
    actions = {
        select_action(network, np.zeros(7), mask, epsilon=1.0, rng=rng)
        for _ in range(50)
    }
    assert actions <= {IpsAction.ALLOW, IpsAction.MONITOR}


def test_replay_buffer_is_bounded_and_sampled_without_replacement():
    buffer = ReplayBuffer(capacity=2, seed=1)
    for action in range(3):
        buffer.append(
            Transition(np.zeros(7), action, 0.0, np.ones(7), False, np.ones(7, dtype=bool))
        )
    assert len(buffer) == 2
    assert {item.action for item in buffer.sample(2)} == {1, 2}


def test_epsilon_schedule_reaches_floor():
    config = DqnConfig(epsilon_decay_steps=100)
    assert epsilon_at_step(0, config) == pytest.approx(config.epsilon_start)
    assert epsilon_at_step(100, config) == pytest.approx(config.epsilon_end)
    assert epsilon_at_step(1_000, config) == pytest.approx(config.epsilon_end)
