"""Tests for scientifically adopted methods from prior IPS/RL work."""

from __future__ import annotations

import numpy as np

from ips.actions import IpsAction
from ips.dqn import Transition
from ips.dataset import EpisodeSplits, IpsEpisode, IpsEvent
from ips.dqn import DqnConfig
from ips.research_adoptions import (
    PrioritizedReplayBuffer,
    RepeatedEvidenceGate,
    effective_sample_size,
    optimal_stopping_policy,
    shared_ip_guardrail,
    train_prioritized_dqn,
    weighted_importance_sampling,
)


def _transition(reward: float) -> Transition:
    return Transition(
        np.zeros(7, dtype=np.float32), int(IpsAction.ALLOW), reward,
        np.ones(7, dtype=np.float32), False, np.ones(7, dtype=bool),
    )


def test_prioritized_replay_samples_high_priority_more_often():
    replay = PrioritizedReplayBuffer(10, alpha=1.0, seed=42)
    replay.append(_transition(0), priority=0.01)
    replay.append(_transition(1), priority=10.0)
    counts = np.zeros(2, dtype=int)
    for _ in range(500):
        _, indices, weights = replay.sample(1, beta=0.4)
        counts[indices[0]] += 1
        assert 0 < weights[0] <= 1
    assert counts[1] > counts[0] * 10


def test_optimal_stopping_policy_respects_valid_mask():
    state = np.array([0.95, 0.8, 0.9, 0.9, 0.0, 1.0, 1.0])
    mask = np.ones(7, dtype=bool)
    mask[IpsAction.ISOLATE_HOST] = False
    action = optimal_stopping_policy(state, mask, threshold=0.7)
    assert action != IpsAction.ISOLATE_HOST
    assert mask[int(action)]


def test_weighted_importance_sampling_and_ess():
    rewards = np.array([1.0, 2.0, 3.0])
    target = np.array([0.5, 0.5, 0.5])
    behavior = np.array([0.25, 0.5, 1.0])
    estimate = weighted_importance_sampling(rewards, target, behavior)
    assert np.isfinite(estimate)
    weights = target / behavior
    assert 1 <= effective_sample_size(weights) <= len(weights)


def test_prioritized_dqn_training_smoke():
    episodes = tuple(
        IpsEpisode(
            f"episode-{index}", f"group-{index}",
            tuple(
                IpsEvent(float(step), 0.8, 0.8, True, (step + 1) / 2, False, "DoS")
                for step in range(2)
            ),
        )
        for index in range(6)
    )
    splits = EpisodeSplits(episodes[:3], episodes[3:5], episodes[5:])
    network, metrics, status = train_prioritized_dqn(
        splits,
        episodes=3,
        validation_interval=1,
        config=DqnConfig(hidden_dim=8, batch_size=2, replay_capacity=20,
                         warmup_steps=2, target_update_steps=2, epsilon_decay_steps=10),
        seed=1,
    )
    assert network is not None
    assert metrics.episodes == 2
    assert status["updates"] > 0


def test_repeated_evidence_gate_activates_and_forgets_per_identity():
    gate = RepeatedEvidenceGate(positives_to_activate=2, negatives_to_forget=2)
    assert not gate.update("shared-source", True)
    assert gate.update("shared-source", True)
    assert gate.update("shared-source", False)
    assert not gate.update("shared-source", False)
    assert not gate.update("different-source", False)


def test_shared_ip_guardrail_downgrades_only_broad_actions():
    mask = np.ones(len(IpsAction), dtype=bool)
    assert shared_ip_guardrail(
        IpsAction.TEMP_BLOCK_SOURCE, mask, shared_ip_probability=0.9
    ) == IpsAction.DROP_FLOW
    assert shared_ip_guardrail(
        IpsAction.RATE_LIMIT, mask, shared_ip_probability=0.9
    ) == IpsAction.RATE_LIMIT
    assert shared_ip_guardrail(
        IpsAction.TEMP_BLOCK_SOURCE, mask, shared_ip_probability=0.2
    ) == IpsAction.TEMP_BLOCK_SOURCE
