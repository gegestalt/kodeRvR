"""Numerical and smoke tests for masked Double-DQN training."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ips.dqn import DqnConfig, QNetwork, ReplayBuffer, Transition
from ips.train_dqn import TrainConfig, optimize_batch, train


def _zero_network(network: QNetwork) -> None:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()


def test_optimize_batch_uses_masked_double_dqn_target():
    config = DqnConfig(
        state_dim=7,
        action_dim=7,
        hidden_dim=4,
        gamma=0.5,
        batch_size=1,
        replay_capacity=2,
    )
    online = QNetwork(7, 7, 4)
    target = QNetwork(7, 7, 4)
    _zero_network(online)
    _zero_network(target)
    with torch.no_grad():
        target.layers[-1].bias[0] = 4.0
        target.layers[-1].bias[6] = 100.0  # masked; must never enter target

    replay = ReplayBuffer(2)
    replay.append(
        Transition(
            state=np.zeros(7, dtype=np.float32),
            action=0,
            reward=2.0,
            next_state=np.ones(7, dtype=np.float32),
            terminated=False,
            next_action_mask=np.array([True, False, False, False, False, False, False]),
        )
    )
    optimizer = torch.optim.SGD(online.parameters(), lr=0.01)

    loss = optimize_batch(online, target, optimizer, replay, config)

    # target = 2 + 0.5*4 = 4; SmoothL1(0, 4) = 3.5
    assert loss == pytest.approx(3.5)


def test_terminal_transition_does_not_bootstrap():
    config = DqnConfig(hidden_dim=4, gamma=0.99, batch_size=1, replay_capacity=1)
    online, target = QNetwork(7, 7, 4), QNetwork(7, 7, 4)
    _zero_network(online)
    _zero_network(target)
    with torch.no_grad():
        target.layers[-1].bias.fill_(100.0)
    replay = ReplayBuffer(1)
    replay.append(
        Transition(
            np.zeros(7), 0, 2.0, np.ones(7), True, np.ones(7, dtype=bool)
        )
    )
    loss = optimize_batch(
        online, target, torch.optim.SGD(online.parameters(), lr=0.01), replay, config
    )
    assert loss == pytest.approx(1.5)  # SmoothL1(0, reward=2)


def test_train_smoke_writes_loadable_artifacts(tmp_path):
    dqn = DqnConfig(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=64,
        warmup_steps=2,
        target_update_steps=2,
        epsilon_decay_steps=20,
    )
    run = TrainConfig(episodes=3, validation_interval=2, validation_episodes=4)

    result = train(dqn, run, seed=7, output_dir=tmp_path)

    assert result.total_steps > 0
    assert result.updates > 0
    assert result.best_checkpoint.exists()
    assert result.history_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["best_validation"]["episodes"] == 4
    checkpoint = torch.load(result.best_checkpoint, map_location="cpu")
    assert checkpoint["dqn_config"]["action_dim"] == 7
