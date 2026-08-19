"""Dataset-backed IPS episode and leakage tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ips.actions import IpsAction
from ips.dataset import assert_group_disjoint, build_episodes, split_episodes_by_group
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.dataset_experiment import DatasetTrainConfig, run_multi_seed
from ips.dqn import DqnConfig


def event_frame(groups: int = 6) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        for step in range(3):
            attack = group % 2 == 0
            rows.append(
                {
                    "episode_id": f"episode-{group}",
                    "group_id": f"campaign-{group}",
                    "timestamp": float(step),
                    "threat_probability": 0.85 if attack else 0.10,
                    "anomaly_score": 0.80 if attack else 0.15,
                    "attack_present": attack,
                    "attack_stage": (step + 1) / 3 if attack else 0.0,
                    "critical_service": group == 0,
                    "attack_family": "R2L" if attack else "normal",
                }
            )
    return pd.DataFrame(rows)


def test_build_episodes_sorts_events_and_preserves_detector_scores():
    frame = event_frame().sample(frac=1.0, random_state=2)
    episodes = build_episodes(frame)
    first = next(ep for ep in episodes if ep.episode_id == "episode-0")
    assert [event.timestamp for event in first.events] == [0.0, 1.0, 2.0]
    assert first.events[0].threat_probability == pytest.approx(0.85)


def test_builder_requires_out_of_fold_detector_score_column():
    with pytest.raises(ValueError, match="threat_probability"):
        build_episodes(event_frame().drop(columns="threat_probability"))


def test_group_split_is_disjoint_and_deterministic():
    episodes = build_episodes(event_frame(10))
    first = split_episodes_by_group(episodes, seed=9)
    second = split_episodes_by_group(episodes, seed=9)
    assert first == second
    assert_group_disjoint(first)
    assert {ep.group_id for ep in first.train}.isdisjoint(
        {ep.group_id for ep in first.test}
    )


def test_episode_cannot_cross_campaign_groups():
    frame = event_frame()
    frame.loc[1, "group_id"] = "leaked-campaign"
    with pytest.raises(ValueError, match="crosses multiple"):
        build_episodes(frame)


def test_dataset_environment_uses_observed_score_and_marks_counterfactual():
    episode = build_episodes(event_frame())[0]
    env = DatasetBackedIpsEnv(episode, seed=3)
    observation, _ = env.reset(seed=3)
    assert observation.threat_probability == pytest.approx(0.85)
    result = env.step(IpsAction.DROP_FLOW)
    assert result.info["dataset_backed_observation"] is True
    assert result.info["counterfactual_outcome"] is True


def test_dataset_environment_masks_unsafe_critical_isolation():
    episode = build_episodes(event_frame())[0]
    env = DatasetBackedIpsEnv(episode, seed=5)
    env.reset(seed=5)
    result = env.step(IpsAction.ISOLATE_HOST)
    assert result.info["action_was_masked"] is True
    assert result.info["executed_action"] == "MONITOR"


def test_multi_seed_protocol_writes_final_test_summary(tmp_path):
    splits = split_episodes_by_group(build_episodes(event_frame(9)), seed=3)
    rows = run_multi_seed(
        splits,
        seeds=(1, 2, 3),
        dqn=DqnConfig(
            hidden_dim=8,
            batch_size=2,
            replay_capacity=64,
            warmup_steps=2,
            target_update_steps=2,
            epsilon_decay_steps=20,
        ),
        training=DatasetTrainConfig(episodes=3, validation_interval=2),
        output_dir=tmp_path,
    )
    assert rows["seed"].tolist() == [1, 2, 3]
    assert (tmp_path / "final_test_summary.json").exists()
