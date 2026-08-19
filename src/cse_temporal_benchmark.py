"""Run identical multi-seed IPS policies on fixed CSE temporal day roles."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ips.actions import IpsAction
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.dataset_experiment import DatasetTrainConfig, train_one_seed
from ips.dqn import DqnConfig, QNetwork
from ips.evidence_analysis import evaluate_detailed, network_policy
from ips.research_adoptions import optimal_stopping_policy, train_prioritized_dqn
from ips.temporal_evidence import log_shadow_decision, split_events_by_role


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "ips_events" / "cse_cic_ids2018_temporal_events.parquet"
OUTPUT = ROOT / "results" / "notebook_ips_lab" / "cse_temporal_evidence"
SEEDS = (42, 43, 44, 45, 46)


def _metrics(method: str, seed: int, runtime: float, summary: dict[str, float]) -> dict[str, object]:
    return {"method": method, "seed": seed, "training_wall_s": runtime, **summary}


def _shadow_log(episodes, *, seed: int = 42, epsilon: float = 0.10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for episode_index, episode in enumerate(episodes):
        env = DatasetBackedIpsEnv(episode, seed=seed + episode_index)
        observation, info = env.reset(seed=seed + episode_index)
        step_index = 0
        while True:
            mask = np.asarray(info["action_mask"], dtype=bool)
            proposed = optimal_stopping_policy(observation.as_array(), mask)
            valid = np.flatnonzero(mask)
            probabilities = np.zeros(len(IpsAction), dtype=float)
            probabilities[valid] = epsilon / len(valid)
            probabilities[int(proposed)] += 1 - epsilon
            executed = IpsAction(int(rng.choice(len(IpsAction), p=probabilities)))
            result = env.step(executed)
            records.append(log_shadow_decision(
                episode_id=episode.episode_id,
                timestamp=episode.events[step_index].timestamp,
                proposed=proposed,
                executed=executed,
                action_mask=mask,
                epsilon=epsilon,
                evidence_kind="counterfactual_dataset_replay",
                observed_reward=result.reward,
            ))
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            step_index += 1
            if result.terminated or result.truncated:
                break
    return pd.DataFrame(records)


def run() -> pd.DataFrame:
    if not EVENTS.exists():
        raise FileNotFoundError(f"run src/cse_temporal_ips.py first: {EVENTS}")
    events = pd.read_parquet(EVENTS)
    splits = split_events_by_role(events, max_events=128)
    config = DqnConfig(
        hidden_dim=32, batch_size=32, replay_capacity=20_000,
        warmup_steps=64, target_update_steps=100, epsilon_decay_steps=8_000,
    )
    training = DatasetTrainConfig(episodes=60, validation_interval=20)
    rows = []
    family_tables = []
    action_tables = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        started = time.perf_counter()
        uniform = train_one_seed(
            splits, seed=seed, dqn=config, training=training,
            output_dir=OUTPUT / "uniform" / f"seed_{seed}", evaluate_test=False,
        )
        checkpoint = torch.load(uniform["checkpoint"], map_location="cpu")
        network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim)
        network.load_state_dict(checkpoint["model_state_dict"])
        families, actions, summary = evaluate_detailed(network_policy(network, seed), splits.test, seed=seed)
        rows.append(_metrics("Uniform masked Double-DQN", seed, time.perf_counter() - started, summary))
        family_tables.append(families.assign(method="Uniform masked Double-DQN", seed=seed))
        action_tables.append(actions.assign(method="Uniform masked Double-DQN", seed=seed))

        started = time.perf_counter()
        per_network, _, _ = train_prioritized_dqn(
            splits, episodes=training.episodes, validation_interval=training.validation_interval,
            config=config, seed=seed,
        )
        families, actions, summary = evaluate_detailed(network_policy(per_network, seed), splits.test, seed=seed)
        rows.append(_metrics("Prioritized replay Double-DQN", seed, time.perf_counter() - started, summary))
        family_tables.append(families.assign(method="Prioritized replay Double-DQN", seed=seed))
        action_tables.append(actions.assign(method="Prioritized replay Double-DQN", seed=seed))

        started = time.perf_counter()
        policy = lambda state, mask: optimal_stopping_policy(state, mask)
        families, actions, summary = evaluate_detailed(policy, splits.test, seed=seed)
        rows.append(_metrics("Optimal-stopping threshold", seed, time.perf_counter() - started, summary))
        family_tables.append(families.assign(method="Optimal-stopping threshold", seed=seed))
        action_tables.append(actions.assign(method="Optimal-stopping threshold", seed=seed))

    runs = pd.DataFrame(rows)
    summary = runs.groupby("method", as_index=False).agg(
        seeds=("seed", "nunique"),
        containment_mean=("containment_rate", "mean"), containment_std=("containment_rate", "std"),
        compromise_mean=("compromise_rate", "mean"), compromise_std=("compromise_rate", "std"),
        return_mean=("mean_return", "mean"), return_std=("mean_return", "std"),
        false_preventions_mean=("false_preventions_per_episode", "mean"),
        training_wall_mean_s=("training_wall_s", "mean"),
    )
    runs.to_csv(OUTPUT / "policy_five_seed_runs.csv", index=False)
    summary.to_csv(OUTPUT / "policy_five_seed_summary.csv", index=False)
    pd.concat(family_tables, ignore_index=True).to_csv(OUTPUT / "attack_family_results.csv", index=False)
    pd.concat(action_tables, ignore_index=True).to_csv(OUTPUT / "action_distribution.csv", index=False)
    shadow = _shadow_log(splits.validation)
    shadow.to_json(OUTPUT / "shadow_propensity_log.jsonl", orient="records", lines=True)
    status = {
        "dataset": "official CSE-CIC-IDS2018 three-day temporal subset",
        "seeds": list(SEEDS),
        "episode_cap": 128,
        "policy_protocol": "identical train/validation/final-test days and reward environment",
        "shadow_rows": len(shadow),
        "shadow_evidence_kind": "counterfactual_dataset_replay",
        "doubly_robust_ope": "BLOCKED: no observed deployment outcomes",
        "cyber_range_transfer": "BLOCKED: temporal dataset benchmark must not be relabelled cyber-range evidence",
    }
    (OUTPUT / "evidence_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run().to_string(index=False))
