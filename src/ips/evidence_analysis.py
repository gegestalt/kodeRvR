"""Detailed IPS evidence: families, actions, rewards, robustness, resources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
import resource
import time

import numpy as np
import pandas as pd
import torch

from ips.actions import IpsAction
from ips.dataset import IpsEpisode, IpsEvent
from ips.dataset_environment import DatasetBackedIpsEnv
from ips.dqn import QNetwork, select_action
from ips.reward import RewardConfig

Policy = Callable[[np.ndarray, np.ndarray], IpsAction]


def network_policy(network: QNetwork, seed: int = 42) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda state, mask: select_action(
        network, state, mask, epsilon=0.0, rng=rng
    )


def evaluate_detailed(
    policy: Policy,
    episodes: tuple[IpsEpisode, ...],
    *,
    seed: int = 42,
    reward_config: RewardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return attack-family outcomes, action distribution, and summary."""
    family = Counter()
    actions = Counter()
    returns: list[float] = []
    compromised = contained = false_preventions = 0
    for index, episode in enumerate(episodes):
        env = DatasetBackedIpsEnv(
            episode, seed=seed + index, reward_config=reward_config
        )
        observation, info = env.reset(seed=seed + index)
        total = 0.0
        episode_contained = episode_compromised = False
        attack_families = sorted(
            {event.attack_family for event in episode.events if event.attack_present}
        )
        while True:
            action = policy(observation.as_array(), np.asarray(info["action_mask"]))
            result = env.step(action)
            executed = str(result.info["executed_action"])
            actions[executed] += 1
            total += result.reward
            episode_contained |= bool(result.info["contained"])
            episode_compromised |= bool(result.info["compromised"])
            if not episode.contains_attack and executed not in {"ALLOW", "MONITOR"}:
                false_preventions += 1
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            if result.terminated or result.truncated:
                break
        returns.append(total)
        contained += int(episode_contained)
        compromised += int(episode_compromised)
        for name in attack_families:
            family[(name, "episodes")] += 1
            family[(name, "contained")] += int(episode_contained)
            family[(name, "compromised")] += int(episode_compromised)
    family_rows = []
    for name in sorted({key[0] for key in family}):
        count = family[(name, "episodes")]
        family_rows.append(
            {
                "attack_family": name,
                "episodes": count,
                "contained": family[(name, "contained")],
                "compromised": family[(name, "compromised")],
                "containment_rate": family[(name, "contained")] / count,
                "compromise_rate": family[(name, "compromised")] / count,
            }
        )
    action_total = sum(actions.values())
    action_rows = [
        {
            "action": action.name,
            "count": actions[action.name],
            "fraction": actions[action.name] / action_total if action_total else 0.0,
        }
        for action in IpsAction
    ]
    attack_episodes = sum(episode.contains_attack for episode in episodes)
    summary = {
        "episodes": float(len(episodes)),
        "attack_episodes": float(attack_episodes),
        "containment_rate": contained / attack_episodes if attack_episodes else 0.0,
        "compromise_rate": compromised / attack_episodes if attack_episodes else 0.0,
        "false_preventions_per_episode": false_preventions / len(episodes),
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
    return pd.DataFrame(family_rows), pd.DataFrame(action_rows), summary


def reward_sensitivity(
    policy: Policy,
    episodes: tuple[IpsEpisode, ...],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate one frozen policy over a one-factor-at-a-time reward sweep."""
    baseline = RewardConfig()
    sweeps = {
        "contained_attack": (4.0, 8.0, 12.0),
        "successful_compromise": (-15.0, -25.0, -40.0),
        "false_prevention": (-2.0, -4.0, -8.0),
        "critical_service_disruption": (-6.0, -12.0, -24.0),
    }
    rows = []
    for parameter, values in sweeps.items():
        for value in values:
            config = replace(baseline, **{parameter: value})
            _, _, summary = evaluate_detailed(
                policy, episodes, seed=seed, reward_config=config
            )
            rows.append(
                {"parameter": parameter, "value": value, **summary}
            )
    return pd.DataFrame(rows)


def perturb_episodes(
    episodes: tuple[IpsEpisode, ...],
    *,
    score_noise: float = 0.0,
    false_positive_shift: float = 0.0,
    telemetry_dropout: float = 0.0,
    seed: int = 42,
) -> tuple[IpsEpisode, ...]:
    """Corrupt detector observations without changing ground-truth outcomes."""
    if min(score_noise, false_positive_shift, telemetry_dropout) < 0:
        raise ValueError("perturbation strengths must be non-negative")
    if telemetry_dropout > 1:
        raise ValueError("telemetry_dropout must be in [0, 1]")
    rng = np.random.default_rng(seed)
    output = []
    for episode in episodes:
        events = []
        for event in episode.events:
            score = event.threat_probability + rng.normal(0, score_noise)
            if not event.attack_present:
                score += false_positive_shift
            anomaly = event.anomaly_score
            if rng.random() < telemetry_dropout:
                score, anomaly = 0.5, 0.5
            events.append(
                replace(
                    event,
                    threat_probability=float(np.clip(score, 0, 1)),
                    anomaly_score=float(np.clip(anomaly, 0, 1)),
                )
            )
        output.append(replace(episode, events=tuple(events)))
    return tuple(output)


def adversarial_noise_sweep(
    policy: Policy,
    episodes: tuple[IpsEpisode, ...],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    scenarios = (
        ("clean", 0.0, 0.0, 0.0),
        ("score_noise_0.10", 0.10, 0.0, 0.0),
        ("score_noise_0.25", 0.25, 0.0, 0.0),
        ("benign_shift_0.20", 0.0, 0.20, 0.0),
        ("telemetry_dropout_25pct", 0.0, 0.0, 0.25),
        ("combined_stress", 0.15, 0.20, 0.25),
    )
    rows = []
    for name, noise, shift, dropout in scenarios:
        perturbed = perturb_episodes(
            episodes,
            score_noise=noise,
            false_positive_shift=shift,
            telemetry_dropout=dropout,
            seed=seed,
        )
        _, _, summary = evaluate_detailed(policy, perturbed, seed=seed)
        rows.append(
            {
                "scenario": name,
                "score_noise": noise,
                "false_positive_shift": shift,
                "telemetry_dropout": dropout,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def profile_policy(
    policy: Policy,
    states: np.ndarray,
    masks: np.ndarray,
    *,
    repeats: int = 20,
) -> dict[str, float]:
    """Measure process CPU/RSS and per-decision latency/throughput."""
    if len(states) != len(masks) or not len(states):
        raise ValueError("states and masks must have equal non-zero length")
    latencies = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for _ in range(repeats):
        for state, mask in zip(states, masks):
            started = time.perf_counter_ns()
            policy(state, mask)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = usage.ru_maxrss / (1024**2 if __import__("sys").platform == "darwin" else 1024)
    values = np.asarray(latencies)
    return {
        "decisions": float(len(values)),
        "latency_mean_ms": float(values.mean()),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
        "throughput_decisions_s": len(values) / wall,
        "process_cpu_s": cpu,
        "cpu_utilization_one_core_pct": 100 * cpu / wall,
        "max_rss_mb": float(rss_mb),
        "rss_scope": "shared_process_peak_not_policy_attributable",
    }


def profile_policy_trials(
    policy: Policy,
    states: np.ndarray,
    masks: np.ndarray,
    *,
    trials: int = 5,
    repeats_per_trial: int = 20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Repeat profiling to expose timing variability instead of one measurement."""
    if trials < 2:
        raise ValueError("trials must be at least two")
    rows = [
        {"trial": trial + 1, **profile_policy(policy, states, masks, repeats=repeats_per_trial)}
        for trial in range(trials)
    ]
    frame = pd.DataFrame(rows)
    summary = {
        "profile_trials": float(trials),
        "latency_p50_mean_ms": float(frame.latency_p50_ms.mean()),
        "latency_p50_std_ms": float(frame.latency_p50_ms.std(ddof=1)),
        "latency_p95_mean_ms": float(frame.latency_p95_ms.mean()),
        "latency_p95_std_ms": float(frame.latency_p95_ms.std(ddof=1)),
        "throughput_mean_decisions_s": float(frame.throughput_decisions_s.mean()),
        "throughput_std_decisions_s": float(frame.throughput_decisions_s.std(ddof=1)),
        "shared_process_peak_rss_mb": float(frame.max_rss_mb.max()),
    }
    return frame, summary
