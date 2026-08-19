"""Seeded evaluation protocol shared by fixed and learned IPS policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ips.actions import IpsAction
from ips.environment import AdaptiveIpsEnv, IpsObservation

Policy = Callable[[IpsObservation, np.ndarray], IpsAction]


@dataclass(frozen=True)
class PolicyMetrics:
    episodes: int
    attack_episodes: int
    benign_episodes: int
    mean_return: float
    return_std: float
    containment_rate: float
    compromise_rate: float
    false_preventions_per_episode: float
    disruptive_actions_per_episode: float
    masked_actions_per_episode: float
    mean_steps: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_policy(
    policy: Policy,
    *,
    episodes: int = 200,
    seed: int = 42,
    attack_probability: float = 0.5,
    critical_probability: float = 0.25,
) -> PolicyMetrics:
    """Evaluate a policy on held-out, reproducible scenario seeds."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= attack_probability <= 1.0:
        raise ValueError("attack_probability must be in [0, 1]")
    if not 0.0 <= critical_probability <= 1.0:
        raise ValueError("critical_probability must be in [0, 1]")

    scenario_rng = np.random.default_rng(seed)
    returns: list[float] = []
    contained = compromised = false_preventions = disruptive = masked = total_steps = 0
    attack_episodes = 0
    disruptive_names = {
        "RATE_LIMIT", "DROP_FLOW", "TEMP_BLOCK_SOURCE",
        "BLOCK_DESTINATION_PORT", "ISOLATE_HOST",
    }

    for episode in range(episodes):
        attack_present = bool(scenario_rng.random() < attack_probability)
        attack_episodes += int(attack_present)
        critical = bool(scenario_rng.random() < critical_probability)
        env = AdaptiveIpsEnv(seed=seed + episode)
        observation, info = env.reset(
            seed=seed + episode,
            attack_present=attack_present,
            critical_service=critical,
        )
        episode_return = 0.0
        while True:
            action = policy(observation, info["action_mask"])
            result = env.step(action)
            episode_return += result.reward
            total_steps += 1
            contained += int(bool(result.info["contained"]))
            compromised += int(bool(result.info["compromised"]))
            masked += int(bool(result.info["action_was_masked"]))
            executed = str(result.info["executed_action"])
            disruptive += int(executed in disruptive_names)
            false_preventions += int(not attack_present and executed in disruptive_names)
            observation, info = result.observation, {"action_mask": result.info["action_mask"]}
            if result.terminated or result.truncated or result.info["contained"]:
                break
        returns.append(episode_return)

    values = np.asarray(returns, dtype=float)
    return PolicyMetrics(
        episodes=episodes,
        attack_episodes=attack_episodes,
        benign_episodes=episodes - attack_episodes,
        mean_return=float(values.mean()),
        return_std=float(values.std(ddof=0)),
        containment_rate=contained / attack_episodes if attack_episodes else 0.0,
        compromise_rate=compromised / attack_episodes if attack_episodes else 0.0,
        false_preventions_per_episode=false_preventions / episodes,
        disruptive_actions_per_episode=disruptive / episodes,
        masked_actions_per_episode=masked / episodes,
        mean_steps=total_steps / episodes,
    )
