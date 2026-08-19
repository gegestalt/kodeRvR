"""IPS environment replaying observed dataset episodes.

Traffic observations are dataset-backed. Prevention outcomes remain a seeded,
auditable counterfactual model until equivalent cyber-range interventions exist.
"""

from __future__ import annotations

import numpy as np

from ips.actions import IpsAction, enforce_action_mask, valid_action_mask
from ips.dataset import IpsEpisode
from ips.environment import AdaptiveIpsEnv, IpsObservation, StepResult
from ips.reward import RewardConfig, calculate_reward


class DatasetBackedIpsEnv:
    """Replay one observed episode while applying counterfactual IPS actions."""

    def __init__(
        self,
        episode: IpsEpisode,
        *,
        initial_budget: int = 20,
        reward_config: RewardConfig | None = None,
        seed: int = 42,
    ) -> None:
        if not episode.events:
            raise ValueError("episode must contain at least one event")
        if initial_budget < 1:
            raise ValueError("initial_budget must be positive")
        self.episode = episode
        self.initial_budget = initial_budget
        self.reward_config = reward_config or RewardConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._budget = initial_budget
        self._finished = False

    def reset(self, *, seed: int | None = None) -> tuple[IpsObservation, dict[str, object]]:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._index = 0
        self._budget = self.initial_budget
        self._finished = False
        return self._observation(), self._info()

    def action_mask(self) -> np.ndarray:
        event = self.episode.events[self._index]
        return valid_action_mask(
            event.threat_probability,
            event.critical_service,
            event.attack_stage,
        )

    def step(self, proposed_action: int | IpsAction) -> StepResult:
        if self._finished:
            raise RuntimeError("episode is finished; call reset()")
        try:
            proposed = IpsAction(int(proposed_action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid IPS action: {proposed_action!r}") from exc
        action = enforce_action_mask(proposed, self.action_mask())
        event = self.episode.events[self._index]
        contained = bool(
            event.attack_present
            and self._rng.random() < AdaptiveIpsEnv.ACTION_EFFECTIVENESS[action]
        )
        compromised = bool(
            event.attack_present and not contained and event.attack_stage >= 1.0
        )
        if action not in {IpsAction.ALLOW, IpsAction.MONITOR}:
            self._budget = max(0, self._budget - 1)
        reward = calculate_reward(
            action=action,
            attack_present=event.attack_present,
            contained=contained,
            compromised=compromised,
            critical_service=event.critical_service,
            config=self.reward_config,
        )

        at_end = self._index == len(self.episode.events) - 1
        if not at_end and not contained and not compromised and self._budget > 0:
            self._index += 1
        self._finished = contained or compromised or at_end or self._budget == 0
        info = self._info()
        info.update(
            {
                "proposed_action": proposed.name,
                "executed_action": action.name,
                "action_was_masked": action != proposed,
                "contained": contained,
                "compromised": compromised,
                "episode_id": self.episode.episode_id,
                "group_id": self.episode.group_id,
                "attack_family": event.attack_family,
                "dataset_backed_observation": True,
                "counterfactual_outcome": True,
            }
        )
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=contained or compromised,
            truncated=self._finished and not (contained or compromised),
            info=info,
        )

    def _observation(self) -> IpsObservation:
        event = self.episode.events[self._index]
        return IpsObservation(
            threat_probability=event.threat_probability,
            anomaly_score=event.anomaly_score,
            attack_stage=event.attack_stage,
            host_compromise=event.attack_stage,
            critical_service=float(event.critical_service),
            recent_attack_rate=float(event.attack_present),
            response_budget=self._budget / self.initial_budget,
        )

    def _info(self) -> dict[str, object]:
        return {"action_mask": self.action_mask()}
