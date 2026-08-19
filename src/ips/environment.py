"""Dependency-light, Gym-like environment for adaptive IPS research.

This is a controlled defensive simulator, not a live packet-control system.
Detector scores are observations; actions alter attack progression and service
availability. The held-out evaluation scenarios must never tune the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ips.actions import IpsAction, enforce_action_mask, valid_action_mask
from ips.belief import observable_belief
from ips.reward import RewardConfig, calculate_reward


@dataclass(frozen=True)
class IpsObservation:
    """Policy-visible belief state at one defensive decision point.

    ``attack_stage`` and ``host_compromise`` retain their stable API names but
    are detector-history estimates, never environment ground truth.
    """

    threat_probability: float
    anomaly_score: float
    attack_stage: float
    host_compromise: float
    critical_service: float
    recent_attack_rate: float
    response_budget: float

    def as_array(self) -> np.ndarray:
        return np.asarray(tuple(self.__dict__.values()), dtype=np.float32)


@dataclass(frozen=True)
class StepResult:
    observation: IpsObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class AdaptiveIpsEnv:
    """Finite-horizon IPS simulator with deterministic seeded transitions."""

    ACTION_EFFECTIVENESS = {
        IpsAction.ALLOW: 0.00,
        IpsAction.MONITOR: 0.05,
        IpsAction.RATE_LIMIT: 0.35,
        IpsAction.DROP_FLOW: 0.55,
        IpsAction.TEMP_BLOCK_SOURCE: 0.75,
        IpsAction.BLOCK_DESTINATION_PORT: 0.85,
        IpsAction.ISOLATE_HOST: 0.98,
    }

    def __init__(
        self,
        *,
        max_steps: int = 50,
        initial_budget: int = 20,
        reward_config: RewardConfig | None = None,
        seed: int = 42,
    ) -> None:
        if max_steps < 1 or initial_budget < 1:
            raise ValueError("max_steps and initial_budget must be positive")
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.reward_config = reward_config or RewardConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._budget = initial_budget
        self._attack_present = False
        self._stage = 0
        self._critical = False
        self._threat_probability = 0.0
        self._anomaly_score = 0.0
        self._score_history: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        attack_present: bool | None = None,
        critical_service: bool | None = None,
    ) -> tuple[IpsObservation, dict[str, object]]:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._step = 0
        self._budget = self.initial_budget
        self._attack_present = (
            bool(self._rng.random() < 0.50) if attack_present is None else attack_present
        )
        self._critical = (
            bool(self._rng.random() < 0.25) if critical_service is None else critical_service
        )
        self._stage = 1 if self._attack_present else 0
        self._refresh_detector_scores()
        self._score_history = [self._threat_probability]
        return self._observation(), {"action_mask": self.action_mask()}

    def action_mask(self) -> np.ndarray:
        _, compromise, _ = observable_belief(
            self._threat_probability, self._anomaly_score, self._score_history
        )
        return valid_action_mask(
            self._threat_probability,
            self._critical,
            compromise,
        )

    def step(self, proposed_action: int | IpsAction) -> StepResult:
        if self._step >= self.max_steps or self._stage >= 4:
            raise RuntimeError("episode is finished; call reset()")
        try:
            proposed = IpsAction(int(proposed_action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid IPS action: {proposed_action!r}") from exc

        action = enforce_action_mask(proposed, self.action_mask())
        contained = False
        compromised = False
        if self._attack_present:
            effectiveness = self.ACTION_EFFECTIVENESS[action]
            contained = bool(self._rng.random() < effectiveness)
            if contained:
                self._attack_present = False
                self._stage = 0
            elif self._rng.random() < 0.55:
                self._stage = min(4, self._stage + 1)
                compromised = self._stage == 4

        if action not in {IpsAction.ALLOW, IpsAction.MONITOR}:
            self._budget = max(0, self._budget - 1)

        reward = calculate_reward(
            action=action,
            attack_present=(contained or self._attack_present or compromised),
            contained=contained,
            compromised=compromised,
            critical_service=self._critical,
            config=self.reward_config,
        )
        self._step += 1
        self._refresh_detector_scores()
        self._score_history.append(self._threat_probability)
        terminated = compromised
        truncated = self._step >= self.max_steps or self._budget == 0
        info = {
            "proposed_action": proposed.name,
            "executed_action": action.name,
            "action_was_masked": action != proposed,
            "contained": contained,
            "compromised": compromised,
            "action_mask": self.action_mask(),
        }
        return StepResult(self._observation(), reward, terminated, truncated, info)

    def _refresh_detector_scores(self) -> None:
        centre = 0.12 if not self._attack_present else 0.45 + 0.12 * self._stage
        self._threat_probability = float(np.clip(self._rng.normal(centre, 0.08), 0, 1))
        self._anomaly_score = float(
            np.clip(self._rng.normal(0.20 if not self._attack_present else 0.75, 0.10), 0, 1)
        )

    def _observation(self) -> IpsObservation:
        stage, compromise, recent = observable_belief(
            self._threat_probability, self._anomaly_score, self._score_history
        )
        return IpsObservation(
            threat_probability=self._threat_probability,
            anomaly_score=self._anomaly_score,
            attack_stage=stage,
            host_compromise=compromise,
            critical_service=float(self._critical),
            recent_attack_rate=recent,
            response_budget=self._budget / self.initial_budget,
        )
