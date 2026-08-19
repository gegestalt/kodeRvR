"""Auditable reward model for balancing containment and availability."""

from __future__ import annotations

from dataclasses import dataclass

from ips.actions import IpsAction


@dataclass(frozen=True)
class RewardConfig:
    """Security and availability costs used by the IPS environment."""

    contained_attack: float = 8.0
    successful_compromise: float = -25.0
    false_prevention: float = -4.0
    critical_service_disruption: float = -12.0
    unnecessary_action: float = -0.25
    monitoring_cost: float = -0.05
    time_under_attack: float = -0.50


DISRUPTIVE_ACTIONS = frozenset(
    {
        IpsAction.RATE_LIMIT,
        IpsAction.DROP_FLOW,
        IpsAction.TEMP_BLOCK_SOURCE,
        IpsAction.BLOCK_DESTINATION_PORT,
        IpsAction.ISOLATE_HOST,
    }
)


def calculate_reward(
    *,
    action: IpsAction,
    attack_present: bool,
    contained: bool,
    compromised: bool,
    critical_service: bool,
    config: RewardConfig,
) -> float:
    """Calculate one transition reward from explicit, inspectable terms."""
    reward = 0.0
    if contained:
        reward += config.contained_attack
    if compromised:
        reward += config.successful_compromise
    if attack_present and not contained and not compromised:
        reward += config.time_under_attack
    if action == IpsAction.MONITOR:
        reward += config.monitoring_cost
    if action in DISRUPTIVE_ACTIONS:
        if not attack_present:
            reward += config.false_prevention
        if critical_service and action in {
            IpsAction.BLOCK_DESTINATION_PORT,
            IpsAction.ISOLATE_HOST,
        }:
            reward += config.critical_service_disruption
    elif not attack_present and action != IpsAction.ALLOW:
        reward += config.unnecessary_action
    return float(reward)
