"""Non-learning IPS policies required as honest RL baselines."""

from __future__ import annotations

import numpy as np

from ips.actions import IpsAction
from ips.environment import IpsObservation


def rule_based_policy(observation: IpsObservation, action_mask: np.ndarray) -> IpsAction:
    """Select the least disruptive valid action for the estimated threat."""
    p = observation.threat_probability
    candidates = (
        (0.90, IpsAction.ISOLATE_HOST),
        (0.80, IpsAction.TEMP_BLOCK_SOURCE),
        (0.65, IpsAction.DROP_FLOW),
        (0.45, IpsAction.RATE_LIMIT),
        (0.25, IpsAction.MONITOR),
    )
    for threshold, action in candidates:
        if p >= threshold and action_mask[int(action)]:
            return action
    return IpsAction.ALLOW


def allow_policy(observation: IpsObservation, action_mask: np.ndarray) -> IpsAction:
    """No-prevention lower baseline."""
    del observation, action_mask
    return IpsAction.ALLOW


def aggressive_policy(observation: IpsObservation, action_mask: np.ndarray) -> IpsAction:
    """Most disruptive valid action, used to expose availability costs."""
    del observation
    for action in reversed(list(IpsAction)):
        if action_mask[int(action)]:
            return action
    return IpsAction.ALLOW
