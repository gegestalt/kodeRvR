"""Defensive actions and safety constraints for the adaptive IPS."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class IpsAction(IntEnum):
    """Discrete actions available to an IPS policy."""

    ALLOW = 0
    MONITOR = 1
    RATE_LIMIT = 2
    DROP_FLOW = 3
    TEMP_BLOCK_SOURCE = 4
    BLOCK_DESTINATION_PORT = 5
    ISOLATE_HOST = 6


def valid_action_mask(
    threat_probability: float,
    critical_service: bool,
    host_compromise: float,
) -> np.ndarray:
    """Return a boolean mask that prevents unjustified disruptive actions.

    The mask is a hard safety layer, independent of what an RL policy proposes.
    Critical hosts require stronger evidence before port blocking or isolation.
    """
    if not 0.0 <= threat_probability <= 1.0:
        raise ValueError("threat_probability must be in [0, 1]")
    if not 0.0 <= host_compromise <= 1.0:
        raise ValueError("host_compromise must be in [0, 1]")

    mask = np.ones(len(IpsAction), dtype=bool)
    mask[IpsAction.TEMP_BLOCK_SOURCE] = threat_probability >= 0.60
    mask[IpsAction.BLOCK_DESTINATION_PORT] = (
        threat_probability >= (0.90 if critical_service else 0.70)
    )
    mask[IpsAction.ISOLATE_HOST] = (
        host_compromise >= (0.90 if critical_service else 0.70)
        and threat_probability >= 0.80
    )
    return mask


def enforce_action_mask(action: IpsAction, mask: np.ndarray) -> IpsAction:
    """Fail closed to MONITOR if a policy proposes an unsafe action."""
    if mask.shape != (len(IpsAction),):
        raise ValueError(f"mask must have shape ({len(IpsAction)},)")
    return action if bool(mask[int(action)]) else IpsAction.MONITOR
