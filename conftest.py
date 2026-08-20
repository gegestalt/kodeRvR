"""Project-wide test tiers: network access is always explicit."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("external data")
    group.addoption("--run-live-data", action="store_true", help="enable network-backed live-data tests")
    group.addoption("--live-data-limit", type=int, default=3, help="maximum records fetched per live dataset")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    expression = config.option.markexpr or ""
    enabled = config.getoption("--run-live-data") or "live_data" in expression
    if enabled:
        return
    skip = pytest.mark.skip(reason="live data disabled; use --run-live-data or -m live_data")
    for item in items:
        if "live_data" in item.keywords:
            item.add_marker(skip)
