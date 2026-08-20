"""Minimal pytest plugin that emits authoritative structured outcome counts."""

from __future__ import annotations

import json
import os
from pathlib import Path

_state: dict[str, int | bool] = {}


def pytest_sessionstart(session) -> None:
    del session
    _state.clear()
    _state.update(
        discovered=0, selected=0, deselected=0, passed=0, failed=0,
        errors=0, skipped=0, xfailed=0, xpassed=0, interrupted=False,
        collection_errors=0,
    )


def pytest_deselected(items) -> None:
    _state["deselected"] = int(_state["deselected"]) + len(items)


def pytest_collection_finish(session) -> None:
    _state["selected"] = len(session.items)
    _state["discovered"] = len(session.items) + int(_state["deselected"])


def pytest_collectreport(report) -> None:
    if report.failed:
        _state["collection_errors"] = int(_state["collection_errors"]) + 1


def pytest_runtest_logreport(report) -> None:
    if report.when == "call":
        was_xfail = hasattr(report, "wasxfail")
        if was_xfail:
            key = "xfailed" if report.skipped else "xpassed"
        elif report.passed:
            key = "passed"
        elif report.skipped:
            key = "skipped"
        else:
            key = "failed"
        _state[key] = int(_state[key]) + 1
    elif report.skipped:
        _state["skipped"] = int(_state["skipped"]) + 1
    elif report.failed:
        _state["errors"] = int(_state["errors"]) + 1


def pytest_keyboard_interrupt(excinfo) -> None:
    del excinfo
    _state["interrupted"] = True


def pytest_sessionfinish(session, exitstatus) -> None:
    del session
    destination = os.environ.get("CODE_PROVENANCE_PYTEST_REPORT")
    if not destination:
        return
    payload = {
        "schema_version": "1.0",
        **_state,
        "exit_code": int(exitstatus),
        "complete": not bool(_state["interrupted"]) and int(_state["collection_errors"]) == 0,
    }
    Path(destination).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
