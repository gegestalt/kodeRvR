"""Static security signals for provenance-aware review triage.

Signals are review evidence, not proof of vulnerability. Package existence is
resolved through an injected registry function so tests and offline runs never
silently treat network failure as package hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


@dataclass(frozen=True)
class SecuritySignal:
    rule_id: str
    severity: str
    evidence: str
    line: int


_RULES = (
    ("PY-EVAL", "high", re.compile(r"\beval\s*\("), "dynamic eval"),
    ("PY-EXEC", "high", re.compile(r"\bexec\s*\("), "dynamic exec"),
    ("SHELL-TRUE", "high", re.compile(r"shell\s*=\s*True"), "shell execution"),
    ("SQL-FORMAT", "high", re.compile(r"(?:execute|query)\s*\(\s*f?[\"'].*\{"), "formatted SQL"),
    ("SECRET-LITERAL", "critical", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*[\"'][^\"']{8,}"), "credential-like literal"),
    ("TLS-VERIFY-OFF", "high", re.compile(r"verify\s*=\s*False"), "TLS verification disabled"),
    ("PICKLE-LOAD", "medium", re.compile(r"pickle\.loads?\s*\("), "unsafe deserialization surface"),
)


def scan_code(code: str) -> list[SecuritySignal]:
    output = []
    for rule_id, severity, pattern, evidence in _RULES:
        for match in pattern.finditer(code):
            output.append(SecuritySignal(rule_id, severity, evidence, code.count("\n", 0, match.start()) + 1))
    return output


def python_dependencies(code: str) -> set[str]:
    names = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", code, re.MULTILINE))
    return names


def package_risk(
    packages: set[str],
    resolver: Callable[[str], str],
) -> list[dict[str, str]]:
    """Classify package state; resolver returns exists/missing/unknown/new."""
    rows = []
    for package in sorted(packages):
        state = resolver(package)
        if state not in {"exists", "missing", "unknown", "new"}:
            raise ValueError("package resolver returned an unsupported state")
        risk = {"exists": "low", "new": "high", "missing": "critical", "unknown": "unknown"}[state]
        rows.append({"package": package, "registry_state": state, "supply_chain_risk": risk})
    return rows
