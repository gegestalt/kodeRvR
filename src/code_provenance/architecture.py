"""Conservative Python dependency evidence for patch-health review.

This analyzer reports facts it can establish statically: local import edges,
dependency cycles, and parse failures. It does not convert style preferences
into architecture defects.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ArchitectureStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArchitectureSignal:
    rule_id: str
    severity: str
    evidence: str
    path: str


@dataclass(frozen=True)
class ArchitectureReport:
    status: ArchitectureStatus
    confidence: float
    modules_analyzed: int
    dependency_edges: int
    cycles: tuple[tuple[str, ...], ...]
    signals: tuple[ArchitectureSignal, ...]


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _python_files(root: Path) -> list[Path]:
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    return sorted(
        path for path in root.rglob("*.py")
        if not any(part in excluded for part in path.relative_to(root).parts)
        and not path.relative_to(root).parts[0] in {"tests", "test"}
    )


def _local_imports(
    tree: ast.AST,
    known: set[str],
    current_module: str,
    *,
    is_package: bool,
) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = current_module.split(".") if is_package else current_module.split(".")[:-1]
                climb = node.level - 1
                package = package[:max(0, len(package) - climb)]
                base = ".".join((*package, *(node.module or "").split("."))).strip(".")
            else:
                base = node.module or ""
            if base in known:
                imports.add(base)
            for alias in node.names:
                candidate = ".".join(part for part in (base, alias.name) if part)
                if candidate in known:
                    imports.add(candidate)
    return imports


def _strongly_connected(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = lowlinks[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in active:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            members: list[str] = []
            while True:
                member = stack.pop()
                active.remove(member)
                members.append(member)
                if member == node:
                    break
            if len(members) > 1 or node in graph[node]:
                cycles.append(tuple(sorted(members)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(sorted(set(cycles)))


def analyze_python_architecture(root: Path) -> ArchitectureReport:
    """Analyze local Python dependencies without importing repository code."""
    root = root.resolve()
    files = _python_files(root)
    module_paths = {
        _module_name(root, path): path
        for path in files
        if _module_name(root, path)
    }
    known = set(module_paths)
    parsed: dict[str, ast.AST] = {}
    signals: list[ArchitectureSignal] = []
    for module, path in module_paths.items():
        try:
            parsed[module] = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError) as error:
            signals.append(ArchitectureSignal(
                "ARCH-PARSE",
                "high",
                f"Cannot establish dependency evidence: {error.msg if isinstance(error, SyntaxError) else error}",
                str(path.relative_to(root)),
            ))

    graph = {
        module: _local_imports(
            tree,
            known,
            module,
            is_package=module_paths[module].name == "__init__.py",
        )
        for module, tree in parsed.items()
    }
    for module in known - set(graph):
        graph[module] = set()
    cycles = _strongly_connected(graph)
    for cycle in cycles:
        signals.append(ArchitectureSignal(
            "ARCH-CYCLE",
            "high",
            "Local dependency cycle: " + " → ".join((*cycle, cycle[0])),
            ",".join(str(module_paths[item].relative_to(root)) for item in cycle),
        ))

    if any(signal.rule_id == "ARCH-PARSE" for signal in signals):
        status, confidence = ArchitectureStatus.UNKNOWN, 0.0
    elif cycles:
        status, confidence = ArchitectureStatus.FAIL, 1.0
    else:
        status, confidence = ArchitectureStatus.PASS, 1.0
    return ArchitectureReport(
        status=status,
        confidence=confidence,
        modules_analyzed=len(module_paths),
        dependency_edges=sum(len(targets) for targets in graph.values()),
        cycles=cycles,
        signals=tuple(signals),
    )
