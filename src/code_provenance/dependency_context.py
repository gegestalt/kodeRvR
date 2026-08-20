"""Conservative, traceable Python import/call dependency context."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
import subprocess

from code_provenance.change_context import ChangeContext
from code_provenance.evidence import EvidenceTarget
from code_provenance.symbol_index import ChangedSymbolIndex, PythonSymbol, index_python_symbols


class DependencyNodeKind(StrEnum):
    MODULE = "module"
    SYMBOL = "symbol"
    UNRESOLVED = "unresolved"


class DependencyEdgeKind(StrEnum):
    IMPORTS = "imports"
    CALLS = "calls"
    DECORATES = "decorates"
    DYNAMIC_IMPORT = "dynamic_import"


@dataclass(frozen=True)
class DependencyNode:
    node_id: str
    kind: DependencyNodeKind
    module: str
    path: str | None
    qualified_name: str


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    kind: DependencyEdgeKind
    expression: str
    resolved: bool


@dataclass(frozen=True)
class UnresolvedDependency:
    source: str
    expression: str
    reason: str


@dataclass(frozen=True)
class DependencyImpact:
    symbol_id: str
    qualified_name: str
    direct_dependencies: tuple[str, ...]
    direct_dependents: tuple[str, ...]
    transitive_dependents: tuple[str, ...]
    direct_dependent_count: int
    transitive_dependent_count: int
    dependency_depth: int
    affected_module_count: int
    unresolved_references: tuple[str, ...]


@dataclass(frozen=True)
class DependencyContext:
    target: EvidenceTarget
    nodes: tuple[DependencyNode, ...]
    edges: tuple[DependencyEdge, ...]
    unresolved: tuple[UnresolvedDependency, ...]
    impacts: tuple[DependencyImpact, ...]
    max_depth: int

    def node(self, node_id: str) -> DependencyNode:
        return next(item for item in self.nodes if item.node_id == node_id)


def _id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def _module(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _relative(current: str, module: str | None, level: int) -> str:
    package = current if current.endswith(".__init__") else current.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level > 1:
        parts = parts[:max(0, len(parts) - level + 1)]
    if module:
        parts.extend(module.split("."))
    return ".".join(filter(None, parts))


def _source_symbol(symbols: tuple[PythonSymbol, ...], line: int) -> PythonSymbol | None:
    matches = [item for item in symbols if item.start_line <= line <= item.end_line]
    return min(matches, key=lambda item: item.end_line - item.start_line) if matches else None


def _tracked_python(root: Path) -> tuple[str, ...]:
    output = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
        check=True, capture_output=True,
    ).stdout
    return tuple(sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0") if item and item.endswith(b".py")
    ))


def build_dependency_context(
    root: Path,
    change: ChangeContext,
    changed_symbols: ChangedSymbolIndex,
    *,
    max_depth: int = 4,
) -> DependencyContext:
    """Build conservative static facts; graph degree is never interpreted as risk."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if change.target != changed_symbols.target:
        raise ValueError("change and symbol contexts target different snapshots")
    root = root.resolve()
    nodes: dict[str, DependencyNode] = {}
    symbols_by_module: dict[str, tuple[PythonSymbol, ...]] = {}
    trees: dict[str, ast.Module] = {}
    paths: dict[str, str] = {}
    symbol_lookup: dict[tuple[str, str], PythonSymbol] = {}

    for path in _tracked_python(root):
        candidate = root / path
        try:
            code = candidate.read_text(encoding="utf-8")
            tree = ast.parse(code)
            symbols = index_python_symbols(path, code)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            continue
        module = _module(path)
        paths[module] = path
        trees[module] = tree
        symbols_by_module[module] = symbols
        module_id = _id("module", module)
        nodes[module_id] = DependencyNode(module_id, DependencyNodeKind.MODULE, module, path, module)
        for symbol in symbols:
            nodes[symbol.symbol_id] = DependencyNode(
                symbol.symbol_id, DependencyNodeKind.SYMBOL, module, path, symbol.qualified_name
            )
            symbol_lookup[(module, symbol.qualified_name)] = symbol

    edges: set[DependencyEdge] = set()
    unresolved: set[UnresolvedDependency] = set()

    def unresolved_edge(source: str, expression: str, kind: DependencyEdgeKind, reason: str) -> None:
        target = _id("unresolved", expression)
        nodes.setdefault(target, DependencyNode(target, DependencyNodeKind.UNRESOLVED, "", None, expression))
        edges.add(DependencyEdge(source, target, kind, expression, False))
        unresolved.add(UnresolvedDependency(source, expression, reason))

    for module, tree in trees.items():
        module_id = _id("module", module)
        aliases: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
                    target = _id("module", alias.name)
                    if target in nodes:
                        edges.add(DependencyEdge(module_id, target, DependencyEdgeKind.IMPORTS, alias.name, True))
                    else:
                        unresolved_edge(module_id, alias.name, DependencyEdgeKind.IMPORTS, "external_or_missing_module")
            elif isinstance(node, ast.ImportFrom):
                imported_module = _relative(module, node.module, node.level) if node.level else (node.module or "")
                for alias in node.names:
                    if alias.name == "*":
                        target = _id("module", imported_module)
                        if target in nodes:
                            edges.add(DependencyEdge(module_id, target, DependencyEdgeKind.IMPORTS, ast.unparse(node), True))
                        unresolved_edge(module_id, ast.unparse(node), DependencyEdgeKind.IMPORTS, "star_import_names_unresolved")
                        continue
                    possible_module = f"{imported_module}.{alias.name}" if imported_module else alias.name
                    if possible_module in paths:
                        aliases[alias.asname or alias.name] = (possible_module, None)
                        target = _id("module", possible_module)
                    else:
                        aliases[alias.asname or alias.name] = (imported_module, alias.name)
                        symbol = symbol_lookup.get((imported_module, alias.name))
                        target = symbol.symbol_id if symbol else _id("module", imported_module)
                    if target in nodes:
                        edges.add(DependencyEdge(module_id, target, DependencyEdgeKind.IMPORTS, ast.unparse(node), True))
                    else:
                        unresolved_edge(module_id, ast.unparse(node), DependencyEdgeKind.IMPORTS, "unresolved_import")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            owner = _source_symbol(symbols_by_module[module], node.lineno)
            source = owner.symbol_id if owner else module_id
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    expression = ast.unparse(decorator)
                    name = expression.split("(", 1)[0]
                    alias = aliases.get(name)
                    if alias and alias[1] and (target_symbol := symbol_lookup.get(alias)):
                        edges.add(DependencyEdge(source, target_symbol.symbol_id, DependencyEdgeKind.DECORATES, expression, True))
                    elif name not in {"staticmethod", "classmethod", "property"}:
                        unresolved_edge(source, expression, DependencyEdgeKind.DECORATES, "unresolved_decorator")
                continue
            expression = ast.unparse(node.func)
            if expression in {"__import__", "importlib.import_module"}:
                unresolved_edge(source, ast.unparse(node), DependencyEdgeKind.DYNAMIC_IMPORT, "dynamic_import")
                continue
            target_symbol: PythonSymbol | None = None
            if isinstance(node.func, ast.Name):
                alias = aliases.get(node.func.id)
                if alias and alias[1]:
                    target_symbol = symbol_lookup.get(alias)
                else:
                    target_symbol = symbol_lookup.get((module, node.func.id))
                if target_symbol is None and node.func.id in dir(builtins):
                    continue
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                root_name = node.func.value.id
                alias = aliases.get(root_name)
                if alias:
                    target_symbol = symbol_lookup.get((alias[0], node.func.attr))
                elif root_name in {"self", "cls"} and owner and "." in owner.qualified_name:
                    class_name = owner.qualified_name.rsplit(".", 1)[0]
                    target_symbol = symbol_lookup.get((module, f"{class_name}.{node.func.attr}"))
            if target_symbol:
                edges.add(DependencyEdge(source, target_symbol.symbol_id, DependencyEdgeKind.CALLS, expression, True))
            else:
                unresolved_edge(source, expression, DependencyEdgeKind.CALLS, "conservative_static_resolution_failed")

    ordered_edges = tuple(sorted(edges, key=lambda item: (item.source, item.target, item.kind.value, item.expression)))
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for edge in ordered_edges:
        if edge.resolved and edge.kind in {DependencyEdgeKind.CALLS, DependencyEdgeKind.DECORATES}:
            outgoing.setdefault(edge.source, set()).add(edge.target)
            incoming.setdefault(edge.target, set()).add(edge.source)

    impacts = []
    for changed in changed_symbols.changes:
        symbol = changed.after or changed.before
        if symbol is None:
            continue
        direct = set(incoming.get(symbol.symbol_id, set()))
        visited = set(direct)
        frontier = set(direct)
        depth_reached = 1 if frontier else 0
        for depth in range(2, max_depth + 1):
            next_frontier = {parent for item in frontier for parent in incoming.get(item, set())} - visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
            depth_reached = depth
        affected_modules = {nodes[item].module for item in visited if item in nodes and nodes[item].module}
        unresolved_refs = tuple(sorted(
            item.expression for item in unresolved if item.source == symbol.symbol_id
        ))
        impacts.append(DependencyImpact(
            symbol.symbol_id, symbol.qualified_name,
            tuple(sorted(outgoing.get(symbol.symbol_id, set()))), tuple(sorted(direct)),
            tuple(sorted(visited)), len(direct), len(visited), depth_reached,
            len(affected_modules), unresolved_refs,
        ))
    return DependencyContext(
        change.target,
        tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
        ordered_edges,
        tuple(sorted(unresolved, key=lambda item: (item.source, item.expression, item.reason))),
        tuple(sorted(impacts, key=lambda item: (item.qualified_name, item.symbol_id))),
        max_depth,
    )
