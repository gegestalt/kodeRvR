"""Traceable structural relevance between changed symbols and pytest definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from code_provenance.change_context import ChangeContext
from code_provenance.dependency_context import DependencyContext, DependencyEdgeKind, DependencyNodeKind
from code_provenance.evidence import EvidenceTarget
from code_provenance.symbol_index import ChangedSymbolIndex
from code_provenance.test_evidence import TestEvidence


class RelevanceRelation(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class TestDefinition:
    node_id: str
    symbol_id: str
    path: str
    qualified_name: str
    start_line: int


@dataclass(frozen=True)
class RelevantTest:
    test: TestDefinition
    relation: RelevanceRelation
    dependency_distance: int | None
    related_symbols: tuple[str, ...]
    reasons: tuple[str, ...]
    observed_outcome: str


@dataclass(frozen=True)
class TestRelevanceContext:
    target: EvidenceTarget
    inventory: tuple[TestDefinition, ...]
    relevant_tests: tuple[RelevantTest, ...]
    partial: bool
    unresolved_reasons: tuple[str, ...]
    relevant_test_count: int
    direct_relevant_test_count: int
    indirect_relevant_test_count: int
    heuristic_relevant_test_count: int
    relevant_tests_observed_fraction: float


def _pytest_node(path: str, qualified: str) -> str:
    parts = [item for item in qualified.split(".") if item != "<locals>"]
    return "::".join((path, *parts))


def _tokens(value: str) -> set[str]:
    stop = {"test", "tests", "context", "build", "changed", "change", "python"}
    return {
        item.lower() for item in re.findall(r"[A-Za-z][A-Za-z0-9]+", value)
        if len(item) > 2 and item.lower() not in stop
    }


def build_test_relevance_context(
    change: ChangeContext,
    symbols: ChangedSymbolIndex,
    dependencies: DependencyContext,
    test_evidence: TestEvidence | None = None,
    *,
    max_distance: int = 4,
) -> TestRelevanceContext:
    if not (change.target == symbols.target == dependencies.target):
        raise ValueError("change, symbol, and dependency contexts must share one target")
    if test_evidence is not None and test_evidence.target != change.target:
        raise ValueError("test evidence target does not match relevance context")
    if max_distance < 1:
        raise ValueError("max_distance must be positive")
    test_nodes = [
        node for node in dependencies.nodes
        if node.kind is DependencyNodeKind.SYMBOL
        and node.path is not None
        and ("tests" in node.path.split("/") or node.path.rsplit("/", 1)[-1].startswith("test_"))
        and node.qualified_name.split(".")[-1].startswith("test")
    ]
    inventory = tuple(sorted((
        TestDefinition(
            _pytest_node(node.path or "", node.qualified_name), node.node_id,
            node.path or "", node.qualified_name, 0,
        ) for node in test_nodes
    ), key=lambda item: item.node_id))
    incoming: dict[str, set[str]] = {}
    for edge in dependencies.edges:
        if edge.resolved and edge.kind is DependencyEdgeKind.CALLS:
            incoming.setdefault(edge.target, set()).add(edge.source)
    changed = [(item.after or item.before) for item in symbols.changes]
    changed = [item for item in changed if item is not None]
    production_changed = [
        item for item in changed
        if "tests" not in item.path.split("/") and not item.path.rsplit("/", 1)[-1].startswith("test_")
    ]
    if production_changed:
        changed = production_changed
    distances: dict[str, tuple[int, set[str]]] = {}
    for changed_symbol in changed:
        frontier = {changed_symbol.symbol_id}
        visited = {changed_symbol.symbol_id}
        for distance in range(1, max_distance + 1):
            frontier = {source for target in frontier for source in incoming.get(target, set())} - visited
            if not frontier:
                break
            visited.update(frontier)
            for node_id in frontier:
                current = distances.get(node_id)
                if current is None or distance < current[0]:
                    distances[node_id] = (distance, {changed_symbol.qualified_name})
                elif distance == current[0]:
                    current[1].add(changed_symbol.qualified_name)
    observed = {item.node_id: item.outcome for item in test_evidence.test_cases} if test_evidence else {}
    changed_tokens = set().union(*(_tokens(item.qualified_name) for item in changed)) if changed else set()
    relevant = []
    for test in inventory:
        distance_info = distances.get(test.symbol_id)
        relation: RelevanceRelation | None = None
        related: tuple[str, ...] = ()
        reasons: tuple[str, ...] = ()
        distance: int | None = None
        if distance_info:
            distance, names = distance_info
            relation = RelevanceRelation.DIRECT if distance == 1 else RelevanceRelation.INDIRECT
            related = tuple(sorted(names))
            reasons = (f"static_call_path_distance:{distance}",)
        else:
            overlap = sorted(changed_tokens & _tokens(f"{test.path} {test.qualified_name}"))
            if overlap:
                relation = RelevanceRelation.HEURISTIC
                reasons = tuple(f"name_token:{item}" for item in overlap)
        if relation is None:
            continue
        outcome = observed.get(test.node_id, "not_observed")
        if outcome == "not_observed":
            parameterized = [value for key, value in observed.items() if key.startswith(test.node_id + "[")]
            if parameterized:
                outcome = parameterized[0] if len(set(parameterized)) == 1 else "mixed"
        relevant.append(RelevantTest(test, relation, distance, related, reasons, outcome))
    relevant = sorted(relevant, key=lambda item: (item.relation.value, item.test.node_id))
    observed_count = sum(item.observed_outcome != "not_observed" for item in relevant)
    unresolved_reasons = []
    if symbols.partial:
        unresolved_reasons.append("partial_symbol_context")
    if dependencies.unresolved:
        unresolved_reasons.append("unresolved_dependencies_present")
    if test_evidence is None or not test_evidence.complete:
        unresolved_reasons.append("complete_test_evidence_unavailable")
    return TestRelevanceContext(
        change.target, inventory, tuple(relevant), bool(unresolved_reasons),
        tuple(unresolved_reasons), len(relevant),
        sum(item.relation is RelevanceRelation.DIRECT for item in relevant),
        sum(item.relation is RelevanceRelation.INDIRECT for item in relevant),
        sum(item.relation is RelevanceRelation.HEURISTIC for item in relevant),
        observed_count / max(len(relevant), 1),
    )
