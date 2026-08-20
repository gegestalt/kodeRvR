from __future__ import annotations

from pathlib import Path
import subprocess

from code_provenance.change_context import build_change_context
from code_provenance.dependency_context import DependencyEdgeKind, build_dependency_context
from code_provenance.symbol_index import build_changed_symbol_index


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    files = {
        "pkg/repository.py": "def get(token):\n    return token\n",
        "pkg/decoder.py": "def decode(token):\n    return token\n",
        "pkg/service.py": (
            "from .repository import get\nfrom .decoder import decode as decode_jwt\n"
            "def validate(token):\n    return get(decode_jwt(token))\n"
        ),
        "pkg/controller.py": "from .service import validate\ndef login(token):\n    return validate(token)\n",
        "pkg/middleware.py": "from . import service\ndef authenticate(token):\n    return service.validate(token)\n",
        "pkg/__init__.py": "",
    }
    for path, code in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
    git(tmp_path, "add", "pkg")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "pkg/service.py").write_text(
        files["pkg/service.py"].replace("return get", "return bool(get" ).replace("(decode_jwt(token))", "(decode_jwt(token)))"),
        encoding="utf-8",
    )
    return tmp_path


def context(root: Path, depth: int = 4):
    change = build_change_context(root)
    symbols = build_changed_symbol_index(root, change)
    return build_dependency_context(root, change, symbols, max_depth=depth)


def test_imports_aliases_and_static_calls_are_resolved(tmp_path: Path):
    result = context(repository(tmp_path))
    kinds = {edge.kind for edge in result.edges}
    assert DependencyEdgeKind.IMPORTS in kinds
    assert DependencyEdgeKind.CALLS in kinds
    calls = {(result.node(edge.source).qualified_name, result.node(edge.target).qualified_name) for edge in result.edges if edge.kind is DependencyEdgeKind.CALLS and edge.resolved}
    assert ("validate", "get") in calls
    assert ("validate", "decode") in calls
    assert ("login", "validate") in calls
    assert ("authenticate", "validate") in calls


def test_reverse_and_bounded_transitive_dependents_describe_blast_radius(tmp_path: Path):
    result = context(repository(tmp_path), depth=3)
    impact = next(item for item in result.impacts if item.qualified_name == "validate")
    assert impact.direct_dependent_count == 2
    assert impact.transitive_dependent_count >= 2
    assert impact.dependency_depth <= 3
    assert impact.affected_module_count >= 2


def test_unresolved_dynamic_and_unknown_calls_remain_visible(tmp_path: Path):
    root = repository(tmp_path)
    with (root / "pkg/service.py").open("a", encoding="utf-8") as stream:
        stream.write("\ndef dynamic(name):\n    return unknown(name)\n")
    result = context(root)
    assert any("unknown" in item.expression for item in result.unresolved)


def test_cycles_terminate_and_results_are_deterministic(tmp_path: Path):
    root = repository(tmp_path)
    (root / "pkg/a.py").write_text("from .b import b\ndef a(): return b()\n", encoding="utf-8")
    (root / "pkg/b.py").write_text("from .a import a\ndef b(): return a()\n", encoding="utf-8")
    git(root, "add", "pkg/a.py", "pkg/b.py")
    first = context(root, depth=10)
    second = context(root, depth=10)
    assert first == second
    assert all(item.dependency_depth <= 10 for item in first.impacts)


def test_context_is_target_bound_and_non_python_files_are_ignored(tmp_path: Path):
    root = repository(tmp_path)
    (root / "README.md").write_text("service.validate()", encoding="utf-8")
    change = build_change_context(root)
    symbols = build_changed_symbol_index(root, change)
    result = build_dependency_context(root, change, symbols)
    assert result.target == change.target == symbols.target
    assert all(node.path is None or node.path.endswith(".py") for node in result.nodes)


def test_self_cls_calls_and_same_names_resolve_with_class_qualification(tmp_path: Path):
    root = repository(tmp_path)
    (root / "pkg/classes.py").write_text(
        "class A:\n"
        "    def helper(self): return 1\n"
        "    def run(self): return self.helper()\n"
        "class B:\n"
        "    @classmethod\n    def helper(cls): return 2\n"
        "    @classmethod\n    def run(cls): return cls.helper()\n",
        encoding="utf-8",
    )
    git(root, "add", "pkg/classes.py")
    result = context(root)
    calls = {(result.node(edge.source).qualified_name, result.node(edge.target).qualified_name) for edge in result.edges if edge.kind is DependencyEdgeKind.CALLS and edge.resolved}
    assert ("A.run", "A.helper") in calls
    assert ("B.run", "B.helper") in calls


def test_star_and_type_checking_import_limitations_are_visible(tmp_path: Path):
    root = repository(tmp_path)
    (root / "pkg/conditional.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from .service import validate as typed_validate\n"
        "from .decoder import *\n"
        "def use(value): return typed_validate(value)\n",
        encoding="utf-8",
    )
    git(root, "add", "pkg/conditional.py")
    result = context(root)
    assert any("*" in item.expression for item in result.unresolved)
    assert any(edge.expression == "typed_validate" and edge.resolved for edge in result.edges)
