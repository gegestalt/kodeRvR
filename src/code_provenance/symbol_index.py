"""Deterministic Python symbol identities and syntactic base/head change classification."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
import subprocess

from code_provenance.change_context import ChangeContext, ChangedHunk
from code_provenance.evidence import EvidenceTarget


class SymbolKind(StrEnum):
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    NESTED_FUNCTION = "nested_function"


class SymbolChange(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(frozen=True)
class PythonSymbol:
    symbol_id: str
    path: str
    qualified_name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    potentially_public: bool
    decorators: tuple[str, ...]
    signature_hash: str
    body_hash: str


@dataclass(frozen=True)
class ChangedSymbol:
    qualified_name: str
    change: SymbolChange
    before: PythonSymbol | None
    after: PythonSymbol | None
    signature_changed: bool
    body_changed: bool
    affected_hunks: tuple[str, ...]


@dataclass(frozen=True)
class ChangedSymbolIndex:
    target: EvidenceTarget
    changes: tuple[ChangedSymbol, ...]
    partial: bool
    parse_failures: tuple[str, ...]
    unsupported_cases: tuple[str, ...]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node_hash(value: object) -> str:
    if isinstance(value, list):
        value = ast.Module(body=value, type_ignores=[])
    return _hash(ast.dump(value, annotate_fields=True, include_attributes=False))


def _kind(node: ast.AST, parents: tuple[ast.AST, ...]) -> SymbolKind:
    async_node = isinstance(node, ast.AsyncFunctionDef)
    if isinstance(node, ast.ClassDef):
        return SymbolKind.CLASS
    if parents and isinstance(parents[-1], ast.ClassDef):
        return SymbolKind.ASYNC_METHOD if async_node else SymbolKind.METHOD
    if any(isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) for parent in parents):
        return SymbolKind.NESTED_FUNCTION
    return SymbolKind.ASYNC_FUNCTION if async_node else SymbolKind.FUNCTION


def _qualname(name: str, parents: tuple[ast.AST, ...]) -> str:
    parts: list[str] = []
    for parent in parents:
        if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(parent.name)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append("<locals>")
    return ".".join((*parts, name))


def _public(name: str, parents: tuple[ast.AST, ...]) -> bool:
    if name.startswith("_"):
        return False
    if any(isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) for parent in parents):
        return False
    return all(not getattr(parent, "name", "").startswith("_") for parent in parents)


def _signature(node: ast.AST) -> object:
    if isinstance(node, ast.ClassDef):
        return ast.Tuple(elts=[*node.bases, *node.keywords, *node.decorator_list], ctx=ast.Load())
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return ast.Tuple(
        elts=[node.args, node.returns or ast.Constant(None), *node.decorator_list,
              ast.Constant(isinstance(node, ast.AsyncFunctionDef))],
        ctx=ast.Load(),
    )


def _symbols(path: str, code: str) -> tuple[PythonSymbol, ...]:
    tree = ast.parse(code)
    output: list[PythonSymbol] = []

    def visit(body: list[ast.stmt], parents: tuple[ast.AST, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = _qualname(node.name, parents)
                kind = _kind(node, parents)
                identity = _hash("\0".join((path, qualified, kind.value)))[:24]
                decorators = tuple(ast.unparse(item) for item in node.decorator_list)
                start = min((item.lineno for item in node.decorator_list), default=node.lineno)
                output.append(PythonSymbol(
                    identity, path, qualified, kind, start,
                    int(getattr(node, "end_lineno", node.lineno)), _public(node.name, parents),
                    decorators, _node_hash(_signature(node)), _node_hash(node.body),
                ))
                visit(node.body, (*parents, node))
            else:
                nested_bodies = [value for _, value in ast.iter_fields(node) if isinstance(value, list)]
                for nested in nested_bodies:
                    if all(isinstance(item, ast.stmt) for item in nested):
                        visit(nested, parents)
    visit(tree.body, ())
    return tuple(sorted(output, key=lambda item: (item.path, item.qualified_name, item.kind.value)))


def _git_show(root: Path, revision: str, path: str) -> str | None:
    process = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return process.stdout if process.returncode == 0 else None


def _head_code(root: Path, context: ChangeContext, path: str) -> str | None:
    if context.base_sha is not None:
        return _git_show(root, context.head_sha, path)
    candidate = root / path
    if not candidate.exists() or candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _hunk_refs(symbol: PythonSymbol | None, hunks: tuple[ChangedHunk, ...], *, before: bool) -> tuple[str, ...]:
    if symbol is None:
        return ()
    refs = []
    for hunk in hunks:
        if hunk.path != symbol.path:
            continue
        start = hunk.old_start if before else hunk.new_start
        count = hunk.old_lines if before else hunk.new_lines
        end = start + max(count, 1) - 1
        if start <= symbol.end_line and end >= symbol.start_line:
            refs.append(f"{hunk.path}:{hunk.old_start}:{hunk.new_start}")
    return tuple(refs)


def build_changed_symbol_index(root: Path, context: ChangeContext) -> ChangedSymbolIndex:
    """Compare Python syntax for changed files; never infer semantic impact."""
    root = root.resolve()
    before: dict[tuple[str, str, SymbolKind], PythonSymbol] = {}
    after: dict[tuple[str, str, SymbolKind], PythonSymbol] = {}
    failures: list[str] = []
    unsupported: list[str] = []
    failed_paths: set[str] = set()
    for changed in context.changed_files:
        if not changed.path.endswith(".py"):
            continue
        if changed.status in {"renamed", "copied"}:
            unsupported.append(f"{changed.path}:{changed.status}")
            failed_paths.add(changed.path)
        base_code = _git_show(root, context.base_sha or "HEAD", changed.path)
        head_code = _head_code(root, context, changed.path)
        for side, code, destination in (("base", base_code, before), ("head", head_code, after)):
            if code is None:
                continue
            try:
                for symbol in _symbols(changed.path, code):
                    destination[(symbol.path, symbol.qualified_name, symbol.kind)] = symbol
            except (SyntaxError, ValueError):
                failures.append(f"{changed.path}:{side}")
                failed_paths.add(changed.path)
    changes: list[ChangedSymbol] = []
    for key in sorted(set(before) | set(after), key=lambda item: (item[0], item[1], item[2].value)):
        if key[0] in failed_paths:
            continue
        old, new = before.get(key), after.get(key)
        if old is None:
            change = SymbolChange.ADDED
        elif new is None:
            change = SymbolChange.DELETED
        elif old.signature_hash != new.signature_hash or old.body_hash != new.body_hash:
            change = SymbolChange.MODIFIED
        else:
            continue
        refs = tuple(sorted(set(
            (*_hunk_refs(old, context.changed_hunks, before=True),
             *_hunk_refs(new, context.changed_hunks, before=False))
        )))
        changes.append(ChangedSymbol(
            key[1], change, old, new,
            old is None or new is None or old.signature_hash != new.signature_hash,
            old is None or new is None or old.body_hash != new.body_hash,
            refs,
        ))
    return ChangedSymbolIndex(
        context.target,
        tuple(changes),
        bool(failures or unsupported),
        tuple(sorted(failures)),
        tuple(sorted(unsupported)),
    )
