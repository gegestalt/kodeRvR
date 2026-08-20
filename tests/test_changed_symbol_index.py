from __future__ import annotations

from pathlib import Path
import subprocess

from code_provenance.change_context import build_change_context
from code_provenance.symbol_index import SymbolChange, build_changed_symbol_index


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def repo(tmp_path: Path, code: str, path: str = "module.py") -> tuple[Path, str]:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    git(tmp_path, "add", path)
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


def index(root: Path):
    return build_changed_symbol_index(root, build_change_context(root))


def test_added_modified_and_deleted_functions_are_classified(tmp_path: Path):
    root, _ = repo(tmp_path, "def keep(x):\n    return x\n\ndef remove():\n    return 1\n")
    (root / "module.py").write_text(
        "def keep(x):\n    return x + 1\n\ndef added():\n    return 2\n", encoding="utf-8"
    )
    changes = {item.qualified_name: item.change for item in index(root).changes}
    assert changes == {"added": SymbolChange.ADDED, "keep": SymbolChange.MODIFIED, "remove": SymbolChange.DELETED}


def test_signature_and_body_changes_are_distinguished(tmp_path: Path):
    root, _ = repo(tmp_path, "def api(x: int = 1) -> int:\n    return x\n")
    (root / "module.py").write_text("def api(x: str = 'a') -> str:\n    return x + '!'\n", encoding="utf-8")
    changed = index(root).changes[0]
    assert changed.signature_changed is True
    assert changed.body_changed is True

    (root / "module.py").write_text("def api(x: int = 1) -> int:\n    return x + 1\n", encoding="utf-8")
    body_only = index(root).changes[0]
    assert body_only.signature_changed is False
    assert body_only.body_changed is True


def test_public_private_async_decorated_and_nested_symbols(tmp_path: Path):
    root, _ = repo(tmp_path, "pass\n")
    (root / "module.py").write_text(
        "class Public:\n"
        "    @staticmethod\n    async def method(x: int):\n"
        "        def nested(): return x\n        return nested()\n"
        "def _private(): return 1\n",
        encoding="utf-8",
    )
    symbols = {item.qualified_name: item.after for item in index(root).changes}
    assert symbols["Public"].potentially_public is True
    assert symbols["Public.method"].potentially_public is True
    assert symbols["Public.method.<locals>.nested"].potentially_public is False
    assert symbols["_private"].potentially_public is False
    assert symbols["Public.method"].decorators == ("staticmethod",)


def test_identity_ignores_line_shift_but_includes_file_and_class(tmp_path: Path):
    root, _ = repo(tmp_path, "def same(): return 1\n")
    first = index(root)
    (root / "module.py").write_text("\n\n\ndef same(): return 2\n", encoding="utf-8")
    shifted = index(root).changes[0]
    base_symbol = next(item.before for item in shifted_index(first, root) if item.qualified_name == "same")
    assert shifted.before.symbol_id == shifted.after.symbol_id == base_symbol.symbol_id

    (root / "other.py").write_text("def same(): return 3\n", encoding="utf-8")
    others = [item.after for item in index(root).changes if item.after and item.qualified_name == "same"]
    assert len({item.symbol_id for item in others}) == len(others)


def shifted_index(unused, root: Path):
    del unused
    return index(root).changes


def test_same_method_name_in_different_classes_has_distinct_identity(tmp_path: Path):
    root, _ = repo(tmp_path, "pass\n")
    (root / "module.py").write_text(
        "class A:\n    def run(self): return 1\nclass B:\n    def run(self): return 2\n",
        encoding="utf-8",
    )
    methods = [item.after for item in index(root).changes if item.qualified_name.endswith(".run")]
    assert len({item.symbol_id for item in methods}) == 2


def test_parse_failure_is_partial_not_false_symbol_absence(tmp_path: Path):
    root, _ = repo(tmp_path, "def valid(): return 1\n")
    (root / "module.py").write_text("def broken(:\n", encoding="utf-8")
    result = index(root)
    assert result.partial is True
    assert result.parse_failures == ("module.py:head",)
    assert result.changes == ()


def test_hunks_map_to_containing_symbol_and_target_matches_context(tmp_path: Path):
    root, _ = repo(tmp_path, "def first():\n    return 1\n\ndef second():\n    return 2\n")
    (root / "module.py").write_text("def first():\n    return 1\n\ndef second():\n    return 3\n", encoding="utf-8")
    context = build_change_context(root)
    result = build_changed_symbol_index(root, context)
    changed = result.changes[0]
    assert changed.qualified_name == "second"
    assert changed.affected_hunks
    assert result.target == context.target


def test_empty_and_init_modules_are_supported(tmp_path: Path):
    root, _ = repo(tmp_path, "", "pkg/__init__.py")
    (root / "pkg/__init__.py").write_text("class API: pass\n", encoding="utf-8")
    result = index(root)
    assert result.changes[0].qualified_name == "API"
    assert result.partial is False


def test_multiline_unicode_property_and_classmethod_signatures(tmp_path: Path):
    root, _ = repo(tmp_path, "pass\n")
    (root / "module.py").write_text(
        "class Café:\n"
        "    @property\n    def résumé(self) -> str:\n        return 'ok'\n"
        "    @classmethod\n    def build(\n        cls,\n        value: int = 1,\n    ) -> 'Café':\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    symbols = {item.qualified_name: item.after for item in index(root).changes}
    assert symbols["Café.résumé"].decorators == ("property",)
    assert symbols["Café.build"].decorators == ("classmethod",)
    assert len(symbols["Café.build"].signature_hash) == 64


def test_file_rename_is_partial_instead_of_false_symbol_addition(tmp_path: Path):
    root, _ = repo(tmp_path, "def api(): return 1\n")
    git(root, "mv", "module.py", "renamed.py")
    result = index(root)
    assert result.partial is True
    assert result.unsupported_cases
    assert result.changes == ()
