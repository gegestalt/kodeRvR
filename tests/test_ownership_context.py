from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from code_provenance.change_context import build_change_context
from code_provenance.dependency_context import build_dependency_context
from code_provenance.evidence import EvidenceTarget
from code_provenance.ownership_context import OwnerKind, OwnershipIssueKind, build_ownership_context
from code_provenance.symbol_index import build_changed_symbol_index
from code_provenance.test_relevance import build_test_relevance_context


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def repository(tmp_path: Path, codeowners: dict[str, str] | None = None) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / "src/auth").mkdir(parents=True)
    (tmp_path / "src/auth/service.py").write_text("def validate(): return 1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir(exist_ok=True)
    for path, content in (codeowners or {}).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(tmp_path, "add", "src", *(codeowners or {}).keys())
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "src/auth/service.py").write_text("def validate(): return 2\n", encoding="utf-8")
    return tmp_path


def contexts(root: Path):
    change = build_change_context(root)
    symbols = build_changed_symbol_index(root, change)
    dependencies = build_dependency_context(root, change, symbols)
    relevance = build_test_relevance_context(change, symbols, dependencies)
    return change, symbols, dependencies, relevance


def ownership(root: Path):
    return build_ownership_context(root, *contexts(root))


@pytest.mark.parametrize("location", ["CODEOWNERS", "docs/CODEOWNERS"])
def test_root_and_docs_codeowners_are_discovered(tmp_path: Path, location: str):
    result = ownership(repository(tmp_path, {location: "*.py @owner\n"}))
    assert result.source_path == location


def test_github_codeowners_takes_precedence(tmp_path: Path):
    result = ownership(repository(tmp_path, {
        ".github/CODEOWNERS": "*.py @github\n",
        "CODEOWNERS": "*.py @root\n",
        "docs/CODEOWNERS": "*.py @docs\n",
    }))
    assert result.source_path == ".github/CODEOWNERS"
    assert result.declared_owners[0].identifier == "@github"


def test_comments_whitespace_and_owner_kinds_are_parsed(tmp_path: Path):
    result = ownership(repository(tmp_path, {"CODEOWNERS": (
        "# comment\n\n*.py @user @org/team dev@example.com @user # trailing\n"
    )}))
    owners = result.rules[0].owners
    assert [(item.identifier, item.kind) for item in owners] == [
        ("@user", OwnerKind.USER), ("@org/team", OwnerKind.TEAM),
        ("dev@example.com", OwnerKind.EMAIL),
    ]


@pytest.mark.parametrize("pattern", [
    "src/auth/service.py", "*.py", "/src/auth/", "src/**/*.py", "src/auth/*.py",
])
def test_exact_extension_directory_recursive_and_rooted_patterns_match(tmp_path: Path, pattern: str):
    result = ownership(repository(tmp_path, {"CODEOWNERS": f"{pattern} @owner\n"}))
    assert result.paths[0].owners[0].identifier == "@owner"


def test_later_matching_rule_overrides_earlier_rule(tmp_path: Path):
    result = ownership(repository(tmp_path, {"CODEOWNERS": "*.py @general\nsrc/auth/*.py @auth\n"}))
    assert [item.identifier for item in result.paths[0].owners] == ["@auth"]
    assert result.paths[0].matched_rule.line == 2


def test_unmatched_path_is_unowned_and_not_failure(tmp_path: Path):
    result = ownership(repository(tmp_path, {"CODEOWNERS": "docs/ @docs\n"}))
    assert result.unowned_paths == ("src/auth/service.py",)
    assert result.complete is True


def test_changed_symbol_inherits_only_its_path_ownership(tmp_path: Path):
    result = ownership(repository(tmp_path, {"CODEOWNERS": "src/auth/ @auth\n"}))
    symbol = result.symbols[0]
    assert symbol.owners == result.paths[0].owners
    assert symbol.evidence_refs == ("CODEOWNERS:1",)


def test_missing_codeowners_is_complete_declared_absence(tmp_path: Path):
    result = ownership(repository(tmp_path))
    assert result.source_path is None
    assert result.complete is True
    assert result.rules == ()
    assert result.unowned_paths == ("src/auth/service.py",)


def test_malformed_and_unsupported_rules_remain_visible(tmp_path: Path):
    result = ownership(repository(tmp_path, {"CODEOWNERS": "*.py\n!secret.py @owner\n"}))
    assert {item.kind for item in result.issues} == {
        OwnershipIssueKind.MALFORMED_RULE, OwnershipIssueKind.UNSUPPORTED_PATTERN,
    }
    assert result.complete is False


def test_escaped_space_and_non_ascii_paths_match_deterministically(tmp_path: Path):
    root = repository(tmp_path, {"CODEOWNERS": "docs/my\\ file.md @docs\ndocs/équipe.md @intl\n"})
    (root / "docs/my file.md").write_text("x", encoding="utf-8")
    (root / "docs/équipe.md").write_text("x", encoding="utf-8")
    first = ownership(root)
    second = ownership(root)
    assert first == second
    values = {item.path: item.owners[0].identifier for item in first.paths if item.owners}
    assert values["docs/my file.md"] == "@docs"
    assert values["docs/équipe.md"] == "@intl"


def test_target_mismatch_is_rejected(tmp_path: Path):
    root = repository(tmp_path, {"CODEOWNERS": "*.py @owner\n"})
    change, symbols, dependencies, relevance = contexts(root)
    wrong = replace(relevance, target=EvidenceTarget("wrong", "snap_wrong", "0" * 40))
    with pytest.raises(ValueError, match="target"):
        build_ownership_context(root, change, symbols, dependencies, wrong)
