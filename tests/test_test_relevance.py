from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from code_provenance.change_context import build_change_context
from code_provenance.dependency_context import build_dependency_context
from code_provenance.evidence import EvidenceTarget
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.symbol_index import build_changed_symbol_index
from code_provenance.test_evidence import run_pytest_evidence
from code_provenance.test_relevance import RelevanceRelation, build_test_relevance_context


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    (tmp_path / "service.py").write_text("def validate(token):\n    return bool(token)\n", encoding="utf-8")
    (tmp_path / "controller.py").write_text("from service import validate\ndef login(token): return validate(token)\n", encoding="utf-8")
    (tmp_path / "test_service.py").write_text(
        "import pytest\nfrom service import validate\nfrom controller import login\n"
        "def test_validate_token(): assert validate('x')\n"
        "def test_login_flow(): assert login('x')\n"
        "@pytest.mark.skip(reason='fixture')\ndef test_validate_skip(): assert validate('x')\n"
        "def test_unrelated(): assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    git(tmp_path, "add", ".gitignore", "service.py", "controller.py", "test_service.py")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "service.py").write_text("def validate(token):\n    return token is not None\n", encoding="utf-8")
    return tmp_path


def contexts(root: Path, evidence=True):
    change = build_change_context(root)
    symbols = build_changed_symbol_index(root, change)
    dependencies = build_dependency_context(root, change, symbols)
    tests = None
    if evidence:
        tests = run_pytest_evidence(
            root, snapshot=capture_code_snapshot(root),
            command=(sys.executable, "-m", "pytest", "-q"),
        )
    return change, symbols, dependencies, tests


def test_inventory_and_direct_indirect_relevance_are_deterministic(tmp_path: Path):
    values = contexts(repository(tmp_path), evidence=False)
    first = build_test_relevance_context(*values)
    second = build_test_relevance_context(*values)
    assert first == second
    assert len(first.inventory) == 4
    relations = {item.test.qualified_name: item.relation for item in first.relevant_tests}
    assert relations["test_validate_token"] is RelevanceRelation.DIRECT
    assert relations["test_login_flow"] is RelevanceRelation.INDIRECT
    assert first.relevant_tests[1].dependency_distance is not None


def test_observed_passed_skipped_and_not_observed_states_are_explicit(tmp_path: Path):
    root = repository(tmp_path)
    change, symbols, dependencies, evidence = contexts(root)
    result = build_test_relevance_context(change, symbols, dependencies, evidence)
    outcomes = {item.test.qualified_name: item.observed_outcome for item in result.relevant_tests}
    assert outcomes["test_validate_token"] == "passed"
    assert outcomes["test_validate_skip"] == "skipped"
    assert result.relevant_tests_observed_fraction == 1.0

    absent = build_test_relevance_context(change, symbols, dependencies, None)
    assert all(item.observed_outcome == "not_observed" for item in absent.relevant_tests)
    assert absent.partial is True


def test_target_mismatch_is_rejected(tmp_path: Path):
    root = repository(tmp_path)
    change, symbols, dependencies, evidence = contexts(root)
    wrong = replace(evidence, target=EvidenceTarget("wrong", "snap_wrong", "0" * 40))
    with pytest.raises(ValueError, match="target"):
        build_test_relevance_context(change, symbols, dependencies, wrong)


def test_bounded_distance_and_unrelated_test_exclusion(tmp_path: Path):
    values = contexts(repository(tmp_path), evidence=False)
    result = build_test_relevance_context(*values, max_distance=1)
    names = {item.test.qualified_name for item in result.relevant_tests}
    assert "test_validate_token" in names
    assert "test_login_flow" not in names
    assert "test_unrelated" not in names
