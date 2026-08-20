from __future__ import annotations

from pathlib import Path
import subprocess

from code_provenance.change_context import ChangeIntent, build_change_context
from code_provenance.report import descriptive_repository_report


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, str]:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(tmp_path, "add", "app.py")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


def test_working_tree_context_binds_target_files_hunks_and_intent(tmp_path: Path):
    root, head = repository(tmp_path)
    (root / "app.py").write_text("VALUE = 2\nEXTRA = True\n", encoding="utf-8")

    context = build_change_context(
        root,
        intent=ChangeIntent("Fix value handling", "user"),
    )

    assert context.base_sha is None
    assert context.head_sha == head
    assert context.target.head_sha == head
    assert context.target.snapshot_id.startswith("snap_")
    assert [(item.path, item.status) for item in context.changed_files] == [("app.py", "modified")]
    assert context.changed_files[0].additions == 2
    assert context.changed_files[0].deletions == 1
    assert context.changed_hunks[0].path == "app.py"
    assert context.intent.text == "Fix value handling"
    assert context.missing_context == frozenset({"base_sha"})


def test_commit_range_context_is_deterministic_and_base_bound(tmp_path: Path):
    root, base = repository(tmp_path)
    (root / "name with spaces.py").write_text("ENABLED = True\n", encoding="utf-8")
    git(root, "add", "name with spaces.py")
    git(root, "commit", "-qm", "add spaced file")
    head = git(root, "rev-parse", "HEAD")

    first = build_change_context(root, base_sha=base, head_sha=head)
    second = build_change_context(root, base_sha=base, head_sha=head)

    assert first == second
    assert first.base_sha == base
    assert first.head_sha == head
    assert first.changed_files[0].path == "name with spaces.py"
    assert first.changed_files[0].status == "added"
    assert first.missing_context == frozenset({"intent"})


def test_context_rejects_invalid_revision(tmp_path: Path):
    root, _ = repository(tmp_path)

    try:
        build_change_context(root, base_sha="not-a-revision")
    except ValueError as error:
        assert "revision" in str(error)
    else:
        raise AssertionError("invalid revision was accepted")


def test_clean_working_tree_reports_no_changes_without_fabricating_context(tmp_path: Path):
    root, _ = repository(tmp_path)
    context = build_change_context(root)

    assert context.changed_files == ()
    assert context.changed_hunks == ()
    assert context.context_completeness < 1.0
    assert context.missing_context == frozenset({"base_sha", "intent"})


def test_repository_report_exposes_typed_change_context(tmp_path: Path):
    root, head = repository(tmp_path)
    (root / "app.py").write_text("VALUE = 3\n", encoding="utf-8")

    report = descriptive_repository_report(root, intent="Change VALUE")

    context = report["change_context"]
    assert context["head_sha"] == head
    assert context["intent"] == {"text": "Change VALUE", "source": "cli"}
    assert context["changed_files"][0]["path"] == "app.py"
    assert context["missing_context"] == ["base_sha"]
    assert report["feature_space"]["model_feature_count"] >= 50
    assert report["feature_space"]["change"]["change_files"] == 1
    assert report["feature_space"]["repository"]["repository_files"] == 1
    symbols = report["symbol_context"]
    assert symbols["target"] == context["target"]
    assert symbols["changes"] == ()
    dependencies = report["dependency_context"]
    assert dependencies["target"] == context["target"]
    assert dependencies["max_depth"] == 4
    relevance = report["test_relevance"]
    assert relevance["target"] == context["target"]
    assert relevance["partial"] is True
