from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, ReviewAction, TrustDimension
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import run_pytest_evidence


def repo(tmp_path: Path, tests: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    (tmp_path / "test_cases.py").write_text(tests, encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "test_cases.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    return tmp_path


def run(root: Path, *extra: str, timeout: int = 30):
    snapshot = capture_code_snapshot(root)
    return run_pytest_evidence(
        root,
        snapshot=snapshot,
        command=(sys.executable, "-m", "pytest", "-q", *extra),
        timeout_seconds=timeout,
    )


def test_structured_report_preserves_selection_and_outcome_states(tmp_path: Path):
    root = repo(
        tmp_path,
        "import pytest\n"
        "def test_selected(): pass\n"
        "def test_other(): pass\n"
        "@pytest.mark.skip(reason='fixture')\ndef test_skip(): pass\n"
        "@pytest.mark.xfail(reason='fixture')\ndef test_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='fixture')\ndef test_xpass(): pass\n",
    )

    evidence = run(root, "-k", "selected or skip or xfail or xpass")

    assert evidence.discovered == 5
    assert evidence.selected == 4
    assert evidence.deselected == 1
    assert evidence.passed == 1
    assert evidence.skipped == 1
    assert evidence.xfailed == 1
    assert evidence.xpassed == 1
    assert evidence.complete is True
    assert {item.node_id: item.outcome for item in evidence.test_cases} == {
        "test_cases.py::test_selected": "passed",
        "test_cases.py::test_skip": "skipped",
        "test_cases.py::test_xfail": "xfailed",
        "test_cases.py::test_xpass": "xpassed",
    }


def test_failure_is_complete_but_not_passing(tmp_path: Path):
    root = repo(tmp_path, "def test_no(): assert False\n")
    evidence = run(root)
    assert evidence.failed == 1
    assert evidence.exit_code == 1
    assert evidence.complete is True
    assessment = PatchHealthAssessor().assess_repository(
        root, intent="Exercise failure routing", test_evidence=evidence
    )
    assert assessment.dimension(TrustDimension.FUNCTIONAL_EVIDENCE).status is EvidenceStatus.FAIL
    assert assessment.action is not ReviewAction.ALLOW_STANDARD_REVIEW


def test_collection_error_is_incomplete(tmp_path: Path):
    evidence = run(repo(tmp_path, "this is invalid python !!!\n"))
    assert evidence.collection_errors == 1
    assert evidence.complete is False


def test_timeout_is_interrupted_and_incomplete(tmp_path: Path):
    evidence = run(
        repo(tmp_path, "import time\ndef test_slow(): time.sleep(5)\n"),
        timeout=1,
    )
    assert evidence.exit_code == 124
    assert evidence.interrupted is True
    assert evidence.timed_out is True
    assert evidence.complete is False


def test_keyboard_interrupt_is_incomplete(tmp_path: Path):
    evidence = run(
        repo(
            tmp_path,
            "import os, signal\n"
            "def test_interrupt(): os.kill(os.getpid(), signal.SIGINT)\n",
        )
    )
    assert evidence.interrupted is True
    assert evidence.timed_out is False
    assert evidence.complete is False


def test_canonical_report_hash_is_stable_across_identical_runs(tmp_path: Path):
    root = repo(tmp_path, "def test_yes(): pass\n")
    first = run(root)
    second = run(root)
    assert first.report_hash == second.report_hash
    assert first.duration_seconds != second.duration_seconds
