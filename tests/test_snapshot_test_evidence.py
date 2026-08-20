from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from code_provenance.assessment import (
    EvidenceStatus,
    PatchHealthAssessor,
    ReviewAction,
    TrustDimension,
)
from code_provenance.evidence import AttestationLevel
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import run_pytest_evidence
from code_provenance.test_evidence import TestEvidence as PytestEvidence
from provenance_cli import DEFAULT_PYTEST_ARGUMENTS


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.test")
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    git(tmp_path, "add", "module.py", ".gitignore")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def test_clean_and_dirty_snapshots_have_distinct_deterministic_identity(tmp_path: Path):
    root = repository(tmp_path)
    clean = capture_code_snapshot(root)
    assert clean.dirty is False
    assert clean.diff_hash is None

    (root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty_a = capture_code_snapshot(root)
    dirty_b = capture_code_snapshot(root)

    assert dirty_a == dirty_b
    assert dirty_a.dirty is True
    assert dirty_a.diff_hash is not None
    assert dirty_a.snapshot_id != clean.snapshot_id


def test_untracked_content_changes_snapshot_identity(tmp_path: Path):
    root = repository(tmp_path)
    (root / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = capture_code_snapshot(root)
    (root / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_code_snapshot(root)

    assert first.diff_hash != second.diff_hash
    assert first.snapshot_id != second.snapshot_id


def test_snapshot_hashes_untracked_symlink_text_without_following_it(tmp_path: Path):
    root = repository(tmp_path)
    outside = tmp_path.parent / "outside-secret"
    outside.write_text("first", encoding="utf-8")
    (root / "link").symlink_to(outside)
    first = capture_code_snapshot(root)
    outside.write_text("changed externally", encoding="utf-8")
    second = capture_code_snapshot(root)

    assert first == second


def test_pytest_producer_records_observed_snapshot_bound_results(tmp_path: Path):
    root = repository(tmp_path)
    (root / "test_module.py").write_text(
        "from module import VALUE\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    snapshot = capture_code_snapshot(root)

    evidence = run_pytest_evidence(
        root,
        snapshot=snapshot,
        command=(sys.executable, "-m", "pytest", "-q"),
    )

    assert evidence.snapshot_id == snapshot.snapshot_id
    assert evidence.target_sha == snapshot.head_sha
    assert evidence.attestation is AttestationLevel.OBSERVED
    assert evidence.passed == 1
    assert evidence.failed == 0
    assert evidence.exit_code == 0
    assert len(evidence.report_hash) == 64
    assert evidence.repository_changed is False


def test_pytest_evidence_detects_repository_mutation_during_execution(tmp_path: Path):
    root = repository(tmp_path)
    (root / "test_mutation.py").write_text(
        "from pathlib import Path\n\ndef test_mutates_repo():\n"
        "    Path('module.py').write_text('VALUE = 9\\n')\n",
        encoding="utf-8",
    )
    snapshot = capture_code_snapshot(root)

    evidence = run_pytest_evidence(
        root,
        snapshot=snapshot,
        command=(sys.executable, "-m", "pytest", "-q"),
    )

    assert evidence.repository_changed is True
    assert evidence.complete is False


def test_manual_test_assertion_is_warning_not_verified_pass():
    root = Path(__file__).resolve().parents[1]
    result = PatchHealthAssessor().assess_repository(root, tests_passed=True)
    functional = result.dimension(TrustDimension.FUNCTIONAL_EVIDENCE)

    assert functional.status is EvidenceStatus.WARN
    assert functional.confidence <= 0.35
    assert "TEST_RESULT_UNVERIFIED" in functional.tags
    assert result.action is ReviewAction.REQUEST_TARGETED_EVIDENCE
    assert TrustDimension.FUNCTIONAL_EVIDENCE.value in result.missing_evidence


def test_cli_does_not_stack_quiet_flags_that_hide_test_counts():
    assert DEFAULT_PYTEST_ARGUMENTS == ("-m", "pytest")


def test_zero_collected_tests_cannot_pass_functional_gate():
    root = Path(__file__).resolve().parents[1]
    snapshot = capture_code_snapshot(root)
    evidence = PytestEvidence(
        snapshot_id=snapshot.snapshot_id, target_sha=snapshot.head_sha,
        command=("pytest",), framework="pytest", framework_version="fixture",
        discovered=0, selected=0, deselected=0, passed=0, failed=0, errors=0,
        skipped=0, xfailed=0, xpassed=0, collection_errors=0,
        interrupted=False, duration_seconds=0.1, exit_code=0,
        report_hash="d" * 64, complete=True, repository_changed=False,
        attestation=AttestationLevel.OBSERVED,
    )

    result = PatchHealthAssessor().assess_repository(root, test_evidence=evidence)

    assert result.dimension(TrustDimension.FUNCTIONAL_EVIDENCE).status is EvidenceStatus.UNKNOWN
    assert TrustDimension.FUNCTIONAL_EVIDENCE.value in result.missing_evidence
