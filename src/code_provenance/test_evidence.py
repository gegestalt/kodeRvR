"""Structured pytest evidence bound to an exact code snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from code_provenance.evidence import (
    AttestationLevel, EvidenceArtifact, EvidenceTarget, artifact_content_hash,
)
from code_provenance.snapshot import CodeSnapshot, capture_code_snapshot


@dataclass(frozen=True)
class ObservedTestCase:
    node_id: str
    outcome: str


@dataclass(frozen=True)
class TestEvidence:
    target: EvidenceTarget
    command: tuple[str, ...]
    framework: str
    framework_version: str
    discovered: int
    selected: int
    deselected: int
    passed: int
    failed: int
    runtime_errors: int
    skipped: int
    xfailed: int
    xpassed: int
    collection_errors: int
    interrupted: bool
    timed_out: bool
    duration_seconds: float
    exit_code: int
    output_hash: str
    report_hash: str
    complete: bool
    repository_changed: bool
    attestation: AttestationLevel
    test_cases: tuple[ObservedTestCase, ...] = ()

    @property
    def tests_collected(self) -> int:
        """Compatibility alias; selected tests are the checks that could run."""
        return self.selected

    @property
    def snapshot_id(self) -> str:
        return self.target.snapshot_id

    @property
    def target_sha(self) -> str:
        return self.target.head_sha

    @property
    def errors(self) -> int:
        """Compatibility alias for callers predating the structured contract."""
        return self.runtime_errors


def _canonical_report(values: dict[str, object]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def test_evidence_artifact(evidence: TestEvidence, *, repository_id: str) -> EvidenceArtifact:
    """Convert structured outcomes into an independently hashed ledger artifact."""
    report = {
        "schema_version": "1.0",
        "discovered": evidence.discovered,
        "selected": evidence.selected,
        "deselected": evidence.deselected,
        "passed": evidence.passed,
        "failed": evidence.failed,
        "runtime_errors": evidence.runtime_errors,
        "skipped": evidence.skipped,
        "xfailed": evidence.xfailed,
        "xpassed": evidence.xpassed,
        "collection_errors": evidence.collection_errors,
        "interrupted": evidence.interrupted,
        "timed_out": evidence.timed_out,
        "exit_code": evidence.exit_code,
        "complete": evidence.complete,
        "repository_changed": evidence.repository_changed,
        "test_cases": [item.__dict__ for item in evidence.test_cases],
    }
    payload = _canonical_report(report)
    if repository_id != evidence.target.repository_id:
        raise ValueError("repository_id does not match test evidence target")
    target = evidence.target
    return EvidenceArtifact(
        artifact_id=f"pytest:{evidence.report_hash[:16]}",
        kind="test_report",
        producer=evidence.framework,
        producer_version=evidence.framework_version,
        target=target,
        payload=payload,
        content_hash=artifact_content_hash(payload),
        attestation=evidence.attestation,
        execution_id=f"local:{evidence.report_hash}",
        complete=evidence.complete,
        created_at=datetime.now(UTC),
    )


def run_pytest_evidence(
    root: Path,
    *,
    snapshot: CodeSnapshot,
    command: tuple[str, ...],
    timeout_seconds: int = 300,
) -> TestEvidence:
    """Run pytest with a hook plugin; human-readable stdout is never parsed."""
    if not command:
        raise ValueError("test command is required")
    root = root.resolve()
    descriptor, report_name = tempfile.mkstemp(prefix="patch-health-pytest-", suffix=".json")
    os.close(descriptor)
    report_path = Path(report_name)
    report_path.unlink()
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH", ""))))
    env["CODE_PROVENANCE_PYTEST_REPORT"] = str(report_path)
    actual_command = (*command, "-p", "code_provenance.pytest_reporter")
    started = time.perf_counter()
    timed_out = False
    output = ""
    try:
        process = subprocess.run(
            list(actual_command), cwd=root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_seconds, env=env,
        )
        exit_code = process.returncode
        output = process.stdout + "\n" + process.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        output = stdout + "\n" + stderr
    duration = time.perf_counter() - started
    try:
        structured = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    finally:
        report_path.unlink(missing_ok=True)
    after = capture_code_snapshot(root)
    repository_changed = after.snapshot_id != snapshot.snapshot_id
    complete = bool(structured.get("complete", False)) and not timed_out and not repository_changed
    report = {
        "schema_version": "1.0",
        "discovered": int(structured.get("discovered", 0)),
        "selected": int(structured.get("selected", 0)),
        "deselected": int(structured.get("deselected", 0)),
        "passed": int(structured.get("passed", 0)),
        "failed": int(structured.get("failed", 0)),
        "runtime_errors": int(structured.get("runtime_errors", 0)),
        "skipped": int(structured.get("skipped", 0)),
        "xfailed": int(structured.get("xfailed", 0)),
        "xpassed": int(structured.get("xpassed", 0)),
        "collection_errors": int(structured.get("collection_errors", 0)),
        "interrupted": bool(structured.get("interrupted", False)) or timed_out,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "complete": complete,
        "repository_changed": repository_changed,
    }
    test_cases = tuple(
        ObservedTestCase(str(item["node_id"]), str(item["outcome"]))
        for item in structured.get("test_cases", [])
        if isinstance(item, dict) and "node_id" in item and "outcome" in item
    )
    report["test_cases"] = [item.__dict__ for item in test_cases]
    return TestEvidence(
        target=EvidenceTarget(snapshot.repository_id, snapshot.snapshot_id, snapshot.head_sha),
        command=actual_command,
        framework="pytest",
        framework_version=version("pytest"),
        duration_seconds=float(duration),
        output_hash=hashlib.sha256(output.encode()).hexdigest(),
        report_hash=hashlib.sha256(_canonical_report(report).encode()).hexdigest(),
        attestation=AttestationLevel.OBSERVED,
        test_cases=test_cases,
        **{key: value for key, value in report.items() if key not in {"schema_version", "test_cases"}},
    )
