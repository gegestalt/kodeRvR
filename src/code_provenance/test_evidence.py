"""Observed pytest execution evidence bound to an exact code snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
import re
import subprocess
import time

from code_provenance.evidence import AttestationLevel
from code_provenance.snapshot import CodeSnapshot


@dataclass(frozen=True)
class TestEvidence:
    snapshot_id: str
    target_sha: str
    command: tuple[str, ...]
    framework: str
    framework_version: str
    tests_collected: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    exit_code: int
    output_hash: str
    complete: bool
    attestation: AttestationLevel


def _count(output: str, name: str) -> int:
    matches = re.findall(rf"(?:^|\s)(\d+)\s+{name}\b", output)
    return int(matches[-1]) if matches else 0


def run_pytest_evidence(
    root: Path,
    *,
    snapshot: CodeSnapshot,
    command: tuple[str, ...],
    timeout_seconds: int = 300,
) -> TestEvidence:
    """Run pytest and record observed output; this is not CI-verified attestation."""
    if not command:
        raise ValueError("test command is required")
    started = time.perf_counter()
    try:
        process = subprocess.run(
            list(command),
            cwd=root.resolve(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        output = process.stdout + "\n" + process.stderr
        complete = True
        exit_code = process.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + "\n" + (error.stderr or "")
        complete = False
        exit_code = 124
    duration = time.perf_counter() - started
    passed = _count(output, "passed")
    failed = _count(output, "failed")
    skipped = _count(output, "skipped")
    errors = _count(output, "error") + _count(output, "errors")
    collected = passed + failed + skipped + errors
    return TestEvidence(
        snapshot_id=snapshot.snapshot_id,
        target_sha=snapshot.head_sha,
        command=command,
        framework="pytest",
        framework_version=version("pytest"),
        tests_collected=collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration_seconds=float(duration),
        exit_code=exit_code,
        output_hash=hashlib.sha256(output.encode()).hexdigest(),
        complete=complete,
        attestation=AttestationLevel.OBSERVED,
    )
