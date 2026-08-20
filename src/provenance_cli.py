"""Command line for AI code provenance and repository intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from code_provenance.efficiency import load_efficiency_evidence
from code_provenance.evidence import load_evidence_ledger
from code_provenance.evidence_quality import load_evidence_quality
from code_provenance.report import descriptive_repository_report
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.test_evidence import run_pytest_evidence

DEFAULT_PYTEST_ARGUMENTS = ("-m", "pytest")


def main() -> None:
    parser = argparse.ArgumentParser(prog="code-provenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="extract a read-only repository evidence report")
    scan.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    scan.add_argument("--intent", help="authoritative issue, specification, or PR intent")
    test_state = scan.add_mutually_exclusive_group()
    test_state.add_argument("--run-tests", action="store_true", help="run pytest and bind observed results to the code snapshot")
    test_state.add_argument("--tests-passed", action="store_true", help="attach an unverified caller assertion")
    test_state.add_argument("--tests-failed", action="store_true", help="attach an unverified caller assertion")
    scan.add_argument(
        "--efficiency-evidence",
        type=Path,
        help="JSON file containing repeated baseline and candidate resource measurements",
    )
    scan.add_argument(
        "--evidence-quality",
        type=Path,
        help="JSON artifact from a named OOD detector with context and integrity evidence",
    )
    scan.add_argument(
        "--evidence-ledger",
        type=Path,
        help="versioned commit-bound artifact and claim ledger",
    )
    args = parser.parse_args()
    if args.command == "scan":
        tests_passed = True if args.tests_passed else False if args.tests_failed else None
        test_evidence = None
        if args.run_tests:
            snapshot = capture_code_snapshot(args.repository)
            test_evidence = run_pytest_evidence(
                args.repository,
                snapshot=snapshot,
                command=(sys.executable, *DEFAULT_PYTEST_ARGUMENTS),
            )
        efficiency = (
            load_efficiency_evidence(args.efficiency_evidence)
            if args.efficiency_evidence else (None, None)
        )
        evidence_quality = (
            load_evidence_quality(args.evidence_quality)
            if args.evidence_quality else None
        )
        evidence_ledger = (
            load_evidence_ledger(args.evidence_ledger)
            if args.evidence_ledger else None
        )
        print(json.dumps(
            descriptive_repository_report(
                args.repository,
                intent=args.intent,
                tests_passed=tests_passed,
                efficiency_baseline=efficiency[0],
                efficiency_candidate=efficiency[1],
                evidence_quality=evidence_quality,
                evidence_ledger=evidence_ledger,
                test_evidence=test_evidence,
            ),
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
