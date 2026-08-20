"""Command line for AI code provenance and repository intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_provenance.efficiency import load_efficiency_evidence
from code_provenance.evidence import load_evidence_ledger
from code_provenance.evidence_quality import load_evidence_quality
from code_provenance.report import descriptive_repository_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="code-provenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="extract a read-only repository evidence report")
    scan.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    scan.add_argument("--intent", help="authoritative issue, specification, or PR intent")
    test_state = scan.add_mutually_exclusive_group()
    test_state.add_argument("--tests-passed", action="store_true", help="attach passing test evidence")
    test_state.add_argument("--tests-failed", action="store_true", help="attach failing test evidence")
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
            ),
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
