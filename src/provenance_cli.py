"""Command line for AI code provenance and repository intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_provenance.efficiency import load_efficiency_evidence
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
    args = parser.parse_args()
    if args.command == "scan":
        tests_passed = True if args.tests_passed else False if args.tests_failed else None
        efficiency = (
            load_efficiency_evidence(args.efficiency_evidence)
            if args.efficiency_evidence else (None, None)
        )
        print(json.dumps(
            descriptive_repository_report(
                args.repository,
                intent=args.intent,
                tests_passed=tests_passed,
                efficiency_baseline=efficiency[0],
                efficiency_candidate=efficiency[1],
            ),
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
