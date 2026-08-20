"""Command line for AI code provenance and repository intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()
    if args.command == "scan":
        tests_passed = True if args.tests_passed else False if args.tests_failed else None
        print(json.dumps(
            descriptive_repository_report(
                args.repository,
                intent=args.intent,
                tests_passed=tests_passed,
            ),
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
