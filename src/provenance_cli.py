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
    args = parser.parse_args()
    if args.command == "scan":
        print(json.dumps(descriptive_repository_report(args.repository), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
