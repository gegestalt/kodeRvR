"""Single, discoverable command line for adaptive IPS experiments."""

from __future__ import annotations

import argparse
import json

from ips.workspace import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(prog="ips-cli", description="Adaptive IPS project commands")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show dataset and experiment readiness without loading bulk data")
    sub.add_parser("cicapt-audit", help="audit completed CICAPT download (auto-discovered)")
    sub.add_parser("cicapt-profile", help="stream both CICAPT network phases and profile labels/timestamps")
    sub.add_parser("cicapt-build", help="build attack-preserving chronological CICAPT events")
    sub.add_parser("cicapt-benchmark", help="run chronological CICAPT detector benchmark")
    sub.add_parser("cicapt-source-audit", help="statically audit downloaded reference Python files")
    sub.add_parser("cicapt-fusion", help="compare network/provenance/fusion and build tactic beliefs")
    sub.add_parser("cicapt-data-intelligence", help="build CICAPT health, drift, alignment, OOD and graph diagnostics")
    sub.add_parser("cse-build", help="build chronological CSE detector events")
    sub.add_parser("cse-benchmark", help="run the five-seed CSE policy benchmark")
    sub.add_parser("claim-control", help="build scientific claim-control artifacts")
    sub.add_parser("next-phase", help="run detector/POMDP scientific-control report")
    args, remaining = parser.parse_known_args()
    project = ProjectPaths.discover()
    if args.command == "status":
        print(json.dumps({"project": str(project.root), "cicapt": project.cicapt_status()}, indent=2))
    elif args.command == "cicapt-audit":
        import sys
        from experiments.ips.cicapt.audit import main as audit
        sys.argv = ["cicapt_iiot2024_audit.py", *remaining]
        audit()
    elif args.command == "cse-build":
        from experiments.ips.cse.build_events import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-profile":
        from experiments.ips.cicapt.profile import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-build":
        from experiments.ips.cicapt.build_events import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-benchmark":
        from experiments.ips.cicapt.benchmark import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-source-audit":
        from experiments.ips.cicapt.source_audit import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-fusion":
        from experiments.ips.cicapt.fusion import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cicapt-data-intelligence":
        from experiments.ips.cicapt.data_intelligence import run
        print(json.dumps(run(), indent=2))
    elif args.command == "cse-benchmark":
        from experiments.ips.cse.benchmark import run
        print(run().to_string(index=False))
    elif args.command == "claim-control":
        from experiments.ips.reports.claim_control import run
        print(json.dumps(run(), indent=2))
    else:
        from experiments.ips.reports.next_phase import run
        print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
