"""Fetch one curated random repository and emit its static evidence report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_provenance.repository_demo import (
    analyze_demo_repository,
    checkout_demo_repository,
    discover_random_repository,
    load_demo_repositories,
    select_demo_repository,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/code_health/demo_repositories.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/code_health/demo_repositories"))
    parser.add_argument("--repository-id", help="choose one curated fixture instead of random selection")
    parser.add_argument("--seed", type=int, help="make random selection reproducible")
    parser.add_argument("--discover-random", action="store_true",
                        help="discover one bounded public GitHub repository via the API")
    parser.add_argument("--github-language", help="GitHub language filter; omit for any language")
    parser.add_argument("--github-max-size-kb", type=int, default=500_000)
    parser.add_argument("--github-candidates", type=int, default=10)
    parser.add_argument("--max-file-vectors", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/random_repository_demo.json"))
    args = parser.parse_args()

    if args.discover_random:
        selected, discovery = discover_random_repository(
            seed=args.seed, language=args.github_language,
            max_size_kb=args.github_max_size_kb, candidates=args.github_candidates,
        )
        selection_scope = "github_api_discovery"
    else:
        fixtures = load_demo_repositories(args.manifest)
        selected = select_demo_repository(fixtures, seed=args.seed, fixture_id=args.repository_id)
        discovery = None
        selection_scope = "curated_manifest_only"
    checkout = checkout_demo_repository(selected, args.cache_dir)
    report = analyze_demo_repository(
        checkout, selected, max_file_vectors=args.max_file_vectors, selection_scope=selection_scope
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"selected": selected.fixture_id, "source_url": selected.source_url,
                      "base_revision": selected.base_revision,
                      "head_revision": selected.head_revision, "selection_scope": selection_scope,
                      "discovery": discovery, "checkout": str(checkout),
                      "report": str(args.output), "summary": report["summary"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
