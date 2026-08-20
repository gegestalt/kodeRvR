"""Evaluate a verified local cache without network access or fixture execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from code_provenance.external_evaluation import evaluate_codeql_fixture, load_catalog, summarize_oracles
from code_provenance.testdata import FixtureCategory, verify_fetched_fixtures


ORACLES = {
    "python-ql-test-query-tests-Security-CWE-089-SqlInjection-app.py":
        ("CWE-089", True, ("SQL-FORMAT",)),
    "python-ql-test-query-tests-Security-CWE-089-SqlInjection-db_connection.py":
        ("CWE-089", False, ("SQL-FORMAT",)),
    "python-ql-test-query-tests-Security-CWE-089-SqlInjection-sql_injection.py":
        ("CWE-089", True, ("SQL-FORMAT",)),
}


def evaluate(cache: Path) -> dict[str, object]:
    records = load_catalog(cache / "catalog.json")
    verify_fetched_fixtures(cache, records)
    results = []
    counts = {"swebench": 0, "devgpt": 0, "codeql": 0}
    for record in records:
        if record.dataset_id == "swebench-lite":
            counts["swebench"] += 1
        elif record.dataset_id == "devgpt":
            counts["devgpt"] += 1
        elif record.category is FixtureCategory.SECURITY_ANALYZER_ORACLE:
            counts["codeql"] += 1
            payload = json.loads((cache / record.local_path).read_text(encoding="utf-8"))
            oracle, expected, supported_rules = ORACLES[record.record_id]
            results.append(evaluate_codeql_fixture(record.record_id, oracle, payload["content"],
                                                    expected_detection=expected,
                                                    supported_rule_ids=supported_rules))
    summary = summarize_oracles(results)
    return {
        "sources": counts,
        "codeql": asdict(summary),
        "oracle_results": [asdict(row) for row in results],
        "authorship_labels_inferred": sum(record.authorship_label is not None for record in records),
        "acquisition_failures": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/code_health"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cache_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
