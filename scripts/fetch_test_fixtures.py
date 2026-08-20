"""Fetch bounded, pinned external fixtures into a verified untrusted-data cache."""
from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import requests

from code_provenance.external_evaluation import (
    AcquisitionFailureKind,
    ExternalDataError,
    fetch_with_classified_failures,
    load_catalog,
)
from code_provenance.testdata import (
    TestFixtureRecord,
    categorize_codeql_file,
    categorize_devgpt_row,
    categorize_swebench_row,
    validate_fixture_catalog,
    verify_fetched_fixtures,
)

SWE_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
SWE_URL = (
    "https://datasets-server.huggingface.co/first-rows?dataset=princeton-nlp/"
    f"SWE-bench_Lite&config=default&split=test&revision={SWE_REVISION}"
)
DEVGPT_REVISION = "685efd2509dede9a6e996b839ae4e20d33430648"
DEVGPT_URL = (
    f"https://raw.githubusercontent.com/NAIST-SE/DevGPT/{DEVGPT_REVISION}/"
    "snapshot_20231012/ChatGPT_Link_Sharing.csv"
)
CODEQL_REVISION = "87c77cc26ccd1d2d9791b8563be6d425ccdf0874"
CODEQL_PATHS = (
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/app.py",
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/db_connection.py",
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/sql_injection.py",
)
DATASETS = ("swebench", "devgpt", "codeql")


def _get(url: str) -> bytes:
    try:
        return fetch_with_classified_failures(url, requests.get)
    except requests.Timeout as error:
        raise ExternalDataError(AcquisitionFailureKind.TIMEOUT, str(error)) from error
    except requests.RequestException as error:
        raise ExternalDataError(AcquisitionFailureKind.UPSTREAM_UNAVAILABLE, str(error)) from error


def _write_fixture(root: Path, record: TestFixtureRecord, payload: object, *, refresh: bool) -> None:
    destination = root / record.local_path
    if destination.exists() and not refresh:
        verify_fetched_fixtures(root, (record,))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timestamped(record: TestFixtureRecord) -> TestFixtureRecord:
    return replace(record, fetched_at=datetime.now(timezone.utc).isoformat())


def fetch(
    output: Path,
    *,
    limit: int,
    datasets: tuple[str, ...] = DATASETS,
    refresh: bool = False,
) -> tuple[TestFixtureRecord, ...]:
    if not 1 <= limit <= 20:
        raise ValueError("fixture limit must be between 1 and 20")
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    records: list[TestFixtureRecord] = []
    if "swebench" in datasets:
        try:
            wrapped_rows = json.loads(_get(SWE_URL))["rows"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ExternalDataError(AcquisitionFailureKind.SCHEMA_CHANGE, f"SWE-bench: {error}") from error
        for wrapped in wrapped_rows[:limit]:
            row = wrapped["row"]
            record = _timestamped(categorize_swebench_row(row, source_revision=SWE_REVISION))
            _write_fixture(output, record, row, refresh=refresh)
            records.append(record)
    if "devgpt" in datasets:
        rows = list(csv.DictReader(io.StringIO(_get(DEVGPT_URL).decode("utf-8-sig"))))
        for row in rows[:limit]:
            record = _timestamped(categorize_devgpt_row(row, source_revision=DEVGPT_REVISION))
            _write_fixture(output, record, row, refresh=refresh)
            records.append(record)
    if "codeql" in datasets:
        for path in CODEQL_PATHS[:limit]:
            url = f"https://raw.githubusercontent.com/github/codeql/{CODEQL_REVISION}/{path}"
            content = _get(url).decode("utf-8", errors="replace")
            record = _timestamped(categorize_codeql_file(
                path=path, content=content, source_revision=CODEQL_REVISION
            ))
            _write_fixture(output, record, {"path": path, "content": content}, refresh=refresh)
            records.append(record)
    validated = validate_fixture_catalog(tuple(records))
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps([item.to_dict() for item in validated], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_fetched_fixtures(output, validated)
    return validated


def verify_only(output: Path) -> tuple[TestFixtureRecord, ...]:
    records = load_catalog(output / "catalog.json")
    return verify_fetched_fixtures(output, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", "--output", dest="output", type=Path,
        default=Path(".cache/code_health"),
    )
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    records = verify_only(args.output) if args.verify_only else fetch(
        args.output,
        limit=args.limit,
        datasets=tuple(args.dataset or DATASETS),
        refresh=args.refresh,
    )
    counts: dict[str, int] = {}
    for item in records:
        counts[item.category.value] = counts.get(item.category.value, 0) + 1
    print(json.dumps(
        {"records": len(records), "categories": counts, "cache": str(args.output)},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
