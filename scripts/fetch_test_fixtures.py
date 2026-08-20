"""Fetch small categorized external fixtures; never downloads full corpora."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import requests

from code_provenance.testdata import (
    TestFixtureRecord,
    categorize_codeql_file,
    categorize_devgpt_row,
    categorize_swebench_row,
    validate_fixture_catalog,
    verify_fetched_fixtures,
)


SWE_URL = (
    "https://datasets-server.huggingface.co/first-rows?"
    "dataset=princeton-nlp/SWE-bench_Lite&config=default&split=test"
)
DEVGPT_URL = (
    "https://raw.githubusercontent.com/NAIST-SE/DevGPT/"
    "685efd2509dede9a6e996b839ae4e20d33430648/"
    "snapshot_20231012/ChatGPT_Link_Sharing.csv"
)
CODEQL_REVISION = "87c77cc26ccd1d2d9791b8563be6d425ccdf0874"
CODEQL_PATHS = (
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/app.py",
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/db_connection.py",
    "python/ql/test/query-tests/Security/CWE-089-SqlInjection/sql_injection.py",
)


def _get(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _write_fixture(root: Path, record: TestFixtureRecord, payload: object) -> None:
    destination = root / record.local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch(output: Path, *, limit: int) -> tuple[TestFixtureRecord, ...]:
    if not 1 <= limit <= 20:
        raise ValueError("fixture limit must be between 1 and 20")
    records: list[TestFixtureRecord] = []

    swe_payload = json.loads(_get(SWE_URL))
    for wrapped in swe_payload["rows"][:limit]:
        row = wrapped["row"]
        record = categorize_swebench_row(row)
        _write_fixture(output, record, row)
        records.append(record)

    devgpt_text = _get(DEVGPT_URL).decode("utf-8-sig")
    for row in list(csv.DictReader(io.StringIO(devgpt_text)))[:limit]:
        record = categorize_devgpt_row(row)
        _write_fixture(output, record, row)
        records.append(record)

    for path in CODEQL_PATHS[:limit]:
        url = f"https://raw.githubusercontent.com/github/codeql/{CODEQL_REVISION}/{path}"
        content = _get(url).decode("utf-8", errors="replace")
        record = categorize_codeql_file(path=path, content=content)
        _write_fixture(output, record, {"path": path, "content": content})
        records.append(record)

    validated = validate_fixture_catalog(tuple(records))
    (output / "catalog.json").write_text(
        json.dumps([item.to_dict() for item in validated], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_fetched_fixtures(output, validated)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/code_health/test_fixtures/raw"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    records = fetch(args.output, limit=args.limit)
    counts: dict[str, int] = {}
    for item in records:
        counts[item.category.value] = counts.get(item.category.value, 0) + 1
    print(json.dumps({"records": len(records), "categories": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
