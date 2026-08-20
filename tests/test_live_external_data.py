"""Bounded live checks. Collection is offline; execution is explicitly opt-in."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import requests

from code_provenance.change_context import build_change_context
from code_provenance.dependency_context import build_dependency_context
from code_provenance.external_evaluation import evaluate_codeql_fixture, fetch_with_classified_failures, sha256_bytes
from code_provenance.snapshot import capture_code_snapshot
from code_provenance.symbol_index import build_changed_symbol_index
from code_provenance.testdata import categorize_codeql_file, categorize_devgpt_row, categorize_swebench_row


pytestmark = [pytest.mark.live_data, pytest.mark.external_fixture]

SWE_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
DEVGPT_REVISION = "685efd2509dede9a6e996b839ae4e20d33430648"
CODEQL_REVISION = "87c77cc26ccd1d2d9791b8563be6d425ccdf0874"


def test_live_swebench_fetch_uses_pinned_revision(request: pytest.FixtureRequest):
    url = ("https://datasets-server.huggingface.co/first-rows?dataset=princeton-nlp/"
           f"SWE-bench_Lite&config=default&split=test&revision={SWE_REVISION}")
    payload = json.loads(fetch_with_classified_failures(url, requests.get))
    row = payload["rows"][0]["row"]
    record = categorize_swebench_row(row, source_revision=SWE_REVISION)
    assert record.source_revision == SWE_REVISION
    assert record.payload_hash == categorize_swebench_row(row).payload_hash
    assert record.authorship_label is None


def test_live_devgpt_fetch_uses_pinned_revision():
    url = f"https://raw.githubusercontent.com/NAIST-SE/DevGPT/{DEVGPT_REVISION}/snapshot_20231012/ChatGPT_Link_Sharing.csv"
    payload = fetch_with_classified_failures(url, requests.get)
    assert sha256_bytes(payload)
    assert DEVGPT_REVISION in url


def test_live_codeql_fixture_uses_pinned_revision_and_runs_analyzer():
    path = "python/ql/test/query-tests/Security/CWE-089-SqlInjection/sql_injection.py"
    url = f"https://raw.githubusercontent.com/github/codeql/{CODEQL_REVISION}/{path}"
    content = fetch_with_classified_failures(url, requests.get).decode("utf-8")
    record = categorize_codeql_file(path=path, content=content, source_revision=CODEQL_REVISION)
    result = evaluate_codeql_fixture(record.record_id, "CWE-089", content, expected_detection=True,
                                     supported_rule_ids=("SQL-FORMAT",))
    assert record.source_revision == CODEQL_REVISION
    assert record.payload_hash == sha256_bytes(content.encode())
    assert result.expected_detection is True


@pytest.mark.slow
def test_real_repository_structural_pipeline_is_deterministic(tmp_path: Path):
    repo = tmp_path / "repo"
    url = "https://github.com/gegestalt/kodeRvR.git"
    base = "31ecb56589f3ecd8faec8e9008b028b0cfcfa5f3"
    head = "41916ef952a9003a6bce96c0c29356c69f5a40af"
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(repo)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", head], check=True, capture_output=True)
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip() == head
    first_snapshot = capture_code_snapshot(repo)
    change = build_change_context(repo, base_sha=base, head_sha=head)
    symbols = build_changed_symbol_index(repo, change)
    first_graph = build_dependency_context(repo, change, symbols)
    second_snapshot = capture_code_snapshot(repo)
    second_graph = build_dependency_context(repo, change, symbols)
    assert first_snapshot.snapshot_id == second_snapshot.snapshot_id
    assert first_graph.nodes == second_graph.nodes
    assert first_graph.edges == second_graph.edges
