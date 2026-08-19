from pathlib import Path

from ips.analysis.source_audit import audit_python_sources


def test_source_audit_detects_shell_delete_wildcard_and_broad_except(tmp_path):
    source = tmp_path / "upstream.py"
    source.write_text("from x import *\nimport os\ntry:\n os.system('x')\nexcept:\n pass\nos.remove('x')\n")
    report = audit_python_sources([source])
    risks = set(report.iloc[0].risks)
    assert {"wildcard_import", "shell_execution", "file_deletion", "broad_except"} <= risks
    assert not bool(report.iloc[0].safe_to_execute)
