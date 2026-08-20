from __future__ import annotations

from pathlib import Path

from code_provenance.architecture import ArchitectureStatus, analyze_python_architecture
from code_provenance.assessment import EvidenceStatus, PatchHealthAssessor, TrustDimension


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_dependency_cycle_is_reported_with_traceable_modules(tmp_path: Path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/a.py", "from pkg import b\n")
    write(tmp_path, "pkg/b.py", "from pkg import a\n")

    report = analyze_python_architecture(tmp_path)

    assert report.status is ArchitectureStatus.FAIL
    assert report.cycles == (("pkg.a", "pkg.b"),)
    assert any(signal.rule_id == "ARCH-CYCLE" for signal in report.signals)


def test_relative_import_cycle_is_not_hidden(tmp_path: Path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/a.py", "from . import b\n")
    write(tmp_path, "pkg/b.py", "from . import a\n")

    report = analyze_python_architecture(tmp_path)

    assert report.cycles == (("pkg.a", "pkg.b"),)


def test_acyclic_modules_pass_without_inventing_design_findings(tmp_path: Path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/domain.py", "VALUE = 1\n")
    write(tmp_path, "pkg/report.py", "from pkg.domain import VALUE\n")

    report = analyze_python_architecture(tmp_path)

    assert report.status is ArchitectureStatus.PASS
    assert report.cycles == ()
    assert report.modules_analyzed == 3


def test_parse_failure_abstains_instead_of_passing(tmp_path: Path):
    write(tmp_path, "broken.py", "def incomplete(\n")

    report = analyze_python_architecture(tmp_path)

    assert report.status is ArchitectureStatus.UNKNOWN
    assert report.confidence == 0.0
    assert report.signals[0].rule_id == "ARCH-PARSE"


def test_repository_assessment_contains_architecture_evidence():
    root = Path(__file__).resolve().parents[1]

    result = PatchHealthAssessor().assess_repository(root)
    architecture = result.dimension(TrustDimension.ARCHITECTURAL_COMPATIBILITY)

    assert architecture.status is not EvidenceStatus.UNKNOWN
    assert architecture.evidence_refs
    assert "architectural_compatibility" not in result.missing_evidence
