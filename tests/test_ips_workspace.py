from pathlib import Path

from ips.workspace import ProjectPaths


def test_cicapt_discovery_accepts_browser_download_or_canonical_raw(tmp_path):
    browser = tmp_path / "CICAPT-IIoT2024"
    browser.mkdir()
    (browser / "phase1_NetworkData.csv").write_text("x\n")
    paths = ProjectPaths(tmp_path)
    assert paths.cicapt_source() == browser


def test_cicapt_inventory_classifies_modalities_without_reading_files(tmp_path):
    root = tmp_path / "data" / "cicapt_iiot2024" / "raw"
    root.mkdir(parents=True)
    for name in ("phase2_NetworkData.csv", "Attack_info.csv", "phase2_provenance.csv", "merged.pcap"):
        (root / name).write_text("x")
    inventory = ProjectPaths(tmp_path).cicapt_inventory()
    assert inventory["network_csv"][0].name == "phase2_NetworkData.csv"
    assert inventory["attack_info"][0].name == "Attack_info.csv"
    assert inventory["provenance"][0].name == "phase2_provenance.csv"


def test_cicapt_readiness_waits_for_all_required_modalities(tmp_path):
    root = tmp_path / "CICAPT-IIoT2024"; root.mkdir()
    (root / "phase1_NetworkData.csv").write_text("x")
    status = ProjectPaths(tmp_path).cicapt_status()
    assert status["download_detected"] is True
    assert status["ready_for_audit"] is False
    assert "provenance" in status["missing_modalities"]


def test_primary_artifacts_resolve_canonical_modality_folders(tmp_path):
    raw = tmp_path / "data" / "cicapt_iiot2024" / "raw"
    paths = {"phase1_network": raw/"network"/"phase1_NetworkData.csv",
             "phase2_network": raw/"network"/"phase2_NetworkData.csv",
             "phase1_provenance": raw/"provenance"/"Phase1_Provenance.csv",
             "phase2_provenance": raw/"provenance"/"Phase2_Provenance.csv",
             "attack_info": raw/"ground_truth"/"attack_info.csv"}
    for path in paths.values(): path.parent.mkdir(parents=True,exist_ok=True); path.write_text("x")
    assert ProjectPaths(tmp_path).cicapt_primary_artifacts() == paths
