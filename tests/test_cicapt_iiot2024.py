import pandas as pd
import pytest

from ips.adapters.cicapt import (
    CicaptPaths,
    build_campaign_timeline,
    build_multimodal_manifest,
    validate_provenance_graph,
    profile_network_csv,
    build_attack_preserving_sample,
)


def test_campaign_timeline_normalizes_attack_metadata_without_hiding_stage():
    attacks = pd.DataFrame({"Attack Time": ["2024-01-01 10:00:00"], "Attack PID": [123],
                            "Attack Category": ["Discovery"], "Technique ID": ["T1082"]})
    timeline = build_campaign_timeline(attacks)
    assert timeline.loc[0, "tactic"] == "Discovery"
    assert timeline.loc[0, "process_id"] == "123"
    assert timeline.loc[0, "technique"] == "T1082"
    assert timeline.loc[0, "policy_visible"] == False


def test_campaign_timeline_treats_numeric_attack_time_as_epoch_seconds():
    attacks = pd.DataFrame({"Time of Attack": [1701469507.0], "PID": [123],
                            "Tactic Name": ["collection"], "Technique Name": ["stage files"]})
    timeline = build_campaign_timeline(attacks)
    assert timeline.loc[0, "attack_time"].year == 2023


def test_provenance_validation_preserves_graph_semantics():
    graph = pd.DataFrame({"id": ["a", "b", "e"], "type": ["Process", "Artifact", "Used"],
                          "from": [None, None, "a"], "to": [None, None, "b"]})
    audit = validate_provenance_graph(graph)
    assert audit["process_nodes"] == 1
    assert audit["artifact_nodes"] == 1
    assert audit["edges"] == 1
    assert audit["dangling_edges"] == 0


def test_manifest_keeps_network_and_provenance_as_separate_modalities(tmp_path):
    paths = CicaptPaths(tmp_path / "network.csv", tmp_path / "provenance.csv", tmp_path / "Attack_info.csv")
    for path in (paths.network, paths.provenance, paths.attack_info):
        path.write_text("x\n1\n")
    manifest = build_multimodal_manifest(paths)
    assert manifest["join_contract"]["strategy"] == "late_fusion"
    assert manifest["modalities"]["network"]["sha256"]
    assert manifest["modalities"]["provenance"]["path"] != manifest["modalities"]["network"]["path"]


def test_manifest_fails_loudly_when_official_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_multimodal_manifest(CicaptPaths(tmp_path/"n", tmp_path/"p", tmp_path/"a"))


def test_network_profiler_streams_labels_and_time_range(tmp_path):
    path = tmp_path / "network.csv"
    path.write_text("ts,label,subLabel,subLabelCat,Source IP,Destination IP\n1,0,0,0,a,b\n2,1,x,attack,a,c\n")
    profile = profile_network_csv(path, chunksize=1)
    assert profile["rows"] == 2
    assert profile["timestamp_min"] == 1.0
    assert profile["label_counts"] == {"0": 1, "1": 1}


def test_attack_preserving_sample_keeps_every_attack_and_hides_labels_from_features(tmp_path):
    path = tmp_path / "network.csv"
    rows = ["ts,flow_duration,Rate,Source IP,label,subLabel,subLabelCat"]
    rows += [f"{i},{i/10},{i},a,0,0,0" for i in range(100)]
    rows += ["101,1,9,b,1,discovery,scan", "102,2,10,b,1,collection,stage"]
    path.write_text("\n".join(rows) + "\n")
    sample, manifest = build_attack_preserving_sample(path, benign_fraction=.1, seed=42, chunksize=30)
    assert sample.attack_present.sum() == 2
    assert manifest["source_attack_rows"] == 2
    assert "Source IP" not in sample
    assert {"attack_present", "attack_tactic", "attack_technique"} <= set(sample)
