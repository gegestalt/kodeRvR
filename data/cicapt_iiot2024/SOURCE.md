# CICAPT-IIoT2024 acquisition and provenance

Official description: <https://www.unb.ca/cic/datasets/iiot-dataset-2024.html>

Official download form: <https://cicresearch.ca/IOTDataset/CICAPT-IIoT-Dataset/>

The download requires the researcher to submit UNB's form, so this repository
does not automate acceptance or fabricate a mirror. The completed download is
organized under `data/cicapt_iiot2024/raw/`:

- `network/`: authoritative phase-1 and phase-2 network CSVs;
- `provenance/`: authoritative phase-1 and phase-2 graph CSVs;
- `ground_truth/attack_info.csv`: hidden Caldera campaign metadata;
- `reference_code/`: dataset-supplied extractor modules, statically audited;
- `archive/`: preserved duplicate PCAP/download-tree/reference material that is
  never selected as an experiment input.

Run `python src/ips_cli.py cicapt-audit` after the download finishes. It
hashes each artifact, audits the graph, normalizes the attack timeline, and
writes only manifests/audits. Network, provenance, and attack ground truth are
never pooled into one table. Attack time, PID, tactic/category, technique and
campaign step are hidden evaluation state and forbidden policy inputs.

Scientific role: sequential APT progression and POMDP/policy evaluation. The
dataset is passive evidence and cannot establish the causal effect of DROP,
RATE_LIMIT, BLOCK or ISOLATE actions without a controlled enforcement study.
