# Dataset Catalog and Local Availability

This file is the referee ledger for dataset claims. A dataset can be on the roadmap without being evaluated; the `current_status` column decides what we are allowed to claim today.

| Dataset | Role | Source | Local path | Expected files | Status | First audit | Blocked claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Controlled baseline for every new method. | <https://www.unb.ca/cic/datasets/nsl.html> | `data/nsl_kdd` | `KDDTrain+.txt` | available | Already complete: preprocessing, supervised baselines, stability. | None for current NSL-KDD baseline; temporal claims remain invalid. |
| CICIoT2023 dev parquet | Fast modern IoT dev sample for quality checks and pilot modelling. | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | `data/ciciot2023` | `*.parquet` | missing | Run src/ciciot.py. | Do not treat dev-sample results as full official raw-release results. |
| CICIoT2023 raw CSV | Primary modern IoT supervised dataset: binary, 8-category, fine-label. | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | `data/ciciot2023/CSV` | `*.csv` | blocked_missing_local_files | Run src/ciciot2023_raw_audit.py. | No full raw CSV modelling or quality claim until CSV files are present. |
| TON_IoT | Multimodal SOC/EDR-style track: network, IoT telemetry, Windows/Linux traces. | <https://research.unsw.edu.au/projects/toniot-datasets> | `data/ton_iot` | `**/*.csv` | blocked_missing_local_files | Create schema inventory for each modality before modelling. | No multimodal fusion, host telemetry, or TON_IoT score is valid yet. |
| CSE-CIC-IDS2018 | Enterprise-scale/day-based drift and chronological evaluation. | <https://www.unb.ca/cic/datasets/ids-2018.html> | `data/cse_cic_ids2018` | `**/*.csv` | available | Complete for the initial 14–16 February subset; see data/cse_cic_ids2018/SOURCE.md. | Only the audited three-day subset is supported; full ten-day and causal IPS claims remain blocked. |
| Common NetFlow schema | Cross-dataset generalization using shared NetFlow-style features. | <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> | `data/netflow` | `**/*.csv` | blocked_missing_local_files | Map feature names and labels before any train-one/test-another experiment. | No cross-dataset score is valid without a common feature schema. |
