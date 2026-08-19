# Adaptive IPS code architecture

The notebook is the presentation layer, not the implementation layer. It reads
saved evidence and calls small public APIs; reusable behavior belongs under
`src/ips/`.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| Domain | `src/ips/actions.py`, `reward.py`, `dataset.py`, `belief.py` | Stable action, reward, event and belief contracts |
| Environments | `src/ips/environment.py`, `dataset_environment.py` | Simulator and dataset replay transitions |
| Policies | `src/ips/dqn.py`, `advanced_policies.py`, `policies.py` | DQN, LinUCB, PPO and deterministic baselines |
| Dataset adapters | `src/ips/adapters/` | Dataset-specific parsing, chronology and leakage gates |
| Analysis | `src/ips/analysis/` | Evidence tables, diagnostics, calibration, provenance, holdout and claim gates |
| Workspace | `src/ips/workspace.py` | Central paths and non-destructive dataset discovery |
| Experiment runners | `src/experiments/ips/` | Dataset-specific orchestration and saved evidence only |
| Interface | `src/ips_cli.py` | One command surface for status, audit and experiment runs |
| Presentation | `notebooks/06_adaptive_ips_full_project_lab.ipynb` | Tables, plots, interpretations and decisions |

## Dataset responsibilities

- CSE-CIC-IDS2018: enterprise chronological detector benchmark.
- CICAPT-IIoT2024: ordered APT campaign, provenance/network late fusion, POMDP policy benchmark.
- CICIoT2023: device- and attack-family-held-out IoT detector development.
- IoT-23: malware-capture external validation only.
- TON_IoT: cross-domain and future multimodal generalization.
- NSL-KDD: legacy regression/reference track only.

Datasets are never concatenated merely because they are tabular. Combining
sources requires a documented semantic intersection, group-disjoint split and
source-identifiability audit.

## Commands

```bash
# Fast readiness check; does not read bulk data
.venv/bin/python src/ips_cli.py status

# After all CICAPT modalities finish downloading
.venv/bin/python src/ips_cli.py cicapt-audit
.venv/bin/python src/ips_cli.py cicapt-profile
.venv/bin/python src/ips_cli.py cicapt-build
.venv/bin/python src/ips_cli.py cicapt-benchmark
.venv/bin/python src/ips_cli.py cicapt-source-audit
.venv/bin/python src/ips_cli.py cicapt-fusion
.venv/bin/python src/ips_cli.py cicapt-data-intelligence

# CSE chronological evidence and policy benchmark
.venv/bin/python src/ips_cli.py cse-build
.venv/bin/python src/ips_cli.py cse-benchmark

# Rebuild claim-control and detector/POMDP evidence
.venv/bin/python src/ips_cli.py claim-control
.venv/bin/python src/ips_cli.py next-phase
```

The workspace accepts either `CICAPT-IIoT2024/` at repository root or the
canonical `data/cicapt_iiot2024/raw/`. Discovery is read-only; files are not
moved while a download may still be active. Readiness requires a network CSV,
provenance CSV and `Attack_info.csv`.
