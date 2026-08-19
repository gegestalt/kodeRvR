# Network Intrusion Detection — an ML study (NSL-KDD → NetFlow-V2)

An end-to-end, test-driven machine-learning study of **network intrusion
detection**, framed from a SOC (Security Operations Centre) analyst's
perspective. **`v1.0` delivers the complete NSL-KDD pipeline**; the project then
extends to modern, standardized **NetFlow-V2** datasets for a cross-dataset
generalization study.

> Built as an MSc-AI portfolio piece. Every modelling decision is documented and
> interview-defensible; plain-language learning notes are in [`docs/learning/`](docs/learning/).

---

## Why ML for intrusion detection (the SOC view)

Signature/rule-based detection (the bread and butter of tools like Splunk,
CrowdStrike, SentinelOne) is precise but brittle: it only catches what someone
already wrote a rule for, and attackers mutate constantly. Supervised ML learns
the *statistical shape* of attacks from labelled traffic, offering first-pass
triage that can generalize beyond exact signatures. The interesting question
isn't "can it hit high accuracy" — it's **where it fails, and whether it
degrades gracefully as attacks evolve**. That failure analysis is the core of
this project.

## Headline results (NSL-KDD, official KDDTest+)

| Model | Binary macro-F1 | 5-class macro-F1 | R2L recall | U2R recall |
|---|---:|---:|---:|---:|
| Random Forest | 0.776 | 0.504 | 0.05 | 0.06 |
| LightGBM | 0.785 | 0.281 | 0.001 | 0.00 |
| MLP (unweighted) | **0.810** | **0.563** | 0.013 | 0.269 |
| MLP (class-weighted) | 0.798 | 0.561 | **0.122** | **0.537** |

**Reproducibility (5 seeds, CPU — see [`results/stability.md`](results/stability.md)):**
RF 5-class macro-F1 **0.502 ± 0.008**; MLP-unweighted **0.577 ± 0.046** (high
variance — a single run is unreliable); MLP-weighted **0.568 ± 0.012** with U2R
recall **0.51 ± 0.06** vs unweighted **0.21 ± 0.12**. Takeaway: class weighting
does *not* significantly change macro-F1, but it **reliably** rescues rare-class
recall. Report mean ± std, never a single run.

**Three findings that matter more than the numbers:**

1. **The generalization gap is real and deliberate.** Models scored **~0.99
   macro-F1 in cross-validation** but **~0.78 on the official test set** — because
   the test set contains **17 attack types never seen in training**. A random
   split would hide this; the official split honestly measures how detection
   copes with *novel* attacks. This is the "detection goes stale" problem in one
   number.
2. **Rare = dangerous = missed.** With default tree training, the two most dangerous
   attack families — R2L (remote access) and U2R (privilege escalation) — are
   caught **5–6% of the time**, with ~90% predicted as benign. Accuracy (0.74)
   hides this; **macro-F1 (0.50) exposes it.**
3. **Class weighting changes the trade-off, not magically fixes the benchmark.**
   On the latest stable CPU run, weighting the MLP loss lifted **R2L recall from
   0.013→0.122** and **U2R recall from 0.269→0.537**, while macro-F1 stayed about
   flat (0.563→0.561). For a SOC, that is still an important knob: you may accept
   different false alarms if the alternative is missing privilege-escalation
   attacks.

<p align="center">
  <img src="results/figures/p3_nslkdd_randomforest_multiclass_test_confusion.png" width="90%"><br>
  <em>Baseline RF: R2L/U2R collapse into "normal" (recall 0.05 / 0.06).</em><br><br>
  <img src="results/figures/p4_nslkdd_mlp_weighted_multiclass_confusion.png" width="90%"><br>
  <em>Class-weighted MLP: rare-class recall improves, especially U2R, but the result still needs multi-seed confirmation.</em>
</p>

## Datasets

| Dataset | Era | Role | Split | Status |
|---|---|---|---|---|
| **NSL-KDD** | 1999 | historical baseline | official (+ hard `KDDTest-21`) | ✅ `v1.0` |
| **NF-UNSW-NB15-v2** | 2015 | modern (imbalance study) | official | ▶ upcoming |
| **NF-ToN-IoT-v2** | 2020 | modern IoT (explainability) | official | ▶ upcoming |
| **NF-CSE-CIC-IDS2018-v2** | 2018 | modern at scale | constructed | ▶ upcoming |

The modern datasets are the **NetFlow-V2 standardized family** (Sarhan et al.,
2021), which re-express multiple captures into **one shared 43-feature NetFlow
schema** — enabling *true cross-dataset transfer* (train on one, test on another),
the cleanest form of the staleness experiment.

> **Migration note.** Earlier iterations used the raw UNSW-NB15 and CICIDS2017
> CSVs. These are **retired** in favour of their NetFlow-V2 versions (shared
> schema, cleaner labels, comparable methodology). Their download provenance is
> preserved in `data/unsw_nb15/SOURCE.md` and `data/cicids2017/SOURCE.md`.

Data is downloaded + integrity-verified programmatically (row/column counts +
SHA-256); see `src/download_data.py` and `data/*/SOURCE.md`. Bulk data is
gitignored.

## Approach

1. **EDA** ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)) — class balance,
   the train→test shift, feature separation, correlations.
2. **Preprocessing** (`src/preprocess.py`) — a leakage-safe
   `ColumnTransformer`: one-hot (`handle_unknown='ignore'`) for categoricals,
   `StandardScaler` for numerics, **fit on train only**. Binary + 5-class labels.
3. **Tree baselines** (`src/train_baselines.py`) — Random Forest + **LightGBM**,
   light CV tuning (train-only), full evaluation.
4. **MLP** (`src/train_mlp.py`) — PyTorch net (128-64, dropout, early stopping,
   CPU/Apple-MPS), with the **class-weighting ablation**.
5. **Evaluation** (`src/evaluate.py`) — one model-agnostic module: macro-F1 +
   per-class recall (never accuracy alone), confusion matrices (raw +
   row-normalized), ROC-AUC + PR-AUC for binary.

**Why the official split, not a random one:** the official `KDDTest+` withholds
attack types from training, simulating real-world novelty. A random re-split
would make train and test identically distributed and inflate every score.

## Reproduce

```bash
# 1. environment (Python 3.14)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only (LightGBM/XGBoost OpenMP runtime):
brew install libomp

# 2. data (NSL-KDD; ~22 MB, verified on download)
python src/download_data.py nsl_kdd

# 3. run the pipeline
python src/train_baselines.py     # Phase 3 — RF + LightGBM  (~90s)
python src/train_mlp.py           # Phase 4 — MLP ablation    (~70s)

# 4. tests (28, TDD)
pytest
```

Outputs land in `results/metrics.md` and `results/figures/`. Everything is
deterministic under `RANDOM_STATE = 42`. On macOS, `src/train_baselines.py`
also pins native OpenMP thread counts to avoid LightGBM/libomp crashes from
nested parallelism.

## Full interactive project lab

The single end-to-end notebook is
[`notebooks/06_adaptive_ips_full_project_lab.ipynb`](notebooks/06_adaptive_ips_full_project_lab.ipynb).
Its 22-part main story behaves like an interactive textbook: traffic and data
first; campaign and health; chronology; the simplest detector ladder; failure
diagnosis; network/provenance fusion; hidden-state belief; IPS actions; rules,
bandits and RL; fair policy comparison; safety, OOD, calibration and robustness;
a grounded optional LLM analyst; one final scoreboard; and a standalone
"what did we prove?" page. Every experiment states its question, operational
purpose, expected behavior, observed result, judgement, and next question.

The prior 79-section development notebook is preserved as a 162-cell
**Historical and Diagnostic Appendix**. It retains source audits, synthetic and
real-data controls, parameter grids, multi-seed statistics, OPE limitations,
resource profiles and detailed CICAPT diagnostics without forcing a new reader
to traverse the project chronologically. The main scoreboard never mixes
incompatible flow-row, one-minute-window, passive-observation, or counterfactual
policy evidence.

The IPS package structure and dataset responsibilities are documented in
[`docs/IPS_ARCHITECTURE.md`](docs/IPS_ARCHITECTURE.md). Use the unified command
surface instead of remembering individual scripts:

```bash
.venv/bin/python src/ips_cli.py status
.venv/bin/python src/ips_cli.py cicapt-audit
.venv/bin/python src/ips_cli.py cicapt-profile
.venv/bin/python src/ips_cli.py cicapt-build
.venv/bin/python src/ips_cli.py cicapt-benchmark
.venv/bin/python src/ips_cli.py cicapt-source-audit
.venv/bin/python src/ips_cli.py cicapt-fusion
.venv/bin/python src/ips_cli.py cicapt-data-intelligence
.venv/bin/python src/ips_cli.py cse-build
.venv/bin/python src/ips_cli.py cse-benchmark
.venv/bin/python src/ips_cli.py claim-control
.venv/bin/python src/ips_cli.py next-phase
```

The real CICAPT acquisition is now integrated. Phase 1 contains 12,062,396
benign network rows; phase 2 contains 9,536,823 rows with 1,004 labelled attack
rows across eight tactics. The attack-preserving chronological benchmark uses
December 1 for training, December 2 for validation, December 3 as a development
test, and keeps December 4 locked. Dataset-supplied feature-extraction scripts
are statically audited and used as schema/formula provenance; unsafe shell and
file-deletion orchestration is not executed by the project.

The CICAPT Data Intelligence evidence layer adds an auditable
dataset card and attrition funnel; feature health, missingness, imbalance,
tactic/technique coverage, campaign and cross-modal timelines; PSI/KS/JSD/
Wasserstein drift, attack separation, correlation redundancy, independent
benign-sampling fidelity, clock-alignment and join-tolerance sensitivity;
provenance graph dynamics, unit-labelled detector/fusion results, threshold
curves, explicit UNKNOWN/OOD behavior, attack-only confusion, confidence and
risk-coverage diagnostics, and an independent policy-readiness gate. December 4
remains an untouched locked holdout.

Late fusion currently improves development PR-AUC over either individual
modality, but absolute attack recall and tactic macro-F1 remain weak.
Calibration improves Brier/ECE while collapsing recall at a naive 0.5
threshold, and the tactic belief is overconfident on attack windows. The
notebook therefore blocks RL/policy claims until temporal calibration and
belief quality pass the declared gates; this is a scientific result, not a
hidden failure.

```bash
# Start Jupyter and open notebook 06
.venv/bin/jupyter lab

# Or execute its quick path non-interactively
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/06_adaptive_ips_full_project_lab.ipynb \
  --output results/notebook_ips_lab/executed.ipynb \
  --ExecutePreprocessor.timeout=300
```

The main teaching path reads deterministic saved evidence and executes quickly;
the historical appendix reruns the complete diagnostic suite. The optional
`EvidenceAnalyst` needs no API key and falls back to deterministic Markdown.
An external analyst callable is permitted only as an explanatory component and
is rejected if it introduces numerical claims absent from its structured
evidence object.

To build the parquet file with group-disjoint out-of-fold detector scores, see
[`docs/datasets/ips_events.md`](docs/datasets/ips_events.md).

## Limitations (read these)

- **NSL-KDD is old and synthetic** (derived from 1998 simulated traffic).
  Absolute scores here do **not** transfer to modern networks — the value is the
  *methodology* and *relative* model comparison. (The NetFlow-V2 phase addresses
  modernity directly.)
- **Some classes may be unlearnable.** 52 U2R training examples is likely too few
  for reliable detection; near-zero baseline recall there is an honest finding.
- **Single fixed test split** — one draw, so headline numbers carry variance
  (mitigated by CV *during tuning*).
- **Cross-*dataset* transfer isn't possible with heterogeneous feature spaces** —
  which is exactly why the project moves to the shared-schema NetFlow-V2 data.

## Next steps

- Download + integrate the **NetFlow-V2 trio**; run each with a *distinct* tactic
  (imbalance arsenal / SHAP explainability / big-data scaling).
- **Cross-dataset transfer** experiment on the shared 43-feature schema — the
  headline "detection goes stale" result.
- Class-imbalance techniques beyond weighting (SMOTE, cost-sensitive), threshold
  tuning, and probability calibration.

## Project structure

```
src/            download_data.py · data.py · preprocess.py · evaluate.py
                train_baselines.py · train_mlp.py
tests/          28 pytest tests (schema, leakage, metrics, MLP)
notebooks/      01_eda.ipynb
results/        metrics.md · figures/
docs/           PROGRESS.md · learning/00-03 (plain-language explainers)
.claude/skills/ nids-conventions  (project conventions)
```

## Attribution

Datasets © their respective authors — cite **Tavallaee et al.**, *A Detailed
Analysis of the KDD CUP 99 Data Set* (IEEE CISDA 2009) for NSL-KDD, and
**Sarhan et al.**, *Towards a Standard Feature Set for NIDS Datasets* (2021) for
NetFlow-V2. Code under the MIT License (see [`LICENSE`](LICENSE)).
