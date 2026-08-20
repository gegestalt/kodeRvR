# AI Code Provenance & Security Intelligence

A research-grade AI codebase health reviewer for dependency-aware patch trust,
source-code provenance, public-code reuse, OOD-aware uncertainty, and security
review intelligence.

The system estimates whether code is statistically consistent with verified
`human`, `ai`, or `hybrid` training distributions. It reports `unknown` when
confidence is insufficient or the sample is out of distribution. It never
claims that coding style proves who—or which model—wrote code.

## What it analyzes

```text
Git repository
├── code DNA: lexical style, tokens, comments and identifiers
├── structure: Python AST and conservative cross-language syntax features
├── Git DNA: commit message and change-shape evidence
├── reuse: token-shingle overlap with an audited public corpus
└── security: unsafe API, secret and dependency review signals
                         ↓
       calibrated human / AI / hybrid probabilities
                         ↓
             uncertainty + OOD abstention
                         ↓
       explained provenance and security-review report
```

## Scientific claim boundary

The proposed organic-code index is:

```text
(P(human) + 0.5 × P(hybrid)) × (1 − public_reuse_fraction)
```

It is suppressed when the model abstains. This is a transparent model-derived
index, not a literal measurement of human keystrokes. Public reuse is kept
separate from authorship because both humans and models reuse public code.

Training labels must come from declared authorship, controlled generation,
verified coding-agent accounts, or human-reviewed provenance. Heuristic labels
are rejected as training ground truth. Repository/author groups and
near-duplicate clusters must remain disjoint across splits.

## Current implementation

| Component | Purpose |
|---|---|
| `assessment.py` | Deep patch-level health assessment and explainable review routing |
| `architecture.py` | Conservative Python dependency-cycle and parse-integrity evidence |
| `efficiency.py` | Repeated baseline-versus-candidate runtime, RSS, and throughput evidence |
| `evidence_quality.py` | Named OOD detector, context coverage, and schema-support gate |
| `evidence.py` | Versioned commit-bound evidence artifacts, claims, lineage, and integrity |
| `repository.py` | Read-only tracked-file and commit extraction |
| `features.py` | Interpretable lexical, AST and Git features |
| `reuse.py` | Local token-shingle public-reuse index |
| `dataset.py` | Verified-label corpus contract |
| `model.py` | Group-safe classification, calibration and OOD abstention |
| `security.py` | Static review and supply-chain signals |
| `report.py` | Repository-level descriptive report |
| `provenance_cli.py` | Command-line entry point |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/provenance_cli.py scan /path/to/repository
python src/provenance_cli.py scan . \
  --intent "Implement dependency-aware patch health review" \
  --tests-passed \
  --efficiency-evidence examples/efficiency_evidence.json \
  --evidence-quality examples/evidence_quality.json \
  --evidence-ledger examples/evidence_ledger.json
pytest
jupyter lab notebooks/07_ai_code_provenance_security_lab.ipynb
```

The scan command intentionally returns
`UNAVAILABLE_UNTIL_A_LABELLED_GROUP_DISJOINT_MODEL_IS_FITTED` instead of an AI
percentage when no verified model exists.

The `patch_health` result remains available without a fitted provenance model.
It evaluates dependent evidence and returns an explainable review action.
Unknown provenance does not escalate an otherwise healthy patch. Missing intent,
tests, architecture, efficiency, repository context, or OOD evidence lowers
confidence and remains visible in the decision path.

## Engineering contract

Implementation follows test-driven development:

1. Add a failing behavioral or contract test.
2. Implement the smallest change that passes it.
3. Refactor while the focused test remains green.
4. Run the complete regression suite.
5. Update documentation with the behavior.

Every defect fix requires a regression test. Randomness must be seeded, evidence
fixtures must be reproducible, and no feature is complete while its tests fail.

## External test fixtures

Fetch a small, categorized test slice without downloading complete corpora:

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_test_fixtures.py --limit 3
.venv/bin/python -m pytest
```

The local cache contains three SWE-bench Lite issue/patch cases, three DevGPT
AI-association records, and three official GitHub CodeQL security query-test
files. The cache is ignored; the fetcher, immutable revisions, hashes, category
contract, and [scientific-use table](data/code_health/test_fixtures/SOURCE.md)
are versioned.

These categories are intentionally not interchangeable. SWE-bench supports
correctness evaluation, DevGPT supports AI-link association analysis, and
CodeQL fixtures support analyzer regression. None of them supplies defensible
human/AI authorship ground truth.

The current lightweight security regexes do not detect the fetched CodeQL SQL
injection oracle. This is recorded as analyzer coverage evidence, not hidden by
changing fixture labels.

## Data contract

Place corpus metadata at `data/code_provenance/manifest.csv`; see
[`data/code_provenance/README.md`](data/code_provenance/README.md). Bulk code is
ignored. Every public source needs its URL, immutable revision, license,
acquisition date and content hash.

## Research protocol

The complete design and leakage controls are documented in
[`docs/CODE_PROVENANCE_RESEARCH_DESIGN.md`](docs/CODE_PROVENANCE_RESEARCH_DESIGN.md).
The interactive walkthrough is
[`notebooks/07_ai_code_provenance_security_lab.ipynb`](notebooks/07_ai_code_provenance_security_lab.ipynb).

## Project skill

Invoke `$grill-me` (or ask to be grilled) before a major architecture or research
decision. The project-scoped skill inspects the repository first, maps the design
tree, and asks one decision question at a time with a recommended answer. It does
not implement anything until shared understanding is explicitly confirmed.

## Current limitation

The patch-health interface and tested modelling contracts are implemented, but no
verified authorship corpus is bundled. Consequently, the repository can be
assessed with deterministic evidence rules today, but a scientifically
defensible provenance distribution remains blocked until controlled data is
acquired.
