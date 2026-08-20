# Adaptive Code Review & Patch Health Intelligence

A research-grade system for evidence-driven patch assessment, repository-aware
risk analysis, security review, architecture analysis, uncertainty handling,
and adaptive AI-assisted review. Code provenance is an optional evidence source,
not the product's decision target.

The system evaluates a code change using independent, traceable evidence and
routes it toward standard review, additional evidence collection, or specialist
human review.

> AI may propose and investigate findings, but review decisions must remain
> grounded in attributable evidence and deterministic safety constraints.

An AI model's opinion is not proof that a patch is correct, secure, efficient,
human-authored, or safe to merge.

## Patch-review pipeline

```text
                            Code change
                                 │
                                 ▼
                        Repository context
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
       Security              Architecture           Functional
       evidence                evidence               evidence
          │                      │                      │
          ├──────────────────────┼──────────────────────┤
          ▼                      ▼                      ▼
      Efficiency             Provenance          OOD / context
       evidence                signals               quality
          └──────────────────────┼──────────────────────┘
                                 ▼
                         Evidence artifacts
                                 │
                                 ▼
                     Integrity + snapshot binding
                                 │
                                 ▼
                    Dependency-aware assessment
                                 │
                                 ▼
                      Explainable review action
```

Current actions are:

- `allow_standard_review`
- `request_targeted_evidence`
- `require_security_review`
- `require_architecture_review`
- `require_human_rewrite_or_validation`
- `block_pending_evidence`

`allow_standard_review` means no available evidence triggered a mandatory
escalation. It does **not** mean safe to merge and is never an automatic merge
authorization.

## Safety kernel

The reviewer operates over dependent evidence instead of collapsing unrelated
signals into one opaque trust score:

```text
Evidence integrity
        ↓
Evidence sufficiency
        ↓
Intent alignment
        ↓
Functional evidence
        ↓
Architectural compatibility
        ↓
Security risk
        ↓
Efficiency risk
        ↓
OOD / evidence quality
```

Provenance can contribute supporting evidence, but uncertain or AI-associated
provenance is not itself a correctness or security failure. Deterministic routing
is the safety kernel: future learned policies may prioritize investigation, but
must not override integrity failures, mandatory security escalation, compliance
rules, or abstention boundaries.

## Evidence and attestation

Review conclusions reference explicit artifacts rather than unstructured model
claims. Artifacts may represent tests, CI, static analysis, architecture,
performance, dependencies, repository context, OOD assessment, or provenance.

| Level | Meaning |
|---|---|
| `asserted` | A caller supplied a claim; the system did not observe it. |
| `observed` | The local process produced and recorded the result. |
| `verified` | An independent trusted producer verified it; planned for CI. |
| `demonstration` | Fixture data exercises a contract, not the scanned patch. |

Evidence integrity and substantive outcome are separate questions. A passing
test report for the wrong code snapshot is rejected. `--run-tests` hashes HEAD,
tracked changes, and untracked contents before running pytest. A pytest hook
then emits structured discovery, selection, deselection, pass, failure, error,
skip, xfail, and xpass outcomes; authoritative counts never depend on parsing
human-readable console output. The report also records the command, framework
version, duration, exit code, deterministic canonical report hash, snapshot
identity, and `observed` attestation. Collection errors, interruption, timeout,
repository mutation, and an empty selected suite cannot pass the functional
evidence gate. Functional PASS applies only to the structured checks that were
actually selected and completed.

Observed local execution is not independently CI-verified evidence. It proves
what this process recorded against a snapshot, not that a trusted remote system
executed the same checks in a controlled environment.

Ledger schema `2.0` uses an explicit `{repository_id, snapshot_id, head_sha}`
target. Artifacts carry canonical payloads and claimed SHA-256 digests; the
ledger recomputes each digest instead of trusting a producer-supplied
`integrity_verified` flag. Asserted, demonstration, incomplete, stale, or
tampered evidence cannot produce a passing integrity result.

The compatibility flags `--tests-passed` and `--tests-failed` are unverified
caller assertions. They remain visible but cannot satisfy mandatory functional
evidence.

## Current implementation

| Component | Purpose |
|---|---|
| `assessment.py` | Dependency-aware assessment and deterministic routing |
| `change_context.py` | Snapshot-bound changed-file, hunk, intent, and repository context |
| `symbol_index.py` | Deterministic Python symbol identity and syntactic change classification |
| `dependency_context.py` | Conservative Python import/call graph and bounded dependent traversal |
| `test_relevance.py` | Deterministic pytest inventory, structural relevance, and observed outcome correlation |
| `snapshot.py` | Deterministic clean/dirty Git code-state identity |
| `pytest_reporter.py` | Hook-generated authoritative pytest outcome report |
| `test_evidence.py` | Snapshot-bound structured pytest evidence and artifact producer |
| `evidence.py` | Artifacts, claims, lineage, integrity, and attestation |
| `architecture.py` | Python dependency and parse-integrity analysis |
| `efficiency.py` | Repeated runtime, RSS, and throughput comparison |
| `evidence_quality.py` | OOD metadata, context coverage, and schema checks |
| `security.py` | Lightweight static security and supply-chain signals |
| `repository.py` | Read-only repository and commit extraction |
| `features.py` | Lexical, AST, and Git-derived features |
| `feature_space.py` | Typed feature metadata plus repository/change aggregate vectors |
| `reuse.py` | Public-code token-shingle reuse analysis |
| `dataset.py` | Verified provenance-corpus contracts |
| `model.py` | Group-safe provenance classification and abstention |
| `report.py` | Repository and patch-health reporting |
| `provenance_cli.py` | Command-line interface |

Patch health works without a provenance model. In that case, the system reports
`UNAVAILABLE_UNTIL_A_LABELLED_GROUP_DISJOINT_MODEL_IS_FITTED` instead of
inventing an authorship percentage.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

.venv/bin/python -m pytest
.venv/bin/python src/provenance_cli.py scan /path/to/repository
```

Run snapshot-bound local tests during assessment:

```bash
.venv/bin/python src/provenance_cli.py scan . \
  --intent "Implement snapshot-bound functional evidence" \
  --run-tests
```

Optional example inputs exercise additional contracts:

```bash
.venv/bin/python src/provenance_cli.py scan . \
  --intent "Exercise the complete assessment contract" \
  --run-tests \
  --efficiency-evidence examples/efficiency_evidence.json \
  --evidence-quality examples/evidence_quality.json \
  --evidence-ledger examples/evidence_ledger.json
```

The example JSON files are **demonstration-only**. They do not prove that the
current snapshot was benchmarked, evaluated by a production OOD detector, or
validated by CI. Snapshot or commit mismatches are integrity failures.

## Interpreting results

`request_targeted_evidence` means the available evidence is insufficient and
the report identifies what is missing. A low-confidence
`allow_standard_review` means no mandatory escalation was found in incomplete
context; it does not mean bug-free, secure, efficient, or human-authored.
Confidence represents evidence quality and context completeness.

## Change context

Every report now includes a typed `change_context` describing the exact
repository snapshot, checked-out HEAD, optional base revision, changed files,
zero-context hunks, supplied intent, repository metadata, completeness, and
missing fields. Working-tree scans intentionally report a missing `base_sha`;
commit-range contexts validate both revisions and require the requested head to
match the checked-out snapshot. Symbol, ownership, dependency, and relevant-test
context are deliberately reserved for separate reviewable increments.

## Changed-symbol context

Python changes are indexed by deterministic identities derived from file path,
qualified name, and symbol kind—not line number. Base/head AST comparison marks
functions, async functions, methods, nested functions, and classes as added,
modified, or deleted; records source ranges, decorators, potential public
visibility, signature/body hashes, and maps changed hunks to containing symbols.

Syntax failures and unsupported rename/copy cases produce partial context rather
than false symbol absence. File renames and symbol moves are not yet resolved as
identity continuity across paths. This subsystem establishes only that symbols
were syntactically affected. It does not claim they are buggy, insecure,
breaking, or high-blast-radius.

## Dependency context

Repository reports include deterministic module, symbol, and unresolved nodes;
typed import, call, decorator, and dynamic-import edges; reverse dependent
queries; and bounded cycle-safe transitive traversal for changed symbols.
Descriptive impact records expose direct dependencies, direct and transitive
dependents, traversal depth, affected-module count, and unresolved references.

Resolution supports Python relative/absolute imports, aliases, direct calls,
module-qualified calls, and selected `self`/`cls` method calls. Star imports,
runtime dispatch, monkey patching, reflection, ambiguous re-exports, dynamic
imports, and unresolved external modules remain visible limitations. Graph
degree is a fact—not a risk score. This layer does not establish bugs, security
risk, breaking changes, test relevance, or production blast radius.

## Test relevance

Python pytest functions and methods are inventoried with deterministic node IDs
and joined to changed symbols through traceable static call paths. Direct,
indirect, and name/path heuristic relations remain distinct; dependency distance
and supporting symbol names are preserved. Traversal is bounded and inherits
the dependency graph's unresolved facts.

Structured pytest evidence now retains per-node outcomes, allowing relevant
tests to be reported as passed, failed, skipped, xfailed, xpassed, mixed, or
`not_observed`. Aggregate success never implies that every relevant test ran.
Incomplete execution, unresolved dependencies, partial symbol parsing, or target
mismatch cannot become false certainty. Relevance is review context—not proof of
coverage, correctness, or behavioral impact—and is not added to the provenance
model or the main patch-health score.

## Feature space and evaluation

The provenance research vector currently contains 60 finite lexical, layout,
identifier, documentation, AST, complexity, annotation, commit-message, and
change-shape features. Every feature has explicit scope, family, producer,
language support, missingness, normalization, leakage risk, reliability, and
scientific-role metadata. Repository and patch aggregates are separate vectors;
they are not flattened into the authorship model.

Repository reports include tracked-file composition, text size, file-size
distribution, test-file ratio, extension diversity, and language entropy.
Change reports include file and hunk counts, additions, deletions, churn,
subsystem dispersion, test-file ratio, and binary-file count.

Group-disjoint model evaluation reports macro and weighted F1, balanced
accuracy, MCC, per-class precision/recall/F1/support, confusion counts, log
loss, multiclass Brier score, ten-bin expected calibration error, ROC-AUC,
PR-AUC, and a deterministic group-bootstrap confidence interval. These are
research diagnostics, not evidence that style proves authorship.

Symbol graph, dependency centrality, duplication, changed-code coverage,
ownership history, learning curves, seed stability, permutation importance,
selective-risk curves, OOD benchmark metrics, and resource profiling remain
explicit follow-up experiments rather than fabricated placeholders.

## Security scope

The bundled analyzer is intentionally lightweight. It surfaces selected unsafe
APIs, credential-like literals, and supply-chain signals, but does not replace
CodeQL, Semgrep, secret scanners, dependency vulnerability scanners, or
language-specific analyzers. Future integrations should preserve their findings
as distinct evidence artifacts rather than hiding them in one score.

## Engineering contract

Implementation follows test-driven development:

1. Add a failing behavioral or contract test.
2. Implement the smallest change that passes it.
3. Refactor while focused tests remain green.
4. Run the complete regression suite.
5. Document observable behavior.

Every defect fix requires a regression test. Randomness must be seeded, evidence
fixtures reproducible, and no feature is complete while its tests fail.

## External test fixtures

The test system has three explicit tiers. The default is deterministic and
offline; it never contacts an upstream service:

```bash
.venv/bin/python -m pytest

# Verify/evaluate an existing pinned cache, with no network access.
.venv/bin/python -m pytest -m evaluation

# Explicitly enable bounded network-backed tests.
.venv/bin/python -m pytest -m live_data --live-data-limit 3
# Equivalent explicit gate:
.venv/bin/python -m pytest --run-live-data -m live_data --live-data-limit 3
```

Fetch and independently verify a small categorized slice without downloading
complete corpora:

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_test_fixtures.py --dataset codeql --limit 3
PYTHONPATH=src .venv/bin/python scripts/fetch_test_fixtures.py --verify-only
PYTHONPATH=src .venv/bin/python scripts/evaluate_external_fixtures.py
```

The ignored `.cache/code_health/` directory contains SWE-bench Lite issue/patch
cases, DevGPT AI-association records, and official GitHub CodeQL query tests.
`--dataset` is repeatable; `--refresh` is explicit. Existing verified entries
are not overwritten, and altered cache entries fail SHA-256 verification. The
fetcher, immutable revisions, reviewed regression baseline, category contract, and
[scientific-use table](data/code_health/test_fixtures/SOURCE.md) are versioned.
These categories are not interchangeable and none supplies defensible human/AI
authorship ground truth.

SWE-bench is patch/test-outcome evidence; this layer validates ingestion but does
not claim upstream repositories were executed. DevGPT establishes an associated
AI-development trace and never creates `HUMAN`, `AI`, or `HYBRID` labels. CodeQL
fixtures are query-specific oracles: current misses stay measured as misses. A
network timeout, rate limit, schema change, invalid revision, or hash mismatch is
an acquisition failure—not an analyzer result.

Fetched repositories and payloads are untrusted. The live structural check uses
only a manifest-pinned public repository and performs static snapshot/change/
symbol/dependency analysis. Downloaded code is not executed.

## Provenance research boundary

The optional organic-code index is:

```text
(P(human) + 0.5 × P(hybrid)) × (1 − public_reuse_fraction)
```

It is suppressed when the model abstains. It is a model-derived index, not a
measurement of human keystrokes. Public reuse remains separate because humans
and models both reuse public code. Labels must come from declared authorship,
controlled generation, verified coding-agent accounts, or human review.
Repository/author groups and near-duplicate clusters must remain split-disjoint.

Corpus metadata belongs in `data/code_provenance/manifest.csv`; see the
[data contract](data/code_provenance/README.md). The leakage-controlled protocol
is in the [research design](docs/CODE_PROVENANCE_RESEARCH_DESIGN.md), with an
interactive [notebook](notebooks/07_ai_code_provenance_security_lab.ipynb).

## Roadmap

1. Produce independently verified CI attestations bound to immutable commits
   and artifact hashes.
2. Add adapters for CodeQL, Semgrep, secret scanning, dependency audits, and
   language-native test and coverage reports.
3. Extend the ChangeContext core with changed-symbol, ownership, dependency,
   relevant-test, and blast-radius indexes while retaining PR/commit as the
   primary decision unit.
4. Add an AI investigator that explains evidence, retrieves context, and
   proposes checks without controlling the safety kernel.
5. Learn an adaptive policy for analyzer selection, context retrieval,
   prioritization, and escalation cost—not merge authority.
6. Evaluate calibration, abstention, false-negative risk, latency, resources,
   and policy behavior on group-disjoint repositories and realistic patches.

## Project skill

Invoke `$grill-me` before a major architecture or research decision. The
project-scoped skill inspects the repository, maps dependent decisions, and asks
one focused question at a time before implementation.
