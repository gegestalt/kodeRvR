# AI Code Provenance and Security Intelligence — research design

## Scientific objective

Estimate which available authorship distribution a code change most resembles,
how uncertain that estimate is, how much overlaps a declared public corpus, and
whether the change warrants deeper security review. The system never translates
a classifier probability into proof that a person or model authored code.

## Patch health decision architecture

The PR, or a standalone commit when no PR exists, is the primary decision unit.
Repository, file, symbol, and hunk observations are dependent evidence. The
initial decision engine is deterministic and evaluates:

```text
evidence sufficiency → provenance → intent alignment → functional evidence
→ architectural compatibility → security risk → efficiency risk
→ OOD/evidence quality → review action
```

Downstream confidence cannot exceed materially uncertain prerequisites. Severe
findings remain individually traceable and cannot disappear inside an aggregate
score. Provenance is supporting evidence and cannot independently determine
whether a patch is safe.

`PatchHealthAssessor` is the audited baseline for a later learned dependency
engine. An adaptive model must improve risk-adjusted reviewer utility while
respecting hard safety and abstention constraints.

Architecture evidence initially covers local Python dependency cycles and parse
integrity. These are traceable facts, not a claim that an acyclic repository has
good design. Test-only source is excluded from the production dependency graph,
and both absolute and relative imports are resolved without importing or
executing repository code.

Efficiency evidence compares repeated baseline and candidate measurements for
runtime, peak process RSS, and throughput. The read-only reviewer consumes a
versionable JSON artifact rather than executing untrusted code. Fewer than three
repeats, missing comparable metrics, or invalid measurements produce `unknown`
instead of false precision.

## Labels and admissible evidence

Primary labels are `human`, `ai`, `hybrid`, and `unknown`. Training labels must
come from declared authorship, controlled model generation, verified agent
accounts, or human-reviewed provenance. Heuristic labels may be analyzed but are
forbidden as training ground truth. Generator-family attribution is a separate
task and must include an `unknown` family.

The proposed "organic fraction" is:

```text
(P(human) + 0.5 × P(hybrid)) × (1 − public_reuse_fraction)
```

It is suppressed when the model abstains. This quantity is a transparent index,
not a literal percentage of keystrokes written by a human.

## Leakage-safe evaluation

Random snippet splitting is prohibited. The minimum evaluation is repository-
or author-group disjoint. Stronger protocols hold out language, repository
domain, time period and generator family. Near-duplicate fingerprints are
clustered before splitting so copied templates cannot cross partitions.

Report macro-F1, per-class precision/recall, calibration error, Brier score,
selective risk, coverage, OOD AUROC, and performance under formatting,
identifier-renaming, comment-removal and human-edit perturbations. Bootstrap
confidence intervals use repository or commit—not line—as the sampling unit.

## Public reuse

Public reuse and AI authorship are not opposites: both humans and models reuse
public code. The reuse index therefore remains a separate evidence channel and
must store source URL, revision, license and acquisition date outside model
features. Exact and token-shingle overlap are reported independently.

## Security layer

Static signals identify review surfaces rather than confirmed vulnerabilities.
Production evaluation should integrate CodeQL/Semgrep/Bandit, dependency and
secret scanners, and expert adjudication. Package existence checks must preserve
`unknown` when a registry is unavailable; network failure is not evidence that a
package was hallucinated.

## Research questions

1. Does provenance classification generalize to unseen repositories, languages and generators?
2. Does code + Git/PR behaviour improve over code-only baselines?
3. Can hybrid editing direction be distinguished without duplicate leakage?
4. After matching language, complexity and repository, does provenance correlate with security findings?
5. Can OOD abstention prevent confident claims on unseen generators and domains?
6. Can adaptive review reduce expensive scans and human review while preserving security findings?

## Historical migration

The IPS environment is no longer the active product surface. Its reusable ideas
were calibration, uncertainty, OOD detection, constrained decisions, and
evidence provenance. The former IPS implementation and bulk datasets are not
part of the current working tree.
