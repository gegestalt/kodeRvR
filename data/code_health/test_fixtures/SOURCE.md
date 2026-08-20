# Categorized external test fixtures

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_test_fixtures.py --limit 3
```

The generated `raw/` directory is local and ignored by Git. Every fetched item
is hashed and catalogued with an immutable upstream revision.

| Category | Source | Valid scientific use | Invalid interpretation |
|---|---|---|---|
| `issue_patch_correctness` | SWE-bench Lite | Issue-to-patch correctness and test-outcome evaluation | Human or AI authorship label |
| `ai_assisted_trace` | DevGPT | Association between a shared ChatGPT conversation and a development artifact | Proof that linked code was AI-generated |
| `security_analyzer_oracle` | GitHub CodeQL query tests | Rule-specific security analyzer regression tests | Universal vulnerability or patch-quality label |
| `clean_control` | Controlled local fixtures | Negative-control and false-positive testing | Evidence about public development behavior |
| `adversarial_integrity` | Controlled local fixtures | SHA mismatch, stale artifact, partial scan, and corruption tests | Real-world prevalence estimate |

## Upstream revisions

- SWE-bench Lite dataset: `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`
- DevGPT repository: `685efd2509dede9a6e996b839ae4e20d33430648`
- GitHub CodeQL repository: `87c77cc26ccd1d2d9791b8563be6d425ccdf0874`

DevGPT currently has no machine-readable repository license assertion. Its
download is retained only in the ignored local cache and categorized as
metadata-only research input pending license review.
