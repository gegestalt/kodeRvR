# Code provenance corpus

Bulk source corpora are not committed. Create `manifest.csv` with:

```text
sample_id,repository_id,group_id,language,path,label,label_source,generator_family
```

Allowed labels: `human`, `ai`, `hybrid`, `unknown`.

Admissible labelled sources: `declared`, `controlled_generation`,
`verified_agent_account`, `human_reviewed`. `heuristic` and `unlabelled` rows may
be scored but cannot train the classifier.

Every acquired public source must also record URL, immutable revision, license,
acquisition date and content hash. Keep repository/author groups intact and
cluster near duplicates before splitting.
