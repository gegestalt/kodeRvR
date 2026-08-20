# Controlled provenance corpus

This directory is reserved for verified human, AI-generated, and human-edited
AI samples. Live GitHub repositories do not belong here: they are unlabeled
structural or OOD evidence only.

Every controlled manifest loaded by `code_provenance.dataset.load_manifest`
requires:

- `dataset_id`, `dataset_version`, and `dataset_role`
- `repository_id`, `group_id`, and `author_group_id`
- `label`, `label_source`, and `generator_family`
- `provenance_source`, `source_url`, and immutable `source_revision`
- `content_hash`, `license`, and `acquisition_date`

Use `train`, `validation`, or `test` only for verified labeled samples. Use
`ood` or `structural_only` for unlabeled repositories and external structural
data. The loader verifies the SHA-256 hash of each code payload and rejects
heuristic labels, unlabeled training records, and labeled OOD records.

Use `code_provenance.dataset.build_split_plan` before model fitting. It clusters
near-duplicates first and keeps repository, author, dataset, generator, and
language groups disjoint across train, validation, and test partitions. Its
audit payload records the duplicate-shingle width, language assignment, and
disjoint dimensions for reproducibility.