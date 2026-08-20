from __future__ import annotations

from dataclasses import replace

import pytest

from code_provenance.dataset import build_split_plan
from code_provenance.schema import AuthorshipLabel, CodeSample, DatasetRole, EvidenceSource


def corpus() -> list[CodeSample]:
    records = []
    for language in ("go", "python", "rust"):
        for index in range(3):
            records.append(CodeSample(
                sample_id=f"{language}-{index}", repository_id=f"repo-{language}-{index}",
                group_id=f"repo-group-{language}-{index}", author_group_id=f"author-{language}-{index}",
                language=language,
                code=(
                    f"{language} function {index} unique token_{language}_{index} "
                    f"value_{index} return result_{language}_{index}"
                ),
                label=AuthorshipLabel.HUMAN if index == 0 else AuthorshipLabel.AI,
                label_source=EvidenceSource.CONTROLLED_GENERATION,
                generator_family=f"generator-{language}-{index}",
                dataset_id=f"dataset-{language}-{index}", dataset_version="v1",
                dataset_role=DatasetRole.TRAIN,
            ))
    return records


def keys(records: tuple[CodeSample, ...], attribute: str) -> set[str]:
    return {str(getattr(item, attribute)) for item in records}


def test_split_plan_separates_required_dimensions_and_languages():
    plan = build_split_plan(corpus(), seed=7)

    partitions = (plan.train, plan.validation, plan.test)
    for index, left in enumerate(partitions):
        for right in partitions[index + 1:]:
            for attribute in ("repository_id", "author_group_id", "dataset_id", "generator_family", "language"):
                assert keys(left, attribute).isdisjoint(keys(right, attribute))
    assert sum(map(len, partitions)) == 9
    assert plan.audit["duplicate_cluster_count"] == 9


def test_duplicate_samples_are_kept_in_one_partition():
    records = corpus()
    duplicate = replace(
        records[0], sample_id="go-duplicate", repository_id="repo-duplicate",
        author_group_id="author-duplicate", dataset_id="dataset-duplicate",
        generator_family="generator-duplicate",
    )
    plan = build_split_plan([*records, duplicate], seed=7)
    locations = [
        partition for partition in (plan.train, plan.validation, plan.test)
        if {item.sample_id for item in partition} >= {"go-0", "go-duplicate"}
    ]
    assert len(locations) == 1


def test_structural_only_and_ood_records_are_rejected():
    record = replace(corpus()[0], dataset_role=DatasetRole.STRUCTURAL_ONLY, label=AuthorshipLabel.UNKNOWN,
                     label_source=EvidenceSource.UNLABELLED)
    with pytest.raises(ValueError, match="training partitions"):
        build_split_plan([record])


def test_language_holdout_requires_three_distinct_language_groups():
    with pytest.raises(ValueError, match="three language groups"):
        build_split_plan(corpus()[:2], seed=1)


def test_language_stratified_mode_keeps_each_language_in_each_partition():
    plan = build_split_plan(corpus(), seed=7, mode="language_stratified")

    for partition in (plan.train, plan.validation, plan.test):
        assert {item.language for item in partition} == {"go", "python", "rust"}
    assert plan.audit["language_protocol"] == "language_stratified"
    assert "language" not in plan.audit["disjoint_dimensions"]


def test_unknown_language_split_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported language split mode"):
        build_split_plan(corpus(), mode="random")