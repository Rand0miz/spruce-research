import pytest

from scripts.select_screened_prompts import (
    balanced_quotas,
    select_balanced,
    selection_summary,
)


def _record(case_id, length, depth, variant):
    identifier = f"{case_id}-{length}-{depth}-{variant}"
    return {
        "candidate_id": identifier,
        "source_case_id": case_id,
        "requested_length": length,
        "depth": depth,
        "variant": variant,
        "seed": 100 + variant,
        "status": "completed",
        "accepted": True,
    }


def test_balanced_quotas_redistribute_when_one_stratum_is_small():
    quotas = balanced_quotas({"a": 2, "b": 10, "c": 10}, 12)
    assert quotas == {"a": 2, "b": 5, "c": 5}


def test_select_balanced_is_exact_deterministic_and_case_diverse():
    records = [
        _record(case, length, depth, variant)
        for length in (16_384, 32_768)
        for depth in (0.1, 0.5, 0.9)
        for case in ("a", "b", "c", "d")
        for variant in range(3)
    ]
    first, quotas = select_balanced(records, count=48, seed=7)
    second, _ = select_balanced(records, count=48, seed=7)
    assert [row["candidate_id"] for row in first] == [
        row["candidate_id"] for row in second]
    assert len(first) == 48
    assert len({row["candidate_id"] for row in first}) == 48
    assert set(quotas.values()) == {8}
    summary = selection_summary(first)
    assert set(summary["by_case"].values()) == {12}


def test_select_balanced_rejects_insufficient_accepted_prompts():
    with pytest.raises(ValueError, match="only 1"):
        select_balanced(
            [_record("a", 16_384, 0.1, 0)], count=2, seed=0)


def test_select_balanced_enforces_global_case_cap():
    records = [
        _record(case, length, depth, variant)
        for length in (16_384, 32_768)
        for depth in (0.1, 0.5, 0.9)
        for case in ("a", "b", "c", "d")
        for variant in range(2)
    ]
    selected, quotas = select_balanced(
        records, count=12, seed=7, max_per_case=3)
    summary = selection_summary(selected)
    assert len(selected) == 12
    assert set(quotas.values()) == {2}
    assert set(summary["by_case"].values()) == {3}


def test_select_balanced_rejects_infeasible_case_cap():
    records = [
        _record(case, 16_384, 0.5, variant)
        for case in ("a", "b")
        for variant in range(3)
    ]
    with pytest.raises(ValueError, match="max-per-case"):
        select_balanced(
            records, count=5, seed=0, max_per_case=2)
