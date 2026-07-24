import argparse
import json

import pytest

from scripts.extract_teacher_targets import (
    build_jobs,
    choose_depth,
    load_prompt_bank,
    select_cases,
)


def test_load_prompt_bank_accepts_cases_object(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({
        "cases": [{
            "id": "case_a",
            "needle": "The answer is A.",
            "filler": "noise ",
            "question": "what is the answer?",
        }]
    }), encoding="utf-8")

    cases = load_prompt_bank(path)
    assert cases[0]["id"] == "case_a"


def test_select_cases_returns_all_when_requested():
    cases = [{"id": "a"}, {"id": "b"}]
    assert select_cases(cases, True, None) == cases


def test_select_cases_randomly_picks_one_when_not_all():
    class PickLast:
        def choice(self, values):
            return values[-1]

    cases = [{"id": "a"}, {"id": "b"}]
    assert select_cases(cases, False, PickLast()) == [{"id": "b"}]


def test_choose_depth_precedence_fixed_depth():
    args = argparse.Namespace(depth=0.25, depths=[0.75], depth_range=[0.05, 0.95])
    assert choose_depth(args, None) == 0.25


def test_choose_depth_uses_depths_before_range():
    class PickLast:
        def choice(self, values):
            return values[-1]

    args = argparse.Namespace(depth=None, depths=[0.25, 0.75], depth_range=[0.05, 0.95])
    assert choose_depth(args, PickLast()) == 0.75


def test_choose_depth_rejects_invalid_range():
    args = argparse.Namespace(depth=None, depths=None, depth_range=[0.95, 0.05])
    with pytest.raises(ValueError, match="MIN must be <= MAX"):
        choose_depth(args, None)


def test_grid_depth_mode_builds_full_length_case_depth_product():
    args = argparse.Namespace(
        lengths=[64000, 128000], all=True, depth_mode="grid",
        depth=None, depths=[0.1, 0.5, 0.9], depth_range=[0.05, 0.95],
    )
    cases = [{"id": "a"}, {"id": "b"}]
    jobs = build_jobs(args, cases, None, 131072)
    assert len(jobs) == 2 * 2 * 3
    assert {job[2] for job in jobs} == {0.1, 0.5, 0.9}


def test_cycle_depth_mode_balances_depths_without_cartesian_growth():
    args = argparse.Namespace(
        lengths=[64000], all=True, depth_mode="cycle",
        depth=None, depths=[0.1, 0.5, 0.9], depth_range=[0.05, 0.95],
    )
    cases = [{"id": str(index)} for index in range(6)]
    jobs = build_jobs(args, cases, None, 131072)
    assert [job[2] for job in jobs] == [0.1, 0.5, 0.9, 0.1, 0.5, 0.9]
