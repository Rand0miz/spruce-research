import argparse
import json

import pytest

from scripts.extract_teacher_targets import choose_depth, load_prompt_bank, select_cases


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
