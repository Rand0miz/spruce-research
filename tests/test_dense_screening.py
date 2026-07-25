import json

import pytest

from scripts.extract_teacher_targets import (
    VERIFIED_MANIFEST_KIND,
    build_verified_jobs,
    load_prompt_bank,
    load_verified_manifest,
)
from scripts.screen_natural_prompts import (
    build_candidates,
    candidate_id,
    summarize_candidates,
    validate_resume_config,
)


def _case(case_id):
    return {"id": case_id}


def _record(identifier, *, accepted):
    prompt = "Background. The answer is Cedar Spring.\n\nQuestion?"
    return {
        "candidate_id": identifier,
        "id": identifier,
        "source_case_id": "source",
        "builder": "verified_prompt_v1",
        "status": "completed",
        "accepted": accepted,
        "model": "model",
        "rope": {"factor": 1.0},
        "requested_length": 16_384,
        "seq_len": len(prompt),
        "block_size": 64,
        "needle_block": 0,
        "depth": 0.5,
        "needle": "The answer is Cedar Spring.",
        "question": "\n\nQuestion?",
        "answers": ["Cedar Spring"],
        "prompt_text": prompt,
    }


def test_candidate_grid_is_deterministic_and_unique():
    cases = [_case("a"), _case("b")]
    candidates = build_candidates(
        cases, [16_384, 32_768], [0.1, 0.9], variants=2, seed=10)
    assert len(candidates) == 2 * 2 * 2 * 2
    assert len({item["candidate_id"] for item in candidates}) == len(candidates)
    assert candidates[0]["candidate_id"] == candidate_id(
        "a", 16_384, 0.1, 10)
    assert candidates[1]["seed"] == 11


def test_summary_reports_acceptance_by_length():
    records = [
        {**_record("a", accepted=True), "requested_length": 16_384},
        {**_record("b", accepted=False), "requested_length": 16_384},
        {**_record("c", accepted=True), "requested_length": 32_768},
    ]
    summary = summarize_candidates(records)
    assert summary["completed"] == 3
    assert summary["accepted"] == 2
    assert summary["acceptance_rate"] == pytest.approx(2 / 3)
    assert summary["by_length"]["16384"]["accepted"] == 1


def test_verified_manifest_loads_only_dense_correct_prompts(tmp_path):
    path = tmp_path / "screen.json"
    accepted = _record("accepted", accepted=True)
    rejected = _record("rejected", accepted=False)
    path.write_text(json.dumps({
        "kind": VERIFIED_MANIFEST_KIND,
        "candidates": [accepted, rejected],
    }), encoding="utf-8")

    records = load_verified_manifest(path)
    assert [record["id"] for record in records] == ["accepted"]
    jobs = build_verified_jobs(records, maximum_context=32_768)
    assert jobs == [(16_384, accepted, 0.5)]


def test_verified_manifest_rejects_prompt_without_needle(tmp_path):
    record = _record("bad", accepted=True)
    record["prompt_text"] = "No evidence here.\n\nQuestion?"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "kind": VERIFIED_MANIFEST_KIND,
        "candidates": [record],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain its needle"):
        load_verified_manifest(path)


def test_verified_manifest_accepts_exact_chat_prompt(tmp_path):
    record = _record("chat", accepted=True)
    user_prompt = record["prompt_text"]
    record.update({
        "prompt_format": "qwen_chat_v1",
        "user_prompt_text": user_prompt,
        "prompt_text": f"<user>{user_prompt}</user><assistant>",
    })
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({
        "kind": VERIFIED_MANIFEST_KIND,
        "candidates": [record],
    }), encoding="utf-8")

    assert load_verified_manifest(path)[0]["id"] == "chat"


def test_resume_rejects_configuration_drift():
    config = {"model": "one", "lengths": [16_384]}
    validate_resume_config({"config": config}, config)
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_config(
            {"config": config}, {"model": "two", "lengths": [16_384]})


def test_natural_training_and_heldout_banks_are_disjoint():
    training = load_prompt_bank("scripts/prompt_banks/natural_train.json")
    heldout = load_prompt_bank("scripts/prompt_banks/natural_heldout.json")
    assert {case["id"] for case in training}.isdisjoint(
        case["id"] for case in heldout)
