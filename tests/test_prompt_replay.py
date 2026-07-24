import json

from eval.haystack import build_haystack, build_haystack_calibrated
from teacher.prompt_replay import reconstruct_teacher_prompt


class CharacterTokenizer:
    def __call__(self, text):
        return {"input_ids": list(text)}


def test_reconstruct_teacher_prompt_reuses_extraction_logic(tmp_path):
    case = {
        "id": "needle_case", "needle": "NEEDLE", "filler": "fill ",
        "question": "\nQ?",
    }
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    tok = CharacterTokenizer()
    target_tokens = 100
    prompt, _ = build_haystack(
        tok, target_tokens - len(tok(case["question"])["input_ids"]),
        case["needle"], 0.37, case["filler"],
    )
    expected = prompt + case["question"]
    target = {
        "prompt_bank": "train", "case_id": case["id"], "needle": case["needle"],
        "question": case["question"], "depth": 0.37,
        "seq_len": len(tok(expected)["input_ids"]), "block_size": 4,
        "needle_block": len(tok(prompt[:prompt.index(case["needle"])])["input_ids"]) // 4,
        "haystack_token_budget": target_tokens - len(tok(case["question"])["input_ids"]),
    }
    actual, meta = reconstruct_teacher_prompt(tok, target, bank_path=bank)
    assert actual == expected
    assert meta is target


def test_legacy_reconstruction_recovers_unit_count(tmp_path):
    case = {"id": "legacy", "needle": "N", "filler": "abcd ", "question": "\nQ"}
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    tok = CharacterTokenizer()
    n_units, depth = 123, 0.47
    expected = case["filler"] * int(n_units * depth) + case["needle"] + " " + case["filler"] * (n_units - int(n_units * depth)) + case["question"]
    target = {
        "prompt_bank": "train", "case_id": "legacy", "needle": "N", "question": "\nQ",
        "depth": depth, "seq_len": len(tok(expected)["input_ids"]), "block_size": 4,
        "needle_block": len(tok(expected[:expected.index("N")])["input_ids"]) // 4,
    }
    actual, _ = reconstruct_teacher_prompt(tok, target, bank_path=bank)
    assert actual == expected


def test_calibrated_prompt_hits_budget_and_replays_from_filler_units(tmp_path):
    case = {
        "id": "calibrated", "needle": "NEEDLE", "filler": "abcd ",
        "question": "\nQuestion?",
    }
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    tok = CharacterTokenizer()
    prompt, _, filler_units = build_haystack_calibrated(
        tok, 1000, case["needle"], 0.5, case["filler"],
        suffix=case["question"])
    full = prompt + case["question"]
    actual_length = len(tok(full)["input_ids"])
    assert actual_length <= 1000
    assert 1000 - actual_length < len(case["filler"])

    target = {
        "prompt_bank": "heldout", "case_id": case["id"],
        "needle": case["needle"], "question": case["question"],
        "depth": 0.5, "seq_len": actual_length, "block_size": 4,
        "needle_block": len(tok(prompt[:prompt.index(case["needle"])])["input_ids"]) // 4,
        "filler_units": filler_units,
        "haystack_token_budget": 1000,
    }
    replayed, _ = reconstruct_teacher_prompt(tok, target, bank_path=bank)
    assert replayed == full
