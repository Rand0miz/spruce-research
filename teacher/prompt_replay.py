"""Deterministically rebuild a teacher-target prompt from its saved metadata."""
import json
import os

import torch

from eval.haystack import build_haystack


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BANKS = {
    "train": os.path.join(ROOT, "scripts", "prompt_banks", "train.json"),
    "heldout": os.path.join(ROOT, "scripts", "prompt_banks", "heldout.json"),
}
LEGACY_REQUESTED_LENGTHS = (16384, 32768)


def _load_case(bank_path, case_id):
    with open(bank_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data["cases"] if isinstance(data, dict) else data
    for case in cases:
        if case.get("id") == case_id:
            return case
    raise KeyError(f"case_id {case_id!r} was not found in {bank_path}")


def _prompt_from_unit_count(case, depth, n_units):
    """Exact build_haystack text when the filler repetition count is known."""
    n_before = int(n_units * depth)
    return (
        case["filler"] * n_before
        + case["needle"]
        + " "
        + case["filler"] * (n_units - n_before)
        + case["question"]
    )


def _needle_block(tokenizer, prompt, needle, block_size):
    return len(tokenizer(prompt[:prompt.index(needle)])["input_ids"]) // block_size


def _matches_target(tokenizer, prompt, target):
    return (
        len(tokenizer(prompt)["input_ids"]) == int(target["seq_len"])
        and _needle_block(tokenizer, prompt, target["needle"], int(target["block_size"]))
        == int(target["needle_block"])
    )


def _legacy_unit_count(tokenizer, case, target):
    """Recover the original filler count for targets made before replay metadata.

    Existing targets were extracted at 16K or 32K.  Their final token length
    cannot be used as the construction budget because BPE merges across filler
    repetitions.  Try those original requested lengths first, then use a
    bounded monotonic search as a fallback for custom lengths.
    """
    question_tokens = len(tokenizer(case["question"])["input_ids"])
    needle_tokens = len(tokenizer(case["needle"])["input_ids"])
    unit_tokens = len(tokenizer(case["filler"])["input_ids"])
    depth = float(target["depth"])

    candidate_counts = []
    for requested_length in LEGACY_REQUESTED_LENGTHS:
        budget = requested_length - question_tokens
        candidate_counts.append(max(0, (budget - needle_tokens) // unit_tokens))

    # For a custom legacy extraction length, find the smallest repetition count
    # whose actual tokenization reaches the saved sequence length, then inspect
    # the nearby plateau.  BPE length is monotonic but can remain flat briefly.
    def length_at(n_units):
        return len(tokenizer(_prompt_from_unit_count(case, depth, n_units))["input_ids"])

    expected = int(target["seq_len"])
    low, high = 0, max(1, (expected // max(1, unit_tokens)) * 2)
    while length_at(high) < expected:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if length_at(mid) < expected:
            low = mid + 1
        else:
            high = mid
    candidate_counts.extend(range(max(0, low - 128), low + 129))

    for n_units in dict.fromkeys(candidate_counts):
        prompt = _prompt_from_unit_count(case, depth, n_units)
        if _matches_target(tokenizer, prompt, target):
            return n_units
    raise ValueError(
        "could not recover the original filler repetition count from legacy metadata")


def reconstruct_teacher_prompt(tokenizer, teacher_target, bank_path=None):
    """Return the exact extraction prompt from a teacher target path or dict.

    Extraction saved the case ID, needle, question, depth, and final token
    length.  The prompt bank supplies the corresponding filler.  Reusing
    ``build_haystack`` makes replay follow the original construction exactly.
    """
    target = (
        torch.load(teacher_target, map_location="cpu", weights_only=False)
        if isinstance(teacher_target, (str, os.PathLike)) else teacher_target
    )
    required = {"prompt_bank", "case_id", "needle", "question", "depth", "seq_len"}
    missing = required - set(target)
    if missing:
        raise KeyError(f"teacher target is missing reconstruction metadata: {sorted(missing)}")
    # The legacy fallback probes a few longer candidate strings while recovering
    # a filler count.  This is tokenizer-only work, not a model forward; avoid
    # its misleading max-length warning during that bounded search.
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = max(int(tokenizer.model_max_length), int(target["seq_len"]) * 2)

    if "prompt_text" in target:
        full = target["prompt_text"]
        if not isinstance(full, str) or not full:
            raise ValueError("teacher target prompt_text must be a non-empty string")
        if target["needle"] not in full:
            raise ValueError("teacher target needle is absent from prompt_text")
        if target.get("prompt_format") == "qwen_chat_v1":
            user_prompt = target.get("user_prompt_text")
            if (not isinstance(user_prompt, str)
                    or not user_prompt.endswith(target["question"])
                    or user_prompt not in full):
                raise ValueError(
                    "teacher target chat prompt does not preserve its exact "
                    "user prompt and final question")
        elif not full.endswith(target["question"]):
            raise ValueError(
                "teacher target prompt_text does not end with its question")
        if not _matches_target(tokenizer, full, target):
            actual_length = len(tokenizer(full)["input_ids"])
            raise ValueError(
                f"saved prompt_text has {actual_length} tokens or a different "
                "needle block; use the extraction tokenizer")
        return full, target

    bank_name = target["prompt_bank"]
    if bank_path is None:
        try:
            bank_path = DEFAULT_BANKS[bank_name]
        except KeyError as error:
            raise ValueError(f"unknown prompt_bank {bank_name!r}; pass --prompt-bank") from error
    case = _load_case(bank_path, target["case_id"])
    for key in ("needle", "question"):
        if case[key] != target[key]:
            raise ValueError(
                f"teacher target {key} does not match prompt-bank case {target['case_id']!r}")

    question_tokens = len(tokenizer(case["question"])["input_ids"])
    if "filler_units" in target:
        full = _prompt_from_unit_count(
            case, float(target["depth"]), int(target["filler_units"]))
    elif "haystack_token_budget" in target:
        prompt, _ = build_haystack(
            tokenizer, int(target["haystack_token_budget"]), case["needle"],
            float(target["depth"]), case["filler"],
        )
        full = prompt + case["question"]
    else:
        n_units = _legacy_unit_count(tokenizer, case, target)
        full = _prompt_from_unit_count(case, float(target["depth"]), n_units)

    if not _matches_target(tokenizer, full, target):
        actual_length = len(tokenizer(full)["input_ids"])
        raise ValueError(
            f"reconstructed prompt has {actual_length} tokens or a different needle block; "
            "use the same model tokenizer used during extraction.")
    return full, target
