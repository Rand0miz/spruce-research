import json

from benchmarks.run_natural_context_suite import (
    DEFAULT_DEPTHS,
    DEFAULT_LENGTHS,
)
from eval.natural_context import (
    build_natural_prompt_calibrated,
    format_instruct_chat_prompt,
    natural_sentence,
)
from eval.score import score_concise_retrieval, score_retrieval
from scripts.extract_teacher_targets import load_prompt_bank
from teacher.prompt_replay import reconstruct_teacher_prompt


class CharacterTokenizer:
    model_max_length = 1_000_000

    def __call__(self, text):
        return {"input_ids": list(text)}


class ChatCharacterTokenizer(CharacterTokenizer):
    def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user><assistant>"
        )


def _case():
    return {
        "id": "natural_case",
        "builder": "natural_prose_v1",
        "evidence": (
            "The committee selected Lake Orison as the primary water source. "
            "A later audit confirmed the designation."
        ),
        "needle": (
            "The committee selected Lake Orison as the primary water source."
        ),
        "question": "\n\nWhich lake was selected?",
        "answers": ["Lake Orison"],
        "distractors": [
            "Lake Merrow supplies the western farms but was not selected.",
            "Lake Vesta is monitored only for recreation.",
        ],
    }


def test_natural_sentences_do_not_repeat_across_long_context():
    sentences = [
        natural_sentence("case", index, seed=7)
        for index in range(2_500)
    ]
    assert len(set(sentences)) == len(sentences)


def test_natural_prompt_is_calibrated_and_places_evidence():
    tokenizer = CharacterTokenizer()
    prompt, needle, units = build_natural_prompt_calibrated(
        tokenizer, 20_000, _case(), depth=0.7, seed=3)

    assert len(tokenizer(prompt)["input_ids"]) <= 20_000
    assert 20_000 - len(tokenizer(prompt)["input_ids"]) <= 64
    assert needle in prompt
    assert prompt.endswith(_case()["question"])
    assert units > 0
    evidence_fraction = prompt.index(needle) / len(prompt)
    assert 0.6 < evidence_fraction < 0.8


def test_chat_formatted_prompt_is_calibrated_with_generation_marker():
    tokenizer = ChatCharacterTokenizer()
    prompt, needle, units, user_prompt = build_natural_prompt_calibrated(
        tokenizer, 20_000, _case(), depth=0.7, seed=3,
        prompt_formatter=lambda content: format_instruct_chat_prompt(
            tokenizer, content),
        return_content=True,
    )

    assert len(tokenizer(prompt)["input_ids"]) <= 20_000
    assert 20_000 - len(tokenizer(prompt)["input_ids"]) <= 64
    assert prompt.endswith("</user><assistant>")
    assert user_prompt.endswith(_case()["question"])
    assert user_prompt in prompt
    assert needle in prompt
    assert units > 0


def test_natural_prompt_bank_validation_and_prompt_text_replay(tmp_path):
    case = _case()
    bank = tmp_path / "natural.json"
    bank.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    assert load_prompt_bank(bank)[0]["answers"] == ["Lake Orison"]

    tokenizer = CharacterTokenizer()
    prompt, needle, _ = build_natural_prompt_calibrated(
        tokenizer, 10_000, case, depth=0.3, seed=2)
    target = {
        "prompt_bank": "natural_heldout",
        "case_id": case["id"],
        "needle": needle,
        "question": case["question"],
        "depth": 0.3,
        "seq_len": len(prompt),
        "block_size": 64,
        "needle_block": prompt.index(needle) // 64,
        "prompt_text": prompt,
        "reference_answers": case["answers"],
    }
    replayed, _ = reconstruct_teacher_prompt(tokenizer, target)
    assert replayed == prompt


def test_short_answer_alias_scoring_avoids_sentence_copy_requirement():
    score = score_retrieval(
        "The answer is April 17, 1986.",
        "The gallery reopened to the public on 17 April 1986.",
        reference_answers=["17 April 1986", "April 17, 1986"],
    )
    assert score["exact"]
    assert score["fuzzy"] == 1.0


def test_concise_scoring_rejects_document_continuation():
    assert score_concise_retrieval(
        "Cedar Spring.", ["Cedar Spring"])["exact"]
    assert not score_concise_retrieval(
        "Cedar Spring. The report then continued.", ["Cedar Spring"])["exact"]


def test_natural_suite_default_size():
    assert DEFAULT_LENGTHS == (64_000, 80_000, 96_000, 112_000, 128_000)
    assert DEFAULT_DEPTHS == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert len(DEFAULT_LENGTHS) * len(DEFAULT_DEPTHS) * 6 == 150
