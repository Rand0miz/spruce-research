import re

from interfaces.evidence_compiler import locate_prompt_layout
from benchmarks.ruler_cached_factorial import (
    arm_name,
    run_case,
    tokenize_layouts,
)


class WhitespaceTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def __call__(
            self, text, return_offsets_mapping=False,
            add_special_tokens=False, **_kwargs):
        del add_special_tokens
        matches = list(re.finditer(r"\w+|[^\w\s]", text.lower()))
        ids = []
        for match in matches:
            token = match.group(0)
            self.vocabulary.setdefault(token, len(self.vocabulary) + 1)
            ids.append(self.vocabulary[token])
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches]
        return result

    def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def sample():
    paragraphs = [
        f"Archive paragraph {index} describes routine harbor record {index}."
        for index in range(30)
    ]
    paragraphs[9] = (
        "The geology committee designated Lake Orison as the groundwater "
        "source."
    )
    question = (
        "Question: Which lake did the geology committee designate as the "
        "groundwater source?\n"
    )
    prompt = "\n\n".join(paragraphs) + "\n" + question
    return {"input": prompt, "outputs": ["Lake Orison"], "index": 7}


def test_single_tokenization_layouts_match_production_for_every_tail():
    tokenizer = WhitespaceTokenizer()
    row = sample()
    questions, layouts, _mapper = tokenize_layouts(
        tokenizer, row["input"], tails=[2, 3])
    for tail in (2, 3):
        expected = locate_prompt_layout(
            tokenizer, row["input"], row["input"], questions[tail])
        assert layouts[tail] == expected


def test_cached_factorial_builds_each_index_once_and_reuses_rank_prefix():
    tokenizer = WhitespaceTokenizer()
    prefix_validation = set()
    result = run_case(
        sample(),
        "niah_single_1",
        tokenizer,
        tails=[2, 3],
        feature_dims=[64, 128],
        budgets=[4, 9],
        block_size=8,
        block_radius=1,
        boundary="paragraph",
        beam=16,
        unigram_fraction=0.5,
        idf_power=2.0,
        radix=2,
        max_occurrences=256,
        prefix_validation=prefix_validation,
    )

    assert len(result["cache"]) == 4
    assert len(result["arms"]) == 8
    assert prefix_validation == {(2, 64), (2, 128), (3, 64), (3, 128)}
    for tail in (2, 3):
        for dim in (64, 128):
            low = result["arms"][arm_name(tail, dim, 4)]
            high = result["arms"][arm_name(tail, dim, 9)]
            assert low["selected_blocks"] == high["selected_blocks"][:4]
            assert len(high["selected_blocks"]) == 9
