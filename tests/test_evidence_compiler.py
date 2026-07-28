import math

import pytest
import torch

from interfaces.evidence_compiler import (
    compile_evidence_packet,
    compile_evidence_spans,
    document_block_ids,
    expand_block_ids,
    locate_prompt_layout,
    normalize_block_ids,
    render_evidence_content,
)
from selector.evidence import rank_reader_candidate_blocks


class CharacterTokenizer:
    """Tiny offset-preserving tokenizer for compiler unit tests."""

    def __call__(self, text, return_offsets_mapping=False, **_kwargs):
        encoded = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return encoded

    def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def _fixture():
    question = "\n\nWhich code was approved?"
    document = (
        "Opening context that is not relevant.\n\n"
        "The review committee compared three proposals. The approved code "
        "was CEDAR-441, recorded after the final vote.\n\n"
        "Closing context that is also irrelevant."
    )
    user = document + question
    full = f"<system>instructions</system><user>{user}</user><assistant>"
    return CharacterTokenizer(), full, user, question, document


def test_block_ids_are_stable_unique_and_checked():
    assert normalize_block_ids([4, 2, 4, 3], 6) == (2, 3, 4)
    assert expand_block_ids([0, 5], 6, radius=2) == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="outside"):
        normalize_block_ids([-1], 6)
    with pytest.raises(ValueError, match="radius"):
        expand_block_ids([1], 6, radius=-1)


def test_layout_clips_chat_wrapper_and_final_question():
    tokenizer, full, user, question, document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    assert full[layout.document_char_start:layout.document_char_end] == document
    assert layout.document_token_start == layout.document_char_start
    assert layout.document_token_end == layout.document_char_end
    blocks = document_block_ids(layout, block_size=16)
    assert blocks[0] == layout.document_token_start // 16
    assert blocks[-1] == (layout.document_token_end - 1) // 16


def test_paragraph_boundary_repairs_fragment_and_preserves_exact_evidence():
    tokenizer, full, user, question, _document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    needle_position = full.index("CEDAR-441")
    selected = [needle_position // 16]
    spans, selected_ids, expanded = compile_evidence_spans(
        full, layout, selected, block_size=16,
        block_radius=0, boundary="paragraph")

    assert selected_ids == tuple(selected)
    assert expanded == tuple(selected)
    assert len(spans) == 1
    assert spans[0].text == (
        "The review committee compared three proposals. The approved code "
        "was CEDAR-441, recorded after the final vote."
    )
    assert "<system>" not in spans[0].text
    assert question.strip() not in spans[0].text


def test_separated_blocks_remain_ordered_and_adjacent_blocks_merge():
    tokenizer, full, user, question, _document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    separated_block_size = 4
    first = full.index("Opening") // separated_block_size
    middle = full.index("CEDAR-441") // separated_block_size
    closing = full.index("Closing") // separated_block_size

    spans, _, _ = compile_evidence_spans(
        full, layout, [closing, first, middle],
        block_size=separated_block_size,
        boundary="paragraph")
    assert [span.text.split()[0] for span in spans] == [
        "Opening", "The", "Closing",
    ]

    merge_block_size = 12
    merge_middle = full.index("CEDAR-441") // merge_block_size
    merged, _, expanded = compile_evidence_spans(
        full, layout, [merge_middle], block_size=merge_block_size, block_radius=1,
        boundary="block")
    assert len(expanded) == 3
    assert len(merged) == 1


def test_end_to_end_packet_is_dense_chat_prompt_with_provenance():
    tokenizer, full, user, question, _document = _fixture()
    selected = [full.index("CEDAR-441") // 16]
    packet = compile_evidence_packet(
        tokenizer, full, user, question, selected, block_size=16,
        boundary="paragraph")

    assert "CEDAR-441" in packet.prompt
    assert "[Evidence 1 | source blocks" in packet.content
    assert "[Final question]" in packet.content
    assert packet.prompt.endswith("<assistant>")
    assert packet.compiled_prompt_tokens == len(packet.prompt)
    assert packet.evidence_source_tokens == len(packet.spans[0].text)
    assert packet.compression_fraction > 0
    metadata = packet.metadata()
    assert metadata["selected_blocks"] == selected
    assert metadata["span_count"] == 1


def test_renderer_never_silently_drops_a_span():
    tokenizer, full, user, question, _document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    selected = [
        full.index("Opening") // 12,
        full.index("CEDAR-441") // 12,
        full.index("Closing") // 12,
    ]
    spans, _, _ = compile_evidence_spans(
        full, layout, selected, block_size=12, boundary="paragraph")
    content = render_evidence_content(question, spans)
    for index, span in enumerate(spans, start=1):
        assert f"[Evidence {index} |" in content
        assert span.text.strip() in content


def test_selected_wrapper_only_is_rejected_instead_of_fabricating_evidence():
    tokenizer, full, user, question, _document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    with pytest.raises(ValueError, match="do not overlap"):
        compile_evidence_spans(
            full, layout, [0], block_size=8, boundary="block")


def test_reader_candidate_ranking_filters_to_document_and_max_pools():
    scores = torch.zeros(2, 2, 1, 8)
    scores[:, :, :, 0] = 100.0       # wrapper block; must be filtered
    scores[0, 0, 0, 3] = 2.0
    scores[1, 1, 0, 5] = 4.0
    scores[0, 1, 0, 6] = 3.0
    blocks, values = rank_reader_candidate_blocks(
        scores, top_m=2, allowed_blocks=[2, 3, 4, 5, 6])
    assert blocks == [5, 6]
    assert values == [4.0, 3.0]

    subset, subset_values = rank_reader_candidate_blocks(
        scores[..., [3, 5, 6]], top_m=2,
        allowed_blocks=[2, 3, 4, 5, 6], block_ids=[3, 5, 6])
    assert subset == [5, 6]
    assert subset_values == [4.0, 3.0]


def test_reader_candidate_ranking_rejects_bad_budget_and_nonfinite():
    scores = torch.zeros(1, 1, 1, 4)
    with pytest.raises(ValueError, match="top_m"):
        rank_reader_candidate_blocks(scores, 0, [0])
    scores[..., 1] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        rank_reader_candidate_blocks(scores, 1, [0, 1])


@pytest.mark.integration
def test_qwen_tokenizer_compiles_known_natural_evidence_without_rewriting():
    from transformers import AutoTokenizer

    from eval.natural_context import format_instruct_chat_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    question = "\n\nQuestion: What was the approved code?"
    evidence = (
        "The committee reviewed the final register. The approved code was "
        "CEDAR-441."
    )
    opening = "\n\n".join(
        f"Unrelated opening record {index} describes routine maintenance."
        for index in range(40)
    )
    closing = "\n\n".join(
        f"Unrelated closing record {index} describes routine scheduling."
        for index in range(40)
    )
    user = (
        opening + "\n\n"
        + evidence
        + "\n\n" + closing
        + question
    )
    full = format_instruct_chat_prompt(tokenizer, user)
    encoded = tokenizer(full)
    offsets = tokenizer(full, return_offsets_mapping=True)["offset_mapping"]
    evidence_char = full.index("CEDAR-441")
    evidence_token = next(
        index for index, (start, end) in enumerate(offsets)
        if start <= evidence_char < end)
    block_size = 16
    packet = compile_evidence_packet(
        tokenizer, full, user, question,
        [evidence_token // block_size], block_size,
        boundary="paragraph")
    assert evidence in packet.content
    assert packet.compiled_prompt_tokens < len(encoded["input_ids"])
