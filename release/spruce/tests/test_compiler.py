import pytest

from spruce_attn.compiler import (
    compile_evidence_packet,
    compile_evidence_spans,
    document_block_ids,
    expand_block_ids,
    locate_prompt_layout,
    normalize_block_ids,
    render_evidence_content,
)


class CharacterTokenizer:
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
    assert document_block_ids(layout, block_size=16)


def test_paragraph_boundary_preserves_exact_evidence():
    tokenizer, full, user, question, _document = _fixture()
    layout = locate_prompt_layout(tokenizer, full, user, question)
    selected = [full.index("CEDAR-441") // 16]
    spans, selected_ids, expanded = compile_evidence_spans(
        full, layout, selected, block_size=16,
        block_radius=0, boundary="paragraph")
    assert selected_ids == tuple(selected)
    assert expanded == tuple(selected)
    assert spans[0].text == (
        "The review committee compared three proposals. The approved code "
        "was CEDAR-441, recorded after the final vote."
    )
    assert question.strip() not in spans[0].text


def test_end_to_end_packet_contains_provenance():
    tokenizer, full, user, question, _document = _fixture()
    selected = [full.index("CEDAR-441") // 16]
    packet = compile_evidence_packet(
        tokenizer, full, user, question, selected, block_size=16,
        boundary="paragraph")
    assert "CEDAR-441" in packet.prompt
    assert "[Evidence 1 | source blocks" in packet.content
    assert "[Final question]" in packet.content
    assert packet.prompt.endswith("<assistant>")
    assert packet.metadata()["selected_blocks"] == selected


def test_renderer_never_drops_a_span():
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
