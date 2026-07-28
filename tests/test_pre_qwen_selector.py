import pytest
import torch

from interfaces.evidence_compiler import locate_prompt_layout
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    retrieve_pre_qwen_blocks,
    select_pre_qwen_blocks,
)


class WhitespaceTokenizer:
    """Offset-preserving tokenizer with stable word IDs for selector tests."""

    def __init__(self):
        self.vocabulary = {}

    def _tokens(self, text):
        import re
        return list(re.finditer(r"\w+|[^\w\s]", text.lower()))

    def __call__(
            self, text, return_offsets_mapping=False,
            add_special_tokens=False, **_kwargs):
        del add_special_tokens
        matches = self._tokens(text)
        ids = []
        for match in matches:
            token = match.group(0)
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
            ids.append(self.vocabulary[token])
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result


def _fixture():
    tokenizer = WhitespaceTokenizer()
    paragraphs = [
        "Routine harbor schedules list cargo inspections and ordinary repairs.",
        "The geology committee designated Lake Orison as the primary groundwater source.",
        "Unrelated farm reports discuss Lake Merrow and seasonal irrigation.",
        "Closing records describe staffing, budgets, and archive maintenance.",
    ]
    document = "\n\n".join(paragraphs)
    question = (
        "\n\nWhich lake did the geology committee designate as the primary "
        "groundwater source?"
    )
    user = document + question
    full = f"system wrapper user {user} assistant"
    layout = locate_prompt_layout(tokenizer, full, user, question)
    return tokenizer, full, question, layout


def test_odd_tree_has_stable_leaf_first_ids_and_complete_ranges():
    tokenizer, _full, _question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=5, feature_dim=64, radix=2)
    counts = [level.features.shape[0] for level in index.levels]
    assert counts[-1] == 1
    assert all(
        counts[i + 1] == (counts[i] + 1) // 2
        for i in range(len(counts) - 1))
    offsets = [level.node_id_offset for level in index.levels]
    assert offsets == [
        sum(counts[:index]) for index in range(len(counts))
    ]
    assert index.levels[-1].starts.tolist() == [0]
    assert index.levels[-1].ends.tolist() == [index.leaf_count]
    assert tokenizer is not None


def test_parent_union_never_drops_a_leaf_feature():
    _tokenizer, _full, _question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=4, feature_dim=64)
    for level_index in range(1, len(index.levels)):
        children = index.levels[level_index - 1].features
        parents = index.levels[level_index].features
        for child_index in range(children.shape[0]):
            parent_index = child_index // 2
            assert torch.all(
                parents[parent_index] | ~children[child_index])


def test_query_retrieves_the_evidence_block_without_qwen_features():
    tokenizer, full, question, layout = _fixture()
    index, selection = retrieve_pre_qwen_blocks(
        tokenizer, layout, question, block_size=8,
        top_m=2, beam=4, feature_dim=128)
    evidence_token = next(
        index for index, (start, end) in enumerate(layout.offsets)
        if start <= full.index("Lake Orison") < end)
    evidence_block = evidence_token // 8
    assert evidence_block in selection.blocks
    assert selection.final_candidates <= 4
    assert selection.levels_descended == len(index.levels) - 1
    assert selection.visited_nodes <= index.node_count


def test_selection_validation_rejects_bad_shapes_and_budgets():
    tokenizer, _full, question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=8, feature_dim=64)
    weights = question_feature_weights(tokenizer, question, index)
    with pytest.raises(ValueError, match="beam"):
        select_pre_qwen_blocks(index, weights, top_m=4, beam=2)
    with pytest.raises(ValueError, match="query_weights"):
        select_pre_qwen_blocks(index, weights[:-1], top_m=1, beam=1)
    bad = weights.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        select_pre_qwen_blocks(index, bad, top_m=1, beam=1)


@pytest.mark.integration
def test_qwen_tokenizer_live_selector_finds_exact_natural_evidence():
    from transformers import AutoTokenizer

    from eval.natural_context import format_instruct_chat_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", local_files_only=True)
    evidence = (
        "The hydrology panel identified Cedar Spring as the preserve's "
        "dependable dry-season water source."
    )
    filler = "\n\n".join(
        f"Archive record {index} discusses routine maintenance scheduling."
        for index in range(300)
    )
    question = (
        "\n\nQuestion: What did the hydrology panel identify as the "
        "dependable dry-season water source?"
    )
    user = filler + "\n\n" + evidence + "\n\n" + filler + question
    full = format_instruct_chat_prompt(tokenizer, user)
    layout = locate_prompt_layout(tokenizer, full, user, question)
    _index, selection = retrieve_pre_qwen_blocks(
        tokenizer, layout, question, 64,
        top_m=4, beam=8, feature_dim=1024)
    evidence_token = next(
        index for index, (start, end) in enumerate(layout.offsets)
        if start <= full.index("Cedar Spring") < end)
    assert evidence_token // 64 in selection.blocks
