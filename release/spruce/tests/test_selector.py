import pytest
import torch

from spruce_attn.compiler import locate_prompt_layout
from spruce_attn.selector import (
    build_pre_qwen_index,
    question_feature_weights,
    retrieve_pre_qwen_blocks,
    select_pre_qwen_blocks,
)

from tests.helpers import WhitespaceTokenizer


def _fixture():
    tokenizer = WhitespaceTokenizer()
    paragraphs = [
        "Routine harbor schedules list cargo inspections and ordinary repairs.",
        "The geology committee designated Lake Orison as the primary groundwater source.",
        "Unrelated farm reports discuss Lake Merrow and seasonal irrigation.",
        "Closing records describe staffing budgets and archive maintenance.",
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


def test_odd_tree_has_stable_leaf_first_ids():
    _tokenizer, _full, _question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=5, feature_dim=64, radix=2)
    counts = [level.features.shape[0] for level in index.levels]
    assert counts[-1] == 1
    assert all(
        counts[index + 1] == (counts[index] + 1) // 2
        for index in range(len(counts) - 1))
    assert [level.node_id_offset for level in index.levels] == [
        sum(counts[:index]) for index in range(len(counts))
    ]


def test_parent_union_never_drops_leaf_features():
    _tokenizer, _full, _question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=4, feature_dim=64)
    for level_index in range(1, len(index.levels)):
        children = index.levels[level_index - 1].features
        parents = index.levels[level_index].features
        for child_index in range(children.shape[0]):
            assert torch.all(
                parents[child_index // 2] | ~children[child_index])


def test_query_retrieves_evidence_without_model_features():
    tokenizer, full, question, layout = _fixture()
    index, selection = retrieve_pre_qwen_blocks(
        tokenizer, layout, question, block_size=8,
        top_m=2, beam=4, feature_dim=128)
    evidence_token = next(
        index for index, (start, end) in enumerate(layout.offsets)
        if start <= full.index("Lake Orison") < end)
    assert evidence_token // 8 in selection.blocks
    assert selection.final_candidates <= 4
    assert selection.visited_nodes <= index.node_count


def test_selection_rejects_bad_budgets_and_shapes():
    tokenizer, _full, question, layout = _fixture()
    index = build_pre_qwen_index(
        layout, block_size=8, feature_dim=64)
    weights = question_feature_weights(tokenizer, question, index)
    with pytest.raises(ValueError, match="beam"):
        select_pre_qwen_blocks(index, weights, top_m=4, beam=2)
    with pytest.raises(ValueError, match="query_weights"):
        select_pre_qwen_blocks(index, weights[:-1], top_m=1, beam=1)
