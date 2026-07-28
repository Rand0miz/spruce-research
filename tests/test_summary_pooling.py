import math

import pytest
import torch

from interfaces.residual_summaries import (
    build_residual_summary_nodes,
    build_residual_tree_layout,
)
from sparse.attention import sparse_prefill_attention_forward
from sparse.summaries import (
    build_kv_summary_table,
    residual_attention_density,
    token_validity_from_attention_mask,
)


class _AttentionModule(torch.nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx


def _causal_mask(length):
    mask = torch.full((1, 1, length, length), torch.finfo(torch.float32).min)
    return torch.triu(mask, diagonal=1)


def _local_routes(qblocks):
    selected = torch.full((1, 1, 1, qblocks, 2), -1, dtype=torch.int32)
    for query_block in range(qblocks):
        start = max(0, query_block - 1)
        row = torch.arange(start, query_block + 1, dtype=torch.int32)
        selected[0, 0, 0, query_block, : len(row)] = row
    return selected


def _all_routes(qblocks):
    selected = torch.full(
        (1, 1, 1, qblocks, qblocks), -1, dtype=torch.int32
    )
    for query_block in range(qblocks):
        selected[0, 0, 0, query_block, : query_block + 1] = torch.arange(
            query_block + 1, dtype=torch.int32
        )
    return selected


def test_pooling_means_counts_and_tail_empty_prototype():
    key = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    value = (key * 10).clone()
    layout = build_residual_tree_layout(3)
    table = build_kv_summary_table(
        key, value, layout, block_size=2, prototypes=2
    )

    # Parent node 3 covers blocks [0, 2), hence tokens [0, 4).
    assert table.counts[0, 3].tolist() == [2.0, 2.0]
    assert table.keys[0, 0, 3, :, 0].tolist() == [0.5, 2.5]
    assert table.values[0, 0, 3, :, 0].tolist() == [5.0, 25.0]
    torch.testing.assert_close(
        table.log_counts[0, 3],
        torch.tensor([math.log(2.0), math.log(2.0)]),
    )

    # Tail leaf 2 contains only token 4. Proportional splitting leaves the
    # first prototype empty and places the token in the second.
    assert table.counts[0, 2].tolist() == [0.0, 1.0]
    assert torch.isneginf(table.log_counts[0, 2, 0])
    assert table.keys[0, 0, 2, 1, 0].item() == 4.0


def test_padding_tokens_do_not_contribute_to_means_or_counts():
    key = torch.tensor([[[[100.0], [100.0], [2.0], [4.0]]]])
    value = key.clone()
    layout = build_residual_tree_layout(2)
    token_mask = torch.tensor([[False, False, True, True]])
    table = build_kv_summary_table(
        key,
        value,
        layout,
        block_size=2,
        prototypes=1,
        token_mask=token_mask,
    )
    assert table.counts[0, layout.root_id, 0].item() == 2
    assert table.keys[0, 0, layout.root_id, 0, 0].item() == 3.0


def test_token_validity_handles_right_padded_final_query_row():
    floor = torch.finfo(torch.float32).min
    mask = torch.full((1, 1, 5, 5), floor)
    mask[0, 0, 0, 0] = 0
    mask[0, 0, 1, :2] = 0
    mask[0, 0, 2, :3] = 0
    assert token_validity_from_attention_mask(
        mask, batch=1, seq_len=5
    ).tolist() == [[True, True, True, False, False]]


@torch.no_grad()
@pytest.mark.parametrize("prototypes", [1, 2, 4])
def test_multiplicity_corrected_summary_matches_dense_for_identical_keys(prototypes):
    torch.manual_seed(4)
    length = 6
    query = torch.randn(1, 2, length, 3)
    key = torch.ones(1, 1, length, 3)
    value = torch.arange(length * 3, dtype=torch.float32).view(1, 1, length, 3)
    selected = _local_routes(qblocks=3)
    residual = build_residual_summary_nodes(selected)

    actual, _ = sparse_prefill_attention_forward(
        _AttentionModule(),
        query,
        key,
        value,
        _causal_mask(length),
        selected_blocks=selected,
        block_size=2,
        residual_summaries=True,
        residual_summary_nodes=residual,
        summary_prototypes=prototypes,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    dense_scores = query @ expanded_key.transpose(-1, -2) / math.sqrt(3)
    expected = (
        torch.softmax(dense_scores + _causal_mask(length), dim=-1)
        @ expanded_value
    )
    torch.testing.assert_close(actual, expected.transpose(1, 2), atol=1e-6, rtol=1e-6)


def test_summaries_disabled_is_behaviorally_identical():
    torch.manual_seed(8)
    query = torch.randn(1, 1, 6, 2)
    key = torch.randn(1, 1, 6, 2)
    value = torch.randn(1, 1, 6, 2)
    selected = _local_routes(qblocks=3)
    residual = build_residual_summary_nodes(selected)
    baseline, _ = sparse_prefill_attention_forward(
        _AttentionModule(),
        query,
        key,
        value,
        _causal_mask(6),
        selected_blocks=selected,
        block_size=2,
    )
    disabled, _ = sparse_prefill_attention_forward(
        _AttentionModule(),
        query,
        key,
        value,
        _causal_mask(6),
        selected_blocks=selected,
        block_size=2,
        residual_summaries=False,
        residual_summary_nodes=residual,
        summary_prototypes=4,
    )
    assert torch.equal(disabled, baseline)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prototypes", [1, 2, 4])
def test_summary_attention_supports_reference_dtypes(dtype, prototypes):
    torch.manual_seed(12)
    query = torch.randn(1, 2, 5, 4, dtype=dtype)
    key = torch.randn(1, 1, 5, 4, dtype=dtype)
    value = torch.randn(1, 1, 5, 4, dtype=dtype)
    selected = _local_routes(qblocks=3)
    residual = build_residual_summary_nodes(selected)
    output, _ = sparse_prefill_attention_forward(
        _AttentionModule(),
        query,
        key,
        value,
        _causal_mask(5),
        selected_blocks=selected,
        block_size=2,
        residual_summaries=True,
        residual_summary_nodes=residual,
        summary_prototypes=prototypes,
    )
    assert output.dtype == dtype
    assert output.shape == (1, 5, 2, 4)
    assert torch.isfinite(output).all()


def test_all_blocks_has_empty_frontier_and_retains_dense_equivalence():
    torch.manual_seed(9)
    query = torch.randn(1, 2, 5, 3)
    key = torch.randn(1, 1, 5, 3)
    value = torch.randn(1, 1, 5, 3)
    selected = _all_routes(qblocks=3)
    residual = build_residual_summary_nodes(selected)
    actual, _ = sparse_prefill_attention_forward(
        _AttentionModule(),
        query,
        key,
        value,
        _causal_mask(5),
        selected_blocks=selected,
        block_size=2,
        residual_summaries=True,
        residual_summary_nodes=residual,
        summary_prototypes=4,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = query @ expanded_key.transpose(-1, -2) / math.sqrt(3)
    expected = torch.softmax(scores + _causal_mask(5), dim=-1) @ expanded_value
    torch.testing.assert_close(actual, expected.transpose(1, 2))


def test_attention_accounting_charges_blocks_and_only_valid_prototypes():
    selected = _local_routes(qblocks=3)
    residual = build_residual_summary_nodes(selected)
    density = residual_attention_density(
        selected,
        residual,
        seq_len=5,
        block_size=2,
        prototypes=1,
    )
    assert density["exact_attention_entries"] == 10
    assert density["summary_prototype_entries"] == 1
    assert density["charged_attention_entries"] == 11
    assert density["maximum_causal_attention_entries"] == 12
    assert density["charged_attention_fraction"] == 11 / 12
