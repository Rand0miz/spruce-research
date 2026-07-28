import math

import pytest
import torch

from interfaces.residual_summaries import (
    build_residual_summary_nodes,
    build_residual_tree_layout,
    residual_frontier_for_row,
    validate_residual_summary_nodes,
)


def _routes(num_blocks=5):
    selected = torch.full(
        (1, 1, 1, num_blocks, 2), -1, dtype=torch.int32
    )
    for query_block in range(num_blocks):
        start = max(0, query_block - 1)
        values = torch.arange(start, query_block + 1, dtype=torch.int32)
        selected[0, 0, 0, query_block, : len(values)] = values
    return selected


def test_leaf_first_layout_is_stable_for_odd_leaf_count():
    layout = build_residual_tree_layout(5)
    assert layout.level_counts == (5, 3, 2, 1)
    assert layout.level_offsets == (0, 5, 8, 10)
    assert layout.num_nodes == 11
    assert layout.range(5) == (0, 2)
    assert layout.range(7) == (4, 5)
    assert layout.range(9) == (4, 5)
    assert layout.range(layout.root_id) == (0, 5)
    assert layout.children(layout.root_id) == (8, 9)


@pytest.mark.parametrize("num_blocks", [256, 512, 1024, 2048])
def test_layout_covers_16k_through_128k_block_counts(num_blocks):
    layout = build_residual_tree_layout(num_blocks)
    assert layout.range(layout.root_id) == (0, num_blocks)
    assert layout.num_nodes == 2 * num_blocks - 1


def test_frontier_is_maximal_and_complete_for_odd_tail():
    layout = build_residual_tree_layout(5)
    # Query block 4 keeps its local exact pair [3, 4]. The omitted prefix
    # [0, 3) is represented by parent [0, 2) plus leaf [2, 3).
    assert residual_frontier_for_row([3, 4], 4, layout) == [2, 5]

    residual = build_residual_summary_nodes(_routes())
    assert residual.shape == (1, 1, 1, 5, 2)
    assert residual[0, 0, 0].tolist() == [
        [-1, -1],
        [-1, -1],
        [0, -1],
        [5, -1],
        [2, 5],
    ]
    validate_residual_summary_nodes(_routes(), residual)


def test_partition_validator_rejects_overlap_and_future_nodes():
    selected = _routes()
    residual = build_residual_summary_nodes(selected)
    overlap = residual.clone()
    overlap[0, 0, 0, 4] = torch.tensor([2, 2], dtype=torch.int32)
    with pytest.raises(AssertionError, match="Duplicate residual"):
        validate_residual_summary_nodes(selected, overlap)

    future = residual.clone()
    future[0, 0, 0, 3] = torch.tensor([7, -1], dtype=torch.int32)
    with pytest.raises(AssertionError, match="future"):
        validate_residual_summary_nodes(selected, future)


def test_all_exact_route_has_zero_width_residual_frontier():
    query_blocks = 4
    keys = torch.arange(query_blocks, dtype=torch.int32)
    selected = torch.full(
        (1, 1, 1, query_blocks, query_blocks), -1, dtype=torch.int32
    )
    for query_block in range(query_blocks):
        selected[0, 0, 0, query_block, : query_block + 1] = keys[: query_block + 1]
    residual = build_residual_summary_nodes(selected)
    assert residual.shape == (1, 1, 1, query_blocks, 0)
    validate_residual_summary_nodes(selected, residual)

