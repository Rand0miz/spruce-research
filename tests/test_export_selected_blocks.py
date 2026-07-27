import pytest
import torch

from interfaces.validator import validate_selected_blocks
from scripts.export_selected_blocks import (
    selected_ids_to_blocks,
    selected_mask_to_blocks,
)


def test_vectorized_route_packing_is_valid_and_prioritizes_local_blocks():
    selected = torch.zeros((2, 2, 4, 4), dtype=torch.bool)
    selected[..., 0, 0] = True
    selected[..., 1, 0] = True
    selected[..., 2, 0] = True
    selected[..., 3, 0] = True
    selected[..., 3, 1] = True

    packed = selected_mask_to_blocks(selected, k_selected=3, local_window=1)

    assert packed.shape == (1, 2, 2, 4, 3)
    assert packed[0, 0, 0].tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [0, 2, 3],
    ]
    validate_selected_blocks(packed)


def test_vectorized_route_packing_rejects_too_narrow_rows():
    selected = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="cannot fit local window"):
        selected_mask_to_blocks(selected, k_selected=1, local_window=1)


def test_sink_blocks_default_is_a_no_op():
    ids = torch.tensor([[[[0, -1], [1, 0], [0, 2], [1, 3]]]], dtype=torch.long)
    baseline = selected_ids_to_blocks(
        ids, k_selected=3, local_window=1, key_blocks=4)
    explicit_zero = selected_ids_to_blocks(
        ids, k_selected=3, local_window=1, key_blocks=4, sink_blocks=0)
    torch.testing.assert_close(baseline, explicit_zero)


def test_sink_block_forced_into_every_causal_row():
    # Traversal never proposes block 0 outside the local window here, so
    # without forcing, later rows drop the sink entirely.
    ids = torch.tensor(
        [[[[5, 5], [5, 5], [5, 5], [2, 5], [2, 3], [3, 4]]]], dtype=torch.long)
    without = selected_ids_to_blocks(
        ids, k_selected=4, local_window=1, key_blocks=6)
    assert 0 not in without[0, 0, 0, 5].tolist()

    with_sink = selected_ids_to_blocks(
        ids, k_selected=4, local_window=1, key_blocks=6, sink_blocks=1)
    for row in range(6):
        assert with_sink[0, 0, 0, row, 0].item() == 0
    validate_selected_blocks(with_sink)


def test_sink_block_never_duplicates_inside_the_local_window():
    # Rows 0 and 1 already cover block 0 via the local window; forcing the
    # sink must not emit it twice nor spend a second slot on it.
    ids = torch.full((1, 1, 4, 2), -1, dtype=torch.long)
    packed = selected_ids_to_blocks(
        ids, k_selected=3, local_window=1, key_blocks=4, sink_blocks=1)
    assert packed[0, 0, 0, 0].tolist() == [0, -1, -1]
    assert packed[0, 0, 0, 1].tolist() == [0, 1, -1]
    validate_selected_blocks(packed)


def test_sink_blocks_rejected_when_budget_cannot_hold_them():
    ids = torch.zeros((1, 1, 4, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="sink block"):
        selected_ids_to_blocks(
            ids, k_selected=2, local_window=1, key_blocks=4, sink_blocks=1)


def test_compact_id_packing_matches_full_mask_policy():
    ids = torch.tensor([[[[
        0, -1, -1,
    ], [
        1, 0, -1,
    ], [
        0, 2, 1,
    ], [
        1, 0, 3,
    ]]]], dtype=torch.long)
    with_pad = torch.zeros((1, 1, 4, 5), dtype=torch.bool)
    safe = ids.masked_fill(ids < 0, 4)
    with_pad.scatter_(-1, safe, True)
    mask = with_pad[..., :4]

    expected = selected_mask_to_blocks(
        mask, k_selected=3, local_window=1)
    actual = selected_ids_to_blocks(
        ids, k_selected=3, local_window=1, key_blocks=4)

    torch.testing.assert_close(actual, expected)
    validate_selected_blocks(actual)
