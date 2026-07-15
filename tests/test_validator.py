import pytest
import torch

from interfaces.validator import PAD_VALUE, validate_selected_blocks


def make_row(ids, k, pad=PAD_VALUE):
    return list(ids) + [pad] * (k - len(ids))


K = 4


def test_valid_selected_blocks_passes():
    # Valid: 4 query blocks, each causal. Query block 3 attends to 0, 2, 3.
    valid = torch.tensor([[[[
        make_row([0], K),           # q=0 -> {0}
        make_row([1], K),           # q=1 -> {1}
        make_row([0, 2], K),        # q=2 -> {0,2}
        make_row([0, 2, 3], K),     # q=3 -> {0,2,3}
    ]]]], dtype=torch.int32)
    validate_selected_blocks(valid)


def test_rejects_future_block():
    bad_future = torch.tensor([[[[make_row([0, 5], K)]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="future"):
        validate_selected_blocks(bad_future)


def test_rejects_duplicate_block():
    bad_dup = torch.tensor([[[[make_row([0], K), make_row([1, 1], K)]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="Duplicate"):
        validate_selected_blocks(bad_dup)


def test_rejects_unsorted_blocks():
    bad_sort = torch.tensor(
        [[[[make_row([0], K), make_row([1], K), make_row([2, 1], K)]]]],
        dtype=torch.int32,
    )
    with pytest.raises(AssertionError, match="Not sorted"):
        validate_selected_blocks(bad_sort)
