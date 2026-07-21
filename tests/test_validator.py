import pytest
import torch

from interfaces.validator import PAD_VALUE, validate_selected_blocks


def make_row(ids, k, pad=PAD_VALUE):
    return list(ids) + [pad] * (k - len(ids))


K = 4


def test_valid_selected_blocks_passes():
    # Valid: each row is causal, sorted, duplicate-free, and includes q-1/q local window.
    valid = torch.tensor([[[[
        make_row([0], K),
        make_row([0, 1], K),
        make_row([0, 1, 2], K),
        make_row([0, 2, 3], K),
    ]]]], dtype=torch.int32)
    validate_selected_blocks(valid)


def test_rejects_future_block():
    bad_future = torch.tensor([[[[make_row([0, 5], K)]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="future"):
        validate_selected_blocks(bad_future)


def test_rejects_duplicate_block():
    bad_dup = torch.tensor([[[[make_row([0], K), make_row([0, 1, 1], K)]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="Duplicate"):
        validate_selected_blocks(bad_dup)


def test_rejects_unsorted_blocks():
    bad_sort = torch.tensor(
        [[[[make_row([0], K), make_row([0, 1], K), make_row([0, 2, 1], K)]]]],
        dtype=torch.int32,
    )
    with pytest.raises(AssertionError, match="Not sorted"):
        validate_selected_blocks(bad_sort)


def test_rejects_missing_local_window_block():
    missing_local = torch.tensor([[[[
        make_row([0], K),
        make_row([1], K),
    ]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="Missing local-window"):
        validate_selected_blocks(missing_local)


def test_rejects_non_trailing_padding():
    bad_pad = torch.tensor([[[[
        make_row([0], K),
        [0, PAD_VALUE, 1, PAD_VALUE],
    ]]]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="trailing"):
        validate_selected_blocks(bad_pad)

