import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces.validator import validate_selected_blocks, PAD_VALUE

def make_row(ids, k, pad=PAD_VALUE):
    row = list(ids) + [pad] * (k - len(ids))
    return row

K = 4

# Valid: 4 query blocks, each causal. Query block 3 attends to 0, 2, 3.
valid = torch.tensor([[[[
    make_row([0], K),           # q=0 -> {0}
    make_row([1], K),           # q=1 -> {1}
    make_row([0, 2], K),        # q=2 -> {0,2}
    make_row([0, 2, 3], K),     # q=3 -> {0,2,3}
]]]], dtype=torch.int32)
validate_selected_blocks(valid)
print("Valid case passed.")

# Invalid: future block. q=0 selects block 5 (5 > 0).
try:
    bad_future = torch.tensor([[[[make_row([0, 5], K)]]]], dtype=torch.int32)
    validate_selected_blocks(bad_future)
    print("ERROR: should have raised on future block")
except AssertionError as e:
    print("correctly rejected future block:", e)

# Invalid: duplicate ID. q=1 selects [1, 1] -- causal-valid but repeated.
try:
    bad_dup = torch.tensor([[[[make_row([0], K), make_row([1, 1], K)]]]], dtype=torch.int32)
    validate_selected_blocks(bad_dup)
    print("ERROR: should have raised on duplicate block")
except AssertionError as e:
    print("correctly rejected duplicate block:", e)

# Invalid: not sorted. q=2 selects [2, 1] -- causal-valid but out of order.
try:
    bad_sort = torch.tensor(
        [[[[make_row([0], K), make_row([1], K), make_row([2, 1], K)]]]],
        dtype=torch.int32,
    )
    validate_selected_blocks(bad_sort)
    print("ERROR: should have raised on not sorted")
except AssertionError as e:
    print("correctly rejected not sorted:", e)