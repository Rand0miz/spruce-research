import torch

from interfaces.validator import validate_selected_blocks
from scripts.route_overrides import force_needle_routes, teacher_topk_routes


def _toy_selected():
    # B=1, L=1, G=1, qb=6, K=3; block ids sorted, PAD=-1, causal, local diagonal included
    sel = torch.full((1, 1, 1, 6, 3), -1, dtype=torch.int32)
    rows = [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [1, 2, 3],
        [0, 3, 4],
        [2, 4, 5],
    ]
    for q, row in enumerate(rows):
        sel[0, 0, 0, q] = torch.tensor(row, dtype=torch.int32)
    return sel


def test_force_needle_present_in_all_causal_rows():
    out = force_needle_routes(_toy_selected(), needle_block=2)
    assert out.shape == (1, 1, 1, 6, 4)
    for q in range(2, 6):
        assert (out[0, 0, 0, q] == 2).any(), f"row {q} missing needle"


def test_force_needle_noncausal_rows_unchanged():
    sel = _toy_selected()
    out = force_needle_routes(sel, needle_block=2)
    for q in range(2):  # needle_block=2 not causally visible to query blocks 0,1
        old = set(sel[0, 0, 0, q].tolist()) - {-1}
        new = set(out[0, 0, 0, q].tolist()) - {-1}
        assert old == new


def test_force_needle_validator_and_no_duplicates():
    out = force_needle_routes(_toy_selected(), needle_block=2)
    validate_selected_blocks(out)
    for q in range(6):
        row = [b for b in out[0, 0, 0, q].tolist() if b >= 0]
        assert len(row) == len(set(row))
        assert row == sorted(row)


def test_force_needle_rows_already_containing_needle_unchanged_setwise():
    sel = _toy_selected()
    out = force_needle_routes(sel, needle_block=2)
    q = 2  # row [0,1,2] already contains 2
    assert set(sel[0, 0, 0, q].tolist()) - {-1} == set(out[0, 0, 0, q].tolist()) - {-1}


def test_teacher_topk_routes_selects_teacher_mass():
    L, G, qb, kb = 2, 1, 6, 6
    target = torch.zeros(L, G, qb, kb)
    # reader row: mass on blocks 0 and 3
    target[:, :, 5, 0] = 0.6
    target[:, :, 5, 3] = 0.4
    for q in range(qb - 1):
        target[:, :, q, 0] = 1.0
    out = teacher_topk_routes(target, k_selected=4, local_window=1)
    assert out.shape == (1, L, G, qb, 4)
    validate_selected_blocks(out, local_window=1)
    # K=4, local_window=1 -> 2 forced local blocks {4,5} + 2 top-mass {0,3}
    reader = set(out[0, 0, 0, 5].tolist())
    assert {0, 3, 4, 5} <= reader

    # Tight budget: K=3 leaves one nonlocal slot -> only the TOP-mass block
    # survives; the packer must not swap it for a smaller-id lower-mass block.
    tight = teacher_topk_routes(target, k_selected=3, local_window=1)
    tight_reader = set(tight[0, 0, 0, 5].tolist())
    assert 0 in tight_reader and 3 not in tight_reader


def test_dense_reader_routes_full_causal_reader_row():
    from scripts.route_overrides import dense_reader_routes
    sel = _toy_selected()
    out = dense_reader_routes(sel)
    assert out.shape[-1] == 6                      # widened to kb
    assert out[0, 0, 0, 5].tolist() == [0, 1, 2, 3, 4, 5]
    for q in range(5):                             # other rows keep their sets
        assert set(out[0, 0, 0, q].tolist()) - {-1} == set(sel[0, 0, 0, q].tolist()) - {-1}
    validate_selected_blocks(out)
