import pytest
import torch

from interfaces.validator import validate_selected_blocks
from scripts.route_overrides import (
    charged_route_density,
    force_needle_routes,
    teacher_topk_routes,
)


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


def test_dense_evidence_routes_ceiling_control():
    from scripts.route_overrides import dense_evidence_routes
    sel = _toy_selected()
    out = dense_evidence_routes(sel, needle_block=2)
    # reader row dense
    assert out[0, 0, 0, 5, :6].tolist() == [0, 1, 2, 3, 4, 5]
    # evidence row dense over its causal prefix, PAD after
    ev = out[0, 0, 0, 2].tolist()
    assert ev[:3] == [0, 1, 2] and all(b == -1 for b in ev[3:])
    # evidence present in every causal row
    for q in range(2, 6):
        assert 2 in out[0, 0, 0, q].tolist()
    validate_selected_blocks(out)


def test_candidate_span_collapses_overlap_and_clamps():
    from scripts.route_overrides import candidate_span
    # Adjacent candidates at W=1 overlap; union is 4 rows, not 2*(2*1+1)=6.
    assert candidate_span([2, 3], qb=8, neighborhood=1) == [1, 2, 3, 4]
    # Clamped at both ends rather than raising or wrapping.
    assert candidate_span([0, 7], qb=8, neighborhood=2) == [0, 1, 2, 5, 6, 7]
    assert candidate_span([3], qb=8) == [3]


def test_dense_candidate_routes_span_densifies_neighbors():
    from scripts.route_overrides import dense_candidate_routes
    sel = _toy_selected()
    out = dense_candidate_routes(sel, candidate_blocks=[3], neighborhood=1)
    for block in (2, 3, 4):
        row = out[0, 0, 0, block].tolist()
        assert row[:block + 1] == list(range(block + 1))
        assert all(entry == -1 for entry in row[block + 1:])
    validate_selected_blocks(out)


def test_dense_candidate_routes_no_oracle():
    from scripts.route_overrides import dense_candidate_routes
    sel = _toy_selected()
    out = dense_candidate_routes(sel, candidate_blocks=[3])
    # reader row dense
    assert out[0, 0, 0, 5, :6].tolist() == [0, 1, 2, 3, 4, 5]
    # candidate row 3 dense over causal prefix
    row = out[0, 0, 0, 3].tolist()
    assert row[:4] == [0, 1, 2, 3] and all(b == -1 for b in row[4:])
    # untouched row keeps its set
    assert set(out[0, 0, 0, 2].tolist()) - {-1} == {0, 1, 2}
    validate_selected_blocks(out)


def test_teacher_dual_top_p_routes_are_causal_sorted_and_variable_width():
    from scripts.route_overrides import teacher_dual_top_p_routes

    # Uniform residual mass with one sink, one recency, p=0.5 and a two-block
    # floor. Later rows therefore retain sink + recency + half the residual.
    target = torch.zeros(1, 1, 6, 6)
    for query in range(6):
        target[0, 0, query, :query + 1] = 1.0 / (query + 1)
    out = teacher_dual_top_p_routes(
        target, top_p=0.5, block_size=1, sink_tokens=1,
        recency_tokens=1, k_min_tokens=2)

    assert out.shape == (1, 1, 1, 6, 4)
    assert out[0, 0, 0, 0].tolist() == [0, -1, -1, -1]
    assert out[0, 0, 0, 5].tolist() == [0, 1, 2, 5]
    validate_selected_blocks(out, local_window=0)


def test_teacher_dual_top_p_uses_token_equivalent_floor_and_reserves():
    from scripts.route_overrides import teacher_dual_top_p_routes

    target = torch.zeros(1, 1, 10, 10)
    for query in range(10):
        target[0, 0, query, :query + 1] = 1.0 / (query + 1)
    out = teacher_dual_top_p_routes(
        target, top_p=0.01, block_size=64, sink_tokens=128,
        recency_tokens=256, k_min_tokens=512)

    # 512 tokens / 64 = eight blocks minimum once eight causal blocks exist.
    final = out[0, 0, 0, -1]
    assert int((final >= 0).sum()) == 8
    assert {0, 1}.issubset(set(final.tolist()))
    assert {6, 7, 8, 9}.issubset(set(final.tolist()))
    validate_selected_blocks(out)


def test_teacher_dual_top_p_rejects_invalid_inputs():
    from scripts.route_overrides import teacher_dual_top_p_routes

    target = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="top_p"):
        teacher_dual_top_p_routes(target, top_p=0.0)
    target[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        teacher_dual_top_p_routes(target, top_p=0.9)


def test_teacher_dual_top_p_unifies_head_groups_like_spotattention():
    from scripts.route_overrides import teacher_dual_top_p_routes

    target = torch.zeros(1, 2, 4, 4)
    target[:, 0, :, 0] = 1.0
    target[:, 1, :, 1] = 1.0
    out = teacher_dual_top_p_routes(
        target, top_p=0.5, block_size=1, sink_tokens=0,
        recency_tokens=1, k_min_tokens=1)
    assert torch.equal(out[:, :, 0], out[:, :, 1])


def test_charged_route_density_counts_dense_layers_against_sparsity():
    selected = _toy_selected()
    sparse = charged_route_density(selected)
    hybrid = charged_route_density(selected, dense_layers=[0])
    assert sparse["charged_block_entries"] == 15
    assert sparse["maximum_causal_block_entries"] == 21
    assert sparse["charged_attention_fraction"] == 15 / 21
    assert hybrid["charged_attention_fraction"] == 1.0
    assert hybrid["charged_sparsity"] == 0.0
    assert hybrid["dense_layers"] == [0]


def test_charged_route_density_rejects_invalid_layer():
    with pytest.raises(ValueError, match="dense layer"):
        charged_route_density(_toy_selected(), dense_layers=[1])
