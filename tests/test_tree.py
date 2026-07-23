"""Tests for selector tree utilities."""
import torch

from selector.gate import FlatGate
from selector.tree import (
    KeyTreeLevel,
    _merge_node_prototypes,
    build_key_tree,
    build_target_tree,
    tree_kl_loss,
)
from scripts.eval_tree_traversal import (
    child_parent_index,
    leaf_ids_to_mask,
    traverse_to_leaf,
    traverse_to_leaf_ids,
)


@torch.no_grad()
def _full_level_traversal_reference(gate, q_feat, key_levels, beam):
    L, G, qb = q_feat.shape[:3]
    selected = torch.ones((L, G, qb, 1), dtype=torch.bool)
    query = torch.arange(qb)
    for child_index in range(len(key_levels) - 2, -1, -1):
        child = key_levels[child_index]
        parent = key_levels[child_index + 1]
        mapping = child_parent_index(parent, child)
        candidates = selected.index_select(-1, mapping)
        candidates &= (child.starts[None, :] <= query[:, None])[None, None]
        scores = gate(q_feat, child.features)
        masked = scores.masked_fill(~candidates, torch.finfo(scores.dtype).min)
        top = masked.topk(min(beam, child.features.shape[2]), dim=-1).indices
        selected = torch.zeros_like(candidates)
        selected.scatter_(-1, top, True)
        selected &= candidates
    return selected


def _key_tree_loop_reference(k_feat, radix):
    starts = torch.arange(k_feat.shape[2])
    ends = starts + 1
    levels = [KeyTreeLevel(0, k_feat, starts, ends)]
    current = k_feat
    while current.shape[2] > 1:
        previous = levels[-1]
        features, parent_starts, parent_ends = [], [], []
        for start in range(0, current.shape[2], radix):
            end = min(start + radix, current.shape[2])
            features.append(_merge_node_prototypes(current[:, :, start:end]))
            parent_starts.append(previous.starts[start])
            parent_ends.append(previous.ends[end - 1])
        current = torch.stack(features, dim=2)
        levels.append(KeyTreeLevel(
            len(levels), current, torch.stack(parent_starts), torch.stack(parent_ends)))
    return levels


def test_key_tree_shapes_and_ranges():
    L, G, kb, P, d = 2, 2, 5, 4, 3
    k = torch.randn(L, G, kb, P, d)
    levels = build_key_tree(k, radix=2)

    assert [lvl.features.shape[2] for lvl in levels] == [5, 3, 2, 1]
    assert levels[0].features.shape == (L, G, 5, P, d)
    assert levels[1].features.shape == (L, G, 3, P, d)
    assert levels[-1].starts.tolist() == [0]
    assert levels[-1].ends.tolist() == [5]

    for child, parent in zip(levels, levels[1:]):
        mapping = child_parent_index(parent, child)
        assert mapping.shape == child.starts.shape
        assert torch.all(parent.starts[mapping] <= child.starts)
        assert torch.all(child.ends <= parent.ends[mapping])


def test_vectorized_key_tree_matches_loop_reference():
    torch.manual_seed(11)
    k = torch.randn(2, 2, 7, 4, 5)
    for radix in (2, 3):
        expected = _key_tree_loop_reference(k, radix)
        actual = build_key_tree(k, radix)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.features, want.features)
            torch.testing.assert_close(got.starts, want.starts)
            torch.testing.assert_close(got.ends, want.ends)


def test_target_tree_sums_rows_and_masks_nodes():
    target = torch.tensor([[[
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.75, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.5, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.4, 0.0],
        [0.1, 0.2, 0.2, 0.2, 0.3],
    ]]])
    levels = build_target_tree(target, radix=2)

    assert [lvl.target.shape[-1] for lvl in levels] == [5, 3, 2, 1]
    for lvl in levels:
        rows = lvl.target.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows)), (lvl.level, rows)

    # Level 1 nodes cover [0,2), [2,4), [4,5). Query row 1 may only see node 0.
    assert levels[1].cmask[1].tolist() == [True, False, False]
    # Query row 2 has causal content in node 1 because that node starts at leaf 2.
    assert levels[1].cmask[2].tolist() == [True, True, False]


def test_candidate_only_traversal_matches_full_level_reference():
    torch.manual_seed(7)
    L, G, qb, P, d = 2, 2, 5, 3, 4
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, qb, P, d)
    gate = FlatGate(L, d, proj_dim=d).eval()
    levels = build_key_tree(k, radix=2)

    with torch.no_grad():
        expected = _full_level_traversal_reference(gate, q, levels, beam=2)
        actual = traverse_to_leaf(gate, q, levels, beam=2, radix=2)

    torch.testing.assert_close(actual, expected)


def test_compact_leaf_ids_match_mask_path_for_layer_chunks():
    torch.manual_seed(17)
    L, G, qb, P, d = 4, 2, 7, 3, 4
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, qb, P, d)
    gate = FlatGate(L, d, proj_dim=d).eval()
    levels = build_key_tree(k, radix=2)

    ids_one = traverse_to_leaf_ids(
        gate, q, levels, beam=3, radix=2, layer_chunk=1)
    ids_batched = traverse_to_leaf_ids(
        gate, q, levels, beam=3, radix=2, layer_chunk=2)
    mask = traverse_to_leaf(
        gate, q, levels, beam=3, radix=2, layer_chunk=2)

    torch.testing.assert_close(ids_batched, ids_one)
    torch.testing.assert_close(
        leaf_ids_to_mask(ids_batched, qb), mask)


def test_tree_kl_loss_backprops_to_gate():
    L, G, qb, kb, P, d = 2, 2, 4, 4, 3, 5
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, kb, P, d)
    raw = torch.rand(L, G, qb, kb)
    cmask = torch.arange(kb)[None, :] <= torch.arange(qb)[:, None]
    target = raw * cmask[None, None]
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    gate = FlatGate(L, d, proj_dim=d)
    key_levels = build_key_tree(k)
    target_levels = build_target_tree(target)
    loss, stats = tree_kl_loss(gate, q, key_levels, target_levels)
    loss.backward()

    assert len(stats) == len(key_levels)
    assert gate.Wq.grad is not None and torch.count_nonzero(gate.Wq.grad) > 0
    assert gate.Wk.grad is not None and torch.count_nonzero(gate.Wk.grad) > 0
