"""Tests for selector tree utilities."""
import torch

from selector.gate import FlatGate
from selector.tree import build_key_tree, build_target_tree, tree_kl_loss


def test_key_tree_shapes_and_ranges():
    L, G, kb, P, d = 2, 2, 5, 4, 3
    k = torch.randn(L, G, kb, P, d)
    levels = build_key_tree(k, radix=2)

    assert [lvl.features.shape[2] for lvl in levels] == [5, 3, 2, 1]
    assert levels[0].features.shape == (L, G, 5, P, d)
    assert levels[1].features.shape == (L, G, 3, P, d)
    assert levels[-1].starts.tolist() == [0]
    assert levels[-1].ends.tolist() == [5]


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
