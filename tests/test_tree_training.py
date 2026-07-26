from types import SimpleNamespace

import torch

from selector.gate import FlatGate
from selector.targets import causal_block_mask
from selector.train import train_document


class CountingGate(FlatGate):
    def __init__(self, num_layers, head_dim, proj_dim=None):
        super().__init__(num_layers, head_dim, proj_dim)
        self.scored_node_counts = []

    def forward(self, q_feat, k_feat):
        self.scored_node_counts.append(int(k_feat.shape[2]))
        return super().forward(q_feat, k_feat)


def _args(**overrides):
    values = {
        "needle_topk": 2,
        "topk": 2,
        "tree_supervision": True,
        "tree_radix": 2,
        "lambda_topk": 0.5,
        "lambda_boundary": 0.5,
        "lambda_needle": 1.0,
        "topk_margin": 0.25,
        "needle_margin": 0.25,
        "needle_objective": "union",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_train_document_scores_every_discriminative_tree_level():
    torch.manual_seed(31)
    layers, groups, blocks, protos, head_dim = 2, 2, 8, 2, 4
    q_feat = torch.randn(
        layers, groups, blocks, protos, head_dim)
    k_feat = torch.randn(
        layers, groups, blocks, protos, head_dim)
    cmask = causal_block_mask(blocks, blocks)
    raw = torch.rand(layers, groups, blocks, blocks)
    raw[..., -1, 5] = 10.0
    raw = raw * cmask[None, None]
    target = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    doc = {
        "q_feat": q_feat,
        "k_feat": k_feat,
        "target": target,
        "cmask": cmask,
        "meta": {"kb": blocks, "needle_block": 5},
    }

    gate = CountingGate(layers, head_dim, proj_dim=head_dim)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    before = gate.Wq.detach().clone()
    stats = train_document(gate, optimizer, doc, _args())

    assert gate.scored_node_counts == [8, 4, 2]
    assert stats["levels"] == 3
    assert stats["boundary_rows"] > 0
    assert stats["needle_units"] > 0
    assert not torch.equal(before, gate.Wq.detach())


def test_train_document_preserves_leaf_only_compatibility():
    torch.manual_seed(37)
    layers, groups, blocks, protos, head_dim = 1, 1, 4, 2, 3
    q_feat = torch.randn(
        layers, groups, blocks, protos, head_dim)
    k_feat = torch.randn(
        layers, groups, blocks, protos, head_dim)
    cmask = causal_block_mask(blocks, blocks)
    raw = torch.rand(layers, groups, blocks, blocks) * cmask[None, None]
    target = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    doc = {
        "q_feat": q_feat,
        "k_feat": k_feat,
        "target": target,
        "cmask": cmask,
        "meta": {"kb": blocks, "needle_block": 2},
    }
    gate = CountingGate(layers, head_dim, proj_dim=head_dim)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)

    stats = train_document(
        gate, optimizer, doc,
        _args(
            tree_supervision=False,
            lambda_boundary=0.0,
            lambda_needle=0.0,
            needle_objective="group",
        ),
    )

    assert gate.scored_node_counts == [4]
    assert stats["levels"] == 1
