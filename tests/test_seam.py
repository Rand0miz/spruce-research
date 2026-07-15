"""Seam test: load_teacher output feeds straight into FlatGate."""
import torch

from selector.gate import FlatGate
from selector.targets import load_teacher


L, H, G, qb, kb, P, d = 2, 4, 2, 3, 3, 8, 4


def _make_proto_pt(path):
    pooled = torch.rand(1, L, H, qb, kb)            # teacher mass (H heads)
    torch.save({
        "pooled": pooled.half(),
        "pooledQ": torch.rand(1, L, G, qb, P, d).half(),
        "pooledK": torch.rand(1, L, G, kb, P, d).half(),
        "seq_len": qb * 64, "block_size": 64, "needle_block": 1,
        "proto": P, "store_dtype": "float16",
    }, path)


def test_teacher_doc_feeds_gate(tmp_path):
    p = tmp_path / "t.pt"
    _make_proto_pt(p)
    doc = load_teacher(p)

    gate = FlatGate(L, d, proj_dim=d).eval()
    with torch.no_grad():
        scores = gate(doc["q_feat"], doc["k_feat"])
    assert scores.shape == (L, G, qb, kb), scores.shape
