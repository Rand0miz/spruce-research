"""Seam test: a load_teacher document fed straight into FlatGate. Nothing else in the
suite exercises this — teacher-side and selector-side tests each assert their own
shapes against hand-written literals, so a mismatch between load_teacher's output and
FlatGate's expected input would slip through. Run: python scripts/test_seam.py"""
import os, sys, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from selector.targets import load_teacher
from selector.gate import FlatGate

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


def test_teacher_doc_feeds_gate():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.pt")
        _make_proto_pt(p)
        doc = load_teacher(p)

        gate = FlatGate(L, d, proj_dim=d).eval()
        with torch.no_grad():
            scores = gate(doc["q_feat"], doc["k_feat"])
        assert scores.shape == (L, G, qb, kb), scores.shape


if __name__ == "__main__":
    test_teacher_doc_feeds_gate()
    print("OK test_seam")
