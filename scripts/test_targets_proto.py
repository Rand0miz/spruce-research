"""Standalone test for load_teacher proto format. Run: python scripts/test_targets_proto.py"""
import os, sys, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
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


def test_loads_proto_format():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.pt")
        _make_proto_pt(p)
        doc = load_teacher(p)
        assert doc["q_feat"].shape == (L, G, qb, P, d), doc["q_feat"].shape
        assert doc["k_feat"].shape == (L, G, kb, P, d), doc["k_feat"].shape
        assert doc["target"].shape == (L, G, qb, kb), doc["target"].shape
        assert doc["meta"]["proto"] == P


def test_rejects_old_format():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "old.pt")
        torch.save({                                 # old 5-D pooledK, no proto key
            "pooled": torch.rand(1, L, H, qb, kb).half(),
            "pooledQ": torch.rand(1, L, H, qb, d).half(),
            "pooledK": torch.rand(1, L, G, kb, d).half(),
            "seq_len": qb * 64, "block_size": 64, "needle_block": 1,
        }, p)
        try:
            load_teacher(p)
        except KeyError as e:
            assert "re-extract" in str(e).lower() or "re-run" in str(e).lower()
            return
        raise AssertionError("old format should have raised KeyError")


if __name__ == "__main__":
    test_loads_proto_format()
    test_rejects_old_format()
    print("OK test_targets_proto")
