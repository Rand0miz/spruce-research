"""Tests for load_teacher proto format."""
import pytest
import torch

from selector.targets import load_selector_features, load_teacher


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


def test_loads_proto_format(tmp_path):
    p = tmp_path / "t.pt"
    _make_proto_pt(p)
    doc = load_teacher(p)
    assert doc["q_feat"].shape == (L, G, qb, P, d), doc["q_feat"].shape
    assert doc["k_feat"].shape == (L, G, kb, P, d), doc["k_feat"].shape
    assert doc["target"].shape == (L, G, qb, kb), doc["target"].shape
    assert doc["meta"]["proto"] == P


def test_loads_selector_features_without_teacher_mass(tmp_path):
    p = tmp_path / "t.pt"
    _make_proto_pt(p)
    doc = load_selector_features(p)
    assert set(doc) == {"q_feat", "k_feat", "meta"}
    assert doc["q_feat"].shape == (L, G, qb, P, d)
    assert doc["k_feat"].shape == (L, G, kb, P, d)


def test_rejects_old_format(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({                                 # old 5-D pooledK, no proto key
        "pooled": torch.rand(1, L, H, qb, kb).half(),
        "pooledQ": torch.rand(1, L, H, qb, d).half(),
        "pooledK": torch.rand(1, L, G, kb, d).half(),
        "seq_len": qb * 64, "block_size": 64, "needle_block": 1,
    }, p)
    with pytest.raises(KeyError, match="Re-extract|re-extract|re-run"):
        load_teacher(p)
