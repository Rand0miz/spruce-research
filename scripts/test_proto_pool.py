"""Standalone test for _proto_pool_seq. Run: python scripts/test_proto_pool.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from teacher.chunked_extract import _proto_pool_seq


def test_shape_and_mean():
    b, H, L, d, block, P = 1, 2, 128, 4, 64, 8
    x = torch.randn(b, H, L, d)
    out = _proto_pool_seq(x, block, P)
    assert out.shape == (b, H, 2, P, d), out.shape
    # proto 0 == full-block mean for a full (non-ragged) block
    mean0 = x[:, :, :block, :].mean(dim=2)
    assert torch.allclose(out[:, :, 0, 0, :], mean0, atol=1e-5)


def test_outlier_is_kept():
    # one token planted far from the rest -> must appear among protos 1..P-1
    b, H, d, block, P = 1, 1, 4, 64, 8
    x = torch.randn(b, H, block, d) * 0.01          # tight cluster
    x[0, 0, 37, :] = torch.tensor([50.0, 50.0, 50.0, 50.0])   # the needle
    out = _proto_pool_seq(x, block, P)              # [1,1,1,P,d]
    protos = out[0, 0, 0]                           # [P, d]
    needle = x[0, 0, 37]
    hit = (protos - needle).pow(2).sum(-1).min().item()
    assert hit < 1e-6, f"needle not preserved, min dist^2={hit}"


def test_short_block_pads_with_mean():
    # ragged last block with fewer real rows than P-1 outliers
    b, H, d, block, P = 1, 1, 4, 64, 8
    L = 64 + 3                                       # last block has 3 real rows
    x = torch.randn(b, H, L, d)
    out = _proto_pool_seq(x, block, P)              # [1,1,2,P,d]
    last = out[0, 0, 1]                             # [P,d]  last block
    real_mean = x[0, 0, 64:, :].mean(dim=0)
    assert torch.allclose(last[0], real_mean, atol=1e-5)   # proto 0 = real-row mean
    # no proto is a zero-pad row (all pad rows were masked out of selection)
    assert not torch.allclose(last, torch.zeros_like(last))


if __name__ == "__main__":
    test_shape_and_mean()
    test_outlier_is_kept()
    test_short_block_pads_with_mean()
    print("OK test_proto_pool")
