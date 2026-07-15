"""Tests for _proto_pool_seq."""
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


def _needle_survives_dtype(dtype, atol):
    # Regression test for the fp16-overflow bug (Finding 2): a squared-L2 distance
    # of magnitude-40 outlier vs a near-zero mean, in d=128, overflows fp16's ~65504
    # max (128 * ~39.4^2 ~= 198k), so under the OLD code (which computed dist in x's
    # own dtype and used torch.isinf(sel_dist) to detect pad slots) the needle's own
    # +inf overflow distance was misread as "pad" and replaced by the block mean --
    # silently dropping the exact token this whole feature exists to keep. Distance
    # math must run in fp32 and "pad" must be derived from the validity mask, not
    # from isinf, so this must pass in BOTH fp16 and bf16.
    b, H, d, block, P = 1, 1, 128, 64, 8
    torch.manual_seed(0)
    x = (torch.randn(b, H, block, d, dtype=torch.float32) * 0.01).to(dtype)
    x[0, 0, 37, :] = torch.full((d,), 40.0, dtype=dtype)   # the needle
    out = _proto_pool_seq(x, block, P)                    # [1,1,1,P,d]
    protos = out[0, 0, 0].float()                         # [P,d]
    needle = x[0, 0, 37].float()
    hit = (protos - needle).pow(2).sum(-1).min().item()
    assert hit < atol, f"needle not preserved in {dtype}, min dist^2={hit}"


def test_needle_survives_fp16():
    _needle_survives_dtype(torch.float16, atol=1.0)


def test_needle_survives_bf16():
    _needle_survives_dtype(torch.bfloat16, atol=1.0)


def test_short_block_pads_with_mean():
    # ragged last block with fewer real rows than P-1 outliers
    b, H, d, block, P = 1, 1, 4, 64, 8
    L = 64 + 3                                       # last block has 3 real rows
    x = torch.randn(b, H, L, d)
    out = _proto_pool_seq(x, block, P)              # [1,1,2,P,d]
    last = out[0, 0, 1]                             # [P,d]  last block
    real_mean = x[0, 0, 64:, :].mean(dim=0)
    assert torch.allclose(last[0], real_mean, atol=1e-5)   # proto 0 = real-row mean
    # no proto is a zero vector (all spare slots filled with mean, not pad rows)
    assert not (last == 0).all(dim=-1).any()
