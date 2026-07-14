"""Standalone test for _group_avg_tokens. Run: python scripts/test_group_avg.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from teacher.chunked_extract import _group_avg_tokens


def test_group_average():
    b, H, L, d, G = 1, 12, 10, 4, 2                 # rep = 6 query heads per group
    x = torch.randn(b, H, L, d)
    out = _group_avg_tokens(x, G)
    assert out.shape == (b, G, L, d), out.shape
    # group 0 == mean of the first rep=6 heads
    assert torch.allclose(out[:, 0], x[:, 0:6].mean(dim=1), atol=1e-6)
    assert torch.allclose(out[:, 1], x[:, 6:12].mean(dim=1), atol=1e-6)


if __name__ == "__main__":
    test_group_average()
    print("OK test_group_avg")
