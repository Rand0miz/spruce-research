import math
import os

import pytest
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    

from sparse.attention import sparse_prefill_attention_forward


class _AttentionModule(torch.nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx


def _selected(batch=1, layers=1, groups=1, qblocks=2, width=2):
    out = torch.full((batch, layers, groups, qblocks, width), -1, dtype=torch.int32)
    for q in range(qblocks):
        out[:, :, :, q, : q + 1] = torch.arange(q + 1, dtype=torch.int32)
    return out


def _causal_mask(length, dtype=torch.float32):
    mask = torch.full((1, 1, length, length), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def test_all_causal_blocks_matches_dense_attention():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 1, 4, 3)
    value = torch.randn(1, 1, 4, 3)
    output, weights = sparse_prefill_attention_forward(
        _AttentionModule(), query, key, value, _causal_mask(4),
        selected_blocks=_selected(groups=1), block_size=2,
    )
    dense_scores = torch.matmul(query, key.repeat_interleave(2, dim=1).transpose(-1, -2)) / math.sqrt(3)
    expected = torch.softmax(dense_scores + _causal_mask(4), dim=-1) @ value.repeat_interleave(2, dim=1)
    assert output.shape == (1, 4, 2, 3)
    torch.testing.assert_close(output, expected.transpose(1, 2))
    assert weights is None


def test_pruned_block_cannot_change_output():
    query = torch.ones(1, 1, 6, 1)
    key = torch.ones(1, 1, 6, 1)
    value = torch.tensor([[[[7.0], [7.0], [0.0], [0.0], [0.0], [0.0]]]])
    # The last query block keeps its mandatory local blocks 1 and 2, but drops
    # block 0.  Its non-zero values must therefore have no effect.
    selected = torch.tensor([[[[[0, -1, -1], [0, 1, -1], [1, 2, -1]]]]], dtype=torch.int32)
    output, _ = sparse_prefill_attention_forward(
        _AttentionModule(), query, key, value, _causal_mask(6),
        selected_blocks=selected, block_size=2,
    )
    assert torch.equal(output[0, 4:, 0], torch.zeros_like(output[0, 4:, 0]))


def test_fp32_score_accumulation_prevents_fp16_qk_overflow():
    query = torch.full((1, 1, 4, 64), 300.0, dtype=torch.float16)
    key = torch.full((1, 1, 4, 64), 300.0, dtype=torch.float16)
    value = torch.arange(4 * 64, dtype=torch.float16).reshape(1, 1, 4, 64)
    output, _ = sparse_prefill_attention_forward(
        _AttentionModule(), query, key, value, None,
        selected_blocks=_selected(), block_size=2,
    )
    assert torch.isfinite(output).all()


def test_rejects_decode_shape_and_bad_group_count():
    query = torch.randn(1, 2, 2, 4)
    key = value = torch.randn(1, 1, 3, 4)
    with pytest.raises(ValueError, match="prefill-only"):
        sparse_prefill_attention_forward(
            _AttentionModule(), query, key, value, None,
            selected_blocks=_selected(), block_size=2,
        )

    query = key = value = torch.randn(1, 2, 4, 4)
    with pytest.raises(ValueError, match="KV groups"):
        sparse_prefill_attention_forward(
            _AttentionModule(), query, key, value, _causal_mask(4),
            selected_blocks=_selected(groups=1), block_size=2,
        )
