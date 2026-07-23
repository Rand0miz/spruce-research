import os

import pytest
import torch

from kernels.sparse_prefill import (
    KERNEL_VARIANTS,
    selected_blocks_to_head_indices,
    selected_blocks_to_block_mask,
    triton_sparse_prefill_attention_forward,
)
from sparse.attention import sparse_prefill_attention_forward


class _AttentionModule(torch.nn.Module):
    layer_idx = 0


def test_selected_blocks_adapter_expands_kv_groups_to_query_heads():
    # B=1, L=1, Gkv=2, query blocks=3, K=3.  The two KV groups route
    # differently, and each must expand to two Q heads.
    selected = torch.tensor([[[[
        [0, -1, -1], [0, 1, -1], [0, 1, 2],
    ], [
        [0, -1, -1], [1, -1, -1], [0, 2, -1],
    ]]]], dtype=torch.int32)
    mask = selected_blocks_to_block_mask(selected, layer_idx=0, num_query_heads=4)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4, 3, 3)
    expected_g0 = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    expected_g1 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=torch.bool)
    torch.testing.assert_close(mask[0, 0], expected_g0)
    torch.testing.assert_close(mask[0, 1], expected_g0)
    torch.testing.assert_close(mask[0, 2], expected_g1)
    torch.testing.assert_close(mask[0, 3], expected_g1)


def test_selected_blocks_adapter_ignores_padding():
    selected = torch.full((1, 1, 1, 2, 3), -1, dtype=torch.int32)
    selected[0, 0, 0, 0, 0] = 0
    selected[0, 0, 0, 1, :2] = torch.tensor([0, 1], dtype=torch.int32)
    mask = selected_blocks_to_block_mask(selected, layer_idx=0, num_query_heads=1)
    assert mask[0, 0].tolist() == [[True, False], [True, True]]


def test_selected_blocks_head_indices_preserve_compact_rows():
    selected = torch.tensor([[[[[0, -1], [0, 1]]]]], dtype=torch.int32)
    indices = selected_blocks_to_head_indices(selected, layer_idx=0, num_query_heads=2)
    assert indices.shape == (1, 2, 2, 2)
    assert indices[0, 0].tolist() == [[0, -1], [0, 1]]
    torch.testing.assert_close(indices[0, 0], indices[0, 1])


@pytest.mark.skipif(
    os.environ.get("RUN_TRITON_TESTS") != "1",
    reason="set RUN_TRITON_TESTS=1 on CUDA to compile and run Triton parity tests",
)
@pytest.mark.parametrize("kernel_variant", KERNEL_VARIANTS)
def test_triton_kernel_matches_pytorch_sparse_reference(kernel_variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(0)
    # Match Qwen's real projection layout: contiguous [B,T,H,D] storage
    # viewed as non-contiguous [B,H,T,D]. The Triton path must not copy it.
    q = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    assert not q.is_contiguous()
    selected = torch.zeros((1, 1, 1, 1, 1), device="cuda", dtype=torch.int32)
    module = _AttentionModule().cuda()
    expected, _ = sparse_prefill_attention_forward(
        module, q, k, v, None, selected_blocks=selected, block_size=64,
        validate_selected_blocks_input=False,
    )
    actual, _ = triton_sparse_prefill_attention_forward(
        module, q, k, v, None, selected_blocks=selected, block_size=64,
        kernel_variant=kernel_variant,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    os.environ.get("RUN_TRITON_TESTS") != "1",
    reason="set RUN_TRITON_TESTS=1 on CUDA to compile and run Triton parity tests",
)
@pytest.mark.parametrize("kernel_variant", KERNEL_VARIANTS)
def test_triton_tiled_gqa_kernel_matches_reference_across_blocks(kernel_variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(4)
    q = torch.randn(
        1, 128, 4, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(
        1, 128, 2, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(
        1, 128, 2, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    selected = torch.tensor(
        [[[[[0, -1], [0, 1]], [[0, -1], [0, 1]]]]],
        device="cuda", dtype=torch.int32)
    module = _AttentionModule().cuda()
    expected, _ = sparse_prefill_attention_forward(
        module, q, k, v, None, selected_blocks=selected, block_size=64,
        validate_selected_blocks_input=False,
    )
    actual, _ = triton_sparse_prefill_attention_forward(
        module, q, k, v, None, selected_blocks=selected, block_size=64,
        kernel_variant=kernel_variant,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
