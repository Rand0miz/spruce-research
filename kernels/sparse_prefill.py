"""SPRUCE selected-block adapter and Triton sparse-prefill backend."""
import math

import torch

from kernels.seerattention_triton import block_sparse_triton


SPRUCE_TRITON_SPARSE_PREFILL = "spruce_triton_sparse_prefill"


def selected_blocks_to_block_mask(selected_blocks, *, layer_idx, num_query_heads):
    """Convert compact SPRUCE selections to SeerAttention's boolean block mask.

    Input: ``[B, L, Gkv, Qb, K]`` int32 selected IDs.
    Output: ``[B, Hq, Qb, Qb]`` bool, with KV groups expanded over Q heads.
    This uses GPU scatter operations only; it never creates a token-level mask.
    """
    if selected_blocks.dim() != 5:
        raise ValueError("selected_blocks must be [B,L,Gkv,Qb,K]")
    B, L, Gkv, qblocks, _ = selected_blocks.shape
    if not 0 <= layer_idx < L:
        raise ValueError(f"layer_idx {layer_idx} is outside selected_blocks layers={L}")
    if num_query_heads % Gkv:
        raise ValueError("num_query_heads must be divisible by selected KV groups")
    rows = selected_blocks[:, layer_idx].to(torch.long)
    valid = (rows >= 0) & (rows < qblocks)
    # Scatter padding into a throwaway final column.  Clamping -1 to zero would
    # let trailing padding overwrite a real selection of key block zero.
    ids = rows.clamp(min=0, max=qblocks - 1).masked_fill(~valid, qblocks)
    mask_with_pad = torch.zeros(
        (B, Gkv, qblocks, qblocks + 1), device=rows.device, dtype=torch.bool)
    mask_with_pad.scatter_(-1, ids, True)
    mask = mask_with_pad[..., :qblocks]
    return mask.repeat_interleave(num_query_heads // Gkv, dim=1).contiguous()


def triton_sparse_prefill_attention_forward(module, query, key, value, attention_mask, *,
                                             selected_blocks, block_size, **kwargs):
    """Transformers AttentionInterface callback backed by Triton (prefill only)."""
    del kwargs
    if attention_mask is not None:
        raise ValueError("Triton SPRUCE prefill currently supports unpadded batch-size-1 inputs only")
    if query.shape[-2] != key.shape[-2] or key.shape != value.shape:
        raise ValueError("Triton SPRUCE backend is prefill-only with matching Q/K/V sequence lengths")
    if block_size != 64:
        raise ValueError("Triton SPRUCE backend currently requires block_size=64")
    layer_idx = int(getattr(module, "layer_idx"))
    B, Hq, T, D = query.shape
    Hkv = key.shape[1]
    if Hq % Hkv:
        raise ValueError("query heads must divide evenly into KV groups")
    if selected_blocks.shape[0] != B or selected_blocks.shape[2] != Hkv:
        raise ValueError("selected_blocks batch/KV-group dimensions do not match Qwen attention")
    if selected_blocks.shape[3] != math.ceil(T / block_size):
        raise ValueError("selected_blocks query-block count does not match sequence length")
    mask = selected_blocks_to_block_mask(selected_blocks, layer_idx=layer_idx, num_query_heads=Hq)
    group_size = Hq // Hkv
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)
    output = block_sparse_triton(query, key, value, mask, scale=D ** -0.5, block_size=block_size)
    return output.transpose(1, 2).contiguous(), None


def _no_attention_mask(*args, **kwargs):
    """The kernel owns causal masking; Stage 3.3 does not support padding yet."""
    return None


def register_triton_sparse_prefill_attention(name=SPRUCE_TRITON_SPARSE_PREFILL):
    """Register the kernel backend without creating an O(T²) HF attention mask."""
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
    except ImportError as error:
        raise RuntimeError("Triton registration requires transformers") from error
    AttentionInterface.register(name, triton_sparse_prefill_attention_forward)
    AttentionMaskInterface.register(name, _no_attention_mask)
    return name
