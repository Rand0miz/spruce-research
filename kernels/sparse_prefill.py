"""SPRUCE selected-block adapter and Triton sparse-prefill backend."""
import math

import torch

from kernels.seerattention_direct_index_triton import (
    KERNEL_VARIANTS,
    direct_index_sparse_triton,
)


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


def selected_blocks_to_head_indices(selected_blocks, *, layer_idx, num_query_heads):
    """Expand SPRUCE KV-group selections to direct Triton query-head indices."""
    if selected_blocks.dim() != 5:
        raise ValueError("selected_blocks must be [B,L,Gkv,Qb,K]")
    _, layers, kv_groups, _, _ = selected_blocks.shape
    if not 0 <= layer_idx < layers:
        raise ValueError(f"layer_idx {layer_idx} is outside selected_blocks layers={layers}")
    if num_query_heads % kv_groups:
        raise ValueError("num_query_heads must be divisible by selected KV groups")
    return selected_blocks[:, layer_idx].repeat_interleave(
        num_query_heads // kv_groups, dim=1).contiguous()


def triton_sparse_prefill_attention_forward(module, query, key, value, attention_mask, *,
                                             selected_blocks, block_size,
                                             kernel_variant="single_head", **kwargs):
    """Transformers AttentionInterface callback backed by Triton (prefill only)."""
    # SPRUCE deliberately keeps decode dense. ``generate`` keeps forwarding
    # selected_blocks after prefill, so dispatch one-token cached calls to the
    # standard SDPA implementation instead of treating them as an error.
    if query.shape[-2] != key.shape[-2]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    if attention_mask is not None:
        raise ValueError("Triton SPRUCE prefill currently supports unpadded batch-size-1 inputs only")
    if key.shape != value.shape:
        raise ValueError("key and value must have matching shapes")
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
    # Keep indices and K/V at native KV-group resolution. The direct kernel
    # maps query head -> KV head internally, avoiding sixfold K/V expansion on
    # Qwen2.5-Coder-1.5B (12 query heads, 2 KV heads).
    indices = selected_blocks[:, layer_idx].contiguous()
    output = direct_index_sparse_triton(
        query, key, value, indices, scale=D ** -0.5,
        block_size=block_size, kernel_variant=kernel_variant)
    return output, None


def _triton_attention_mask(*args, **kwargs):
    """Avoid O(T²) masks at Triton prefill, retain SDPA's mask for decode."""
    q_length = kwargs.get("q_length", args[1] if len(args) > 1 else None)
    kv_length = kwargs.get("kv_length", args[2] if len(args) > 2 else None)
    if q_length == kv_length:
        return None
    from transformers.masking_utils import sdpa_mask
    return sdpa_mask(*args, **kwargs)


def register_triton_sparse_prefill_attention(name=SPRUCE_TRITON_SPARSE_PREFILL):
    """Register the kernel backend without creating an O(T²) HF attention mask."""
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
    except ImportError as error:
        raise RuntimeError("Triton registration requires transformers") from error
    AttentionInterface.register(name, triton_sparse_prefill_attention_forward)
    AttentionMaskInterface.register(name, _triton_attention_mask)
    return name
