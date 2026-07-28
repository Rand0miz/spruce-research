"""Correctness-first block-sparse prefill attention for Hugging Face models.

``selected_blocks`` follows ``interfaces/selected_blocks_spec.md`` and is
indexed as ``[batch, layer, kv_head_group, query_block, K]``.  This module
implements dense PyTorch score computation with disallowed token blocks masked
out; it is deliberately a reference path, not the Stage 3.3 Triton kernel.
"""

import math
from typing import Optional

import torch

from interfaces.residual_summaries import (
    build_residual_tree_layout,
    validate_residual_summary_nodes,
)
from interfaces.validator import validate_selected_blocks
from sparse.config import SUMMARY_MODES, SUMMARY_PROTOTYPES
from sparse.summaries import (
    build_kv_summary_table,
    token_validity_from_attention_mask,
)


SPARSE_PREFILL_ATTENTION = "spruce_sparse_prefill"


def _layer_index(module: torch.nn.Module) -> int:
    """Get the layer index exposed by current decoder attention modules."""
    for name in ("layer_idx", "layer_id"):
        value = getattr(module, name, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Sparse prefill attention needs module.layer_idx (or module.layer_id) "
        "to select the correct selected_blocks layer."
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int,
    layer_idx: int,
    validate_blocks: bool,
) -> None:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be [batch, heads, tokens, head_dim]")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same batch size and head dimension")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError(
            "Stage 3.2 is prefill-only: query and key lengths must match "
            f"(got {query.shape[-2]} and {key.shape[-2]})."
        )
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if query.shape[1] % key.shape[1]:
        raise ValueError(
            f"query heads ({query.shape[1]}) must be divisible by KV heads ({key.shape[1]})")
    if validate_blocks:
        validate_selected_blocks(selected_blocks)
    batch, layers, groups, query_blocks, _ = selected_blocks.shape
    expected_blocks = math.ceil(query.shape[-2] / block_size)
    if batch != query.shape[0] or layers <= layer_idx:
        raise ValueError(
            f"selected_blocks shape {tuple(selected_blocks.shape)} does not cover "
            f"batch {query.shape[0]} and layer {layer_idx}")
    if groups != key.shape[1]:
        raise ValueError(
            f"selected_blocks has {groups} KV groups, model has {key.shape[1]} KV heads")
    if query_blocks != expected_blocks:
        raise ValueError(
            f"selected_blocks has {query_blocks} query blocks; expected {expected_blocks} "
            f"for seq_len={query.shape[-2]} and block_size={block_size}")


def sparse_prefill_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    selected_blocks: torch.Tensor,
    block_size: int,
    validate_selected_blocks_input: bool = True,
    residual_summaries: bool = False,
    residual_summary_nodes: Optional[torch.Tensor] = None,
    summary_prototypes: int = 1,
    summary_mode: str = "mean",
    summary_checkpoint: Optional[str] = None,
    validate_residual_summary_nodes_input: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transformers AttentionInterface implementation for sparse prefill.

    Pass ``selected_blocks`` and ``block_size`` as keyword arguments to the
    model forward.  The regular Transformers mask remains authoritative for
    causal and padding constraints; block selection only adds restrictions.
    """
    del kwargs
    layer_idx = _layer_index(module)
    _validate_inputs(
        query, key, value, selected_blocks, block_size, layer_idx,
        validate_blocks=validate_selected_blocks_input,
    )
    if summary_prototypes not in SUMMARY_PROTOTYPES:
        raise ValueError(
            f"summary_prototypes must be one of {SUMMARY_PROTOTYPES}, "
            f"got {summary_prototypes}"
        )
    if summary_mode not in SUMMARY_MODES:
        raise ValueError(
            f"summary_mode must be one of {SUMMARY_MODES}, got {summary_mode!r}"
        )
    if residual_summaries:
        if residual_summary_nodes is None:
            raise ValueError(
                "residual_summaries=True requires residual_summary_nodes; "
                "construct the complete frontier before model forward"
            )
        if summary_mode == "learned":
            if not summary_checkpoint:
                raise ValueError(
                    "summary_mode='learned' requires summary_checkpoint"
                )
            raise NotImplementedError(
                "The learned compressor is gated on deterministic-summary "
                "evaluation and is not implemented before that gate passes."
            )
        if validate_residual_summary_nodes_input:
            validate_residual_summary_nodes(
                selected_blocks,
                residual_summary_nodes,
                validate_selected=not validate_selected_blocks_input,
            )

    B, Hq, T, D = query.shape
    Hkv = key.shape[1]
    group_size = Hq // Hkv
    if attention_mask is not None and attention_mask.dim() != 4:
        raise ValueError(
            "Sparse prefill expects the Transformers 4D additive attention mask; "
            f"got shape {tuple(attention_mask.shape)}")

    # Process one query block and KV group at a time.  Besides being the
    # literal block-sparse operation, this prevents the reference path from
    # allocating [batch, heads, seq_len, seq_len] scores.
    output = torch.empty_like(query)
    selected = selected_blocks[:, layer_idx].to(
        device=query.device, dtype=torch.long
    )
    qblocks = selected.shape[2]
    score_floor = torch.finfo(query.dtype).min
    token_offsets = torch.arange(block_size, device=query.device)
    residual = None
    summary_table = None
    if residual_summaries:
        if residual_summary_nodes.shape[:4] != selected_blocks.shape[:4]:
            raise ValueError(
                "residual_summary_nodes must match selected_blocks through "
                "the query-block dimension"
            )
        residual = residual_summary_nodes[:, layer_idx].to(
            device=query.device, dtype=torch.long
        )
        layout = build_residual_tree_layout(qblocks)
        token_mask = token_validity_from_attention_mask(
            attention_mask, batch=B, seq_len=T
        )
        summary_table = build_kv_summary_table(
            key,
            value,
            layout,
            block_size=block_size,
            prototypes=summary_prototypes,
            token_mask=token_mask,
        )

    for qblock in range(qblocks):
        q_start = qblock * block_size
        q_end = min(q_start + block_size, T)
        q_positions = torch.arange(q_start, q_end, device=query.device)
        for batch_idx in range(B):
            for kv_head in range(Hkv):
                block_ids = selected[batch_idx, kv_head, qblock]
                block_ids = block_ids[block_ids >= 0]
                if block_ids.numel() == 0:
                    raise ValueError(
                        "selected_blocks leaves an attention row empty at "
                        f"batch={batch_idx}, layer={layer_idx}, "
                        f"kv_head={kv_head}, query_block={qblock}"
                    )
                # Expand block IDs entirely on-device.  The previous
                # ``block_ids.tolist()`` implementation forced a GPU
                # synchronization for every query-block/head pair, which
                # made the real-scale correctness reference needlessly slow.
                key_indices = (
                    block_ids[:, None] * block_size + token_offsets[None, :]
                ).flatten()
                key_indices = key_indices[key_indices < T]
                head_start = kv_head * group_size
                head_end = head_start + group_size
                q = query[batch_idx:batch_idx + 1, head_start:head_end, q_start:q_end]
                k = key[batch_idx:batch_idx + 1, kv_head:kv_head + 1].index_select(2, key_indices)
                v = value[batch_idx:batch_idx + 1, kv_head:kv_head + 1].index_select(2, key_indices)
                # Accumulate QK in FP32.  Casting only inside softmax is too
                # late: at real context lengths Qwen contains outlier Q/K
                # pairs whose FP16 dot product overflows to inf, making the
                # entire probability row NaN.
                scores = torch.matmul(
                    q.float(), k.float().transpose(-1, -2)
                ) / math.sqrt(D)

                # Same-block routing is legal, but future tokens within the
                # block are not. Keep causality even if no HF mask was passed.
                causal = key_indices[None, :] <= q_positions[:, None]
                scores = scores.masked_fill(~causal[None, None], score_floor)
                if attention_mask is not None:
                    sparse_mask = attention_mask[
                        batch_idx:batch_idx + 1, :, q_start:q_end
                    ].index_select(-1, key_indices)
                    scores = scores + sparse_mask.to(dtype=scores.dtype)

                combined_values = v.float()
                if residual is not None:
                    node_ids = residual[batch_idx, kv_head, qblock]
                    node_ids = node_ids[node_ids >= 0]
                    if node_ids.numel():
                        summary_counts = summary_table.counts[
                            batch_idx
                        ].index_select(0, node_ids).flatten()
                        valid_prototypes = summary_counts > 0
                        summary_keys = summary_table.keys[
                            batch_idx, kv_head
                        ].index_select(0, node_ids).flatten(0, 1)
                        summary_values = summary_table.values[
                            batch_idx, kv_head
                        ].index_select(0, node_ids).flatten(0, 1)
                        summary_bias = summary_table.log_counts[
                            batch_idx
                        ].index_select(0, node_ids).flatten()
                        summary_keys = summary_keys[valid_prototypes]
                        summary_values = summary_values[valid_prototypes]
                        summary_bias = summary_bias[valid_prototypes]
                        if summary_keys.numel():
                            summary_scores = torch.matmul(
                                q.float(),
                                summary_keys[None, None].transpose(-1, -2),
                            ) / math.sqrt(D)
                            summary_scores = (
                                summary_scores
                                + summary_bias[None, None, None, :]
                            )
                            scores = torch.cat([scores, summary_scores], dim=-1)
                            combined_values = torch.cat(
                                [combined_values, summary_values[None, None]],
                                dim=2,
                            )
                probabilities = torch.softmax(scores, dim=-1)
                output[batch_idx:batch_idx + 1, head_start:head_end, q_start:q_end] = torch.matmul(
                    probabilities, combined_values
                ).to(query.dtype)

    # Transformers' Qwen attention module reshapes this directly to
    # [batch, tokens, hidden_size] before its output projection.  Attention
    # backends therefore return token-major [B, T, H, D], as SDPA does.
    return output.transpose(1, 2).contiguous(), None


def register_sparse_prefill_attention(name: str = SPARSE_PREFILL_ATTENTION) -> str:
    """Register the Stage 3.2 reference backend with Transformers.

    Registration is explicit so importing this package never changes global
    Transformers state.  The matching SDPA mask formatter preserves the normal
    decoder causal/padding mask while the backend applies sparse restrictions.
    """
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
    except ImportError as error:  # keeps tensor tests independent of Transformers
        raise RuntimeError("register_sparse_prefill_attention requires transformers") from error

    AttentionInterface.register(name, sparse_prefill_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name
