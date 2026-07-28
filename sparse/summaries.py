"""Deterministic live K/V summaries for residual tree nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from interfaces.residual_summaries import ResidualTreeLayout
from sparse.config import SUMMARY_PROTOTYPES


@dataclass(frozen=True)
class KVSummaryTable:
    """All per-layer node prototypes built from the live post-RoPE K/V.

    Keys and values are ``[B, Hkv, node, P, D]``. Counts and log-count biases
    are ``[B, node, P]`` because padding validity can vary by batch item.
    """

    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    log_counts: torch.Tensor
    range_starts: torch.Tensor
    range_ends: torch.Tensor


def _prototype_token_ranges(
    layout: ResidualTreeLayout,
    *,
    seq_len: int,
    block_size: int,
    prototypes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_starts = torch.tensor(layout.starts, device=device, dtype=torch.long)
    block_ends = torch.tensor(layout.ends, device=device, dtype=torch.long)
    token_starts = (block_starts * block_size).clamp_max(seq_len)
    token_ends = (block_ends * block_size).clamp_max(seq_len)
    lengths = token_ends - token_starts
    prototype_ids = torch.arange(prototypes, device=device, dtype=torch.long)
    starts = token_starts[:, None] + (
        lengths[:, None] * prototype_ids[None, :] // prototypes
    )
    ends = token_starts[:, None] + (
        lengths[:, None] * (prototype_ids[None, :] + 1) // prototypes
    )
    return starts, ends


def build_kv_summary_table(
    key: torch.Tensor,
    value: torch.Tensor,
    layout: ResidualTreeLayout,
    *,
    block_size: int,
    prototypes: int,
    token_mask: torch.Tensor | None = None,
) -> KVSummaryTable:
    """Pool live K/V into positional means using vectorized prefix reductions."""
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("key and value must be [batch, kv_heads, tokens, head_dim]")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {key.shape} vs {value.shape}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if prototypes not in SUMMARY_PROTOTYPES:
        raise ValueError(
            f"prototypes must be one of {SUMMARY_PROTOTYPES}, got {prototypes}"
        )

    batch, kv_heads, seq_len, head_dim = key.shape
    expected_leaves = (seq_len + block_size - 1) // block_size
    if layout.num_leaves != expected_leaves:
        raise ValueError(
            f"tree has {layout.num_leaves} leaves, expected {expected_leaves} "
            f"for seq_len={seq_len}, block_size={block_size}"
        )
    if token_mask is None:
        valid = torch.ones((batch, seq_len), device=key.device, dtype=torch.bool)
    else:
        if token_mask.shape != (batch, seq_len):
            raise ValueError(
                f"token_mask must be {(batch, seq_len)}, got {tuple(token_mask.shape)}"
            )
        valid = token_mask.to(device=key.device, dtype=torch.bool)

    range_starts, range_ends = _prototype_token_ranges(
        layout,
        seq_len=seq_len,
        block_size=block_size,
        prototypes=prototypes,
        device=key.device,
    )
    flat_starts = range_starts.flatten()
    flat_ends = range_ends.flatten()
    valid_float = valid.to(dtype=torch.float32)

    def prefix_pool(source: torch.Tensor) -> torch.Tensor:
        weighted = source.float() * valid_float[:, None, :, None]
        prefix = torch.cat(
            [
                torch.zeros(
                    (batch, kv_heads, 1, head_dim),
                    device=source.device,
                    dtype=torch.float32,
                ),
                weighted.cumsum(dim=2),
            ],
            dim=2,
        )
        pooled = prefix.index_select(2, flat_ends) - prefix.index_select(2, flat_starts)
        return pooled.view(batch, kv_heads, layout.num_nodes, prototypes, head_dim)

    count_prefix = torch.cat(
        [
            torch.zeros((batch, 1), device=key.device, dtype=torch.float32),
            valid_float.cumsum(dim=1),
        ],
        dim=1,
    )
    counts = (
        count_prefix.index_select(1, flat_ends)
        - count_prefix.index_select(1, flat_starts)
    ).view(batch, layout.num_nodes, prototypes)
    denominator = counts[:, None, :, :, None].clamp_min(1.0)
    key_means = prefix_pool(key) / denominator
    value_means = prefix_pool(value) / denominator
    log_counts = torch.where(
        counts > 0,
        counts.log(),
        torch.full_like(counts, float("-inf")),
    )
    return KVSummaryTable(
        keys=key_means,
        values=value_means,
        counts=counts,
        log_counts=log_counts,
        range_starts=range_starts,
        range_ends=range_ends,
    )


def token_validity_from_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    seq_len: int,
) -> torch.Tensor | None:
    """Extract key padding validity from a standard 4D additive decoder mask."""
    if attention_mask is None:
        return None
    if attention_mask.dim() != 4:
        raise ValueError(
            "residual summaries expect a 4D additive attention mask, "
            f"got {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[0] != batch or attention_mask.shape[-1] != seq_len:
        raise ValueError("attention mask batch/key dimensions do not match K/V")
    # A non-padding key has at least one allowed query row. Reducing over query
    # rows also handles right-padded batches whose final row is itself padding.
    return (attention_mask[:, 0] >= 0).any(dim=-2)


def residual_attention_density(
    selected_blocks: torch.Tensor,
    residual_summary_nodes: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    prototypes: int,
) -> dict:
    """Charge exact blocks and valid residual prototypes in token entries."""
    if selected_blocks.dim() != 5 or residual_summary_nodes.dim() != 5:
        raise ValueError("selected and residual routes must both be 5D")
    if selected_blocks.shape[:4] != residual_summary_nodes.shape[:4]:
        raise ValueError("selected and residual routes must match through Q")
    if prototypes not in SUMMARY_PROTOTYPES:
        raise ValueError(
            f"prototypes must be one of {SUMMARY_PROTOTYPES}, got {prototypes}"
        )
    batch, layers, groups, query_blocks, _ = selected_blocks.shape
    if query_blocks != (seq_len + block_size - 1) // block_size:
        raise ValueError("sequence length and route query-block count disagree")

    # Local import avoids making the interface layer depend on sparse code.
    from interfaces.residual_summaries import build_residual_tree_layout

    tree = build_residual_tree_layout(query_blocks)
    valid_prototypes = []
    for start, end in zip(tree.starts, tree.ends):
        token_start = min(start * block_size, seq_len)
        token_end = min(end * block_size, seq_len)
        valid_prototypes.append(min(prototypes, token_end - token_start))
    valid_per_node = torch.tensor(valid_prototypes, dtype=torch.int64)

    exact_entries = int((selected_blocks >= 0).sum().item()) * block_size
    residual_cpu = residual_summary_nodes.detach().to(device="cpu", dtype=torch.long)
    summary_entries = 0
    valid_ids = residual_cpu[residual_cpu >= 0]
    if valid_ids.numel():
        summary_entries = int(valid_per_node.index_select(0, valid_ids).sum().item())
    charged_entries = exact_entries + summary_entries
    maximum_entries = (
        batch
        * layers
        * groups
        * query_blocks
        * (query_blocks + 1)
        // 2
        * block_size
    )
    fraction = charged_entries / maximum_entries if maximum_entries else 0.0
    return {
        "exact_attention_entries": exact_entries,
        "summary_prototype_entries": summary_entries,
        "charged_attention_entries": charged_entries,
        "maximum_causal_attention_entries": maximum_entries,
        "charged_attention_fraction": fraction,
        "charged_sparsity": 1.0 - fraction,
    }
