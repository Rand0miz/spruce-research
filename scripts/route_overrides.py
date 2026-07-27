"""Route post-processing for diagnostic controls.

Both helpers produce validator-compliant ``selected_blocks`` tensors. They are
diagnostic tools: ``force_needle_routes`` builds the oracle-injection control
(is routed evidence SUFFICIENT for correct sparse generation?) and
``teacher_topk_routes`` builds the label-ceiling control (can routes distilled
straight from teacher mass support generation at all?).
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.export_selected_blocks import selected_ids_to_blocks

PAD_VALUE = -1


def charged_route_density(selected, dense_layers=()):
    """Report block-attention density with dense layers charged honestly.

    The denominator is every causal block pair in every layer/KV group. Sparse
    layers contribute the number of non-PAD selected entries; layers dispatched
    to dense SDPA contribute their complete causal triangle.
    """
    if selected.dim() != 5:
        raise ValueError(
            f"selected must be [B,L,G,qb,K], got {tuple(selected.shape)}")
    B, L, G, qb, _ = selected.shape
    layers = sorted({int(layer) for layer in dense_layers})
    if any(layer < 0 or layer >= L for layer in layers):
        raise ValueError(f"dense layer outside [0,{L}): {layers}")
    dense_set = set(layers)
    causal_per_layer = B * G * qb * (qb + 1) // 2
    charged_entries = 0
    for layer in range(L):
        if layer in dense_set:
            charged_entries += causal_per_layer
        else:
            charged_entries += int((selected[:, layer] >= 0).sum().item())
    maximum_entries = causal_per_layer * L
    fraction = charged_entries / maximum_entries if maximum_entries else 0.0
    return {
        "dense_layers": layers,
        "dense_layer_count": len(layers),
        "dense_layer_fraction": len(layers) / L,
        "charged_block_entries": charged_entries,
        "maximum_causal_block_entries": maximum_entries,
        "charged_attention_fraction": fraction,
        "charged_sparsity": 1.0 - fraction,
    }


def force_needle_routes(selected, needle_block, pad_value=PAD_VALUE):
    """Widen ``selected`` [B,L,G,qb,K] by one slot and guarantee ``needle_block``
    in every causally eligible row (query_block >= needle_block)."""
    if selected.dim() != 5:
        raise ValueError(
            f"selected must be [B,L,G,qb,K], got {tuple(selected.shape)}")
    needle_block = int(needle_block)
    B, L, G, qb, K = selected.shape
    if not 0 <= needle_block < qb:
        raise ValueError(f"needle_block {needle_block} outside [0,{qb})")

    device = selected.device
    out = torch.full((B, L, G, qb, K + 1), pad_value,
                     dtype=selected.dtype, device=device)
    out[..., :K] = selected

    query = torch.arange(qb, device=device)
    causal = query >= needle_block                              # [qb]
    present = (selected == needle_block).any(dim=-1)            # [B,L,G,qb]
    inject = causal[None, None, None, :] & ~present
    out[..., K] = torch.where(
        inject,
        torch.full_like(out[..., K], needle_block),
        torch.full_like(out[..., K], pad_value))

    # Re-sort ascending with PAD trailing: map PAD to a sentinel above any id.
    sentinel = qb + 1
    keyed = out.masked_fill(out == pad_value, sentinel)
    keyed, _ = keyed.sort(dim=-1)
    return keyed.masked_fill(keyed == sentinel, pad_value).to(selected.dtype)


def dense_reader_routes(selected, pad_value=PAD_VALUE):
    """Give the final (reader/question) query row EVERY causal block.

    All other rows keep their learned routes, PAD-extended to the new width.
    Separates 'reader-row budget too small' from 'document representations
    corrupted by sparsity elsewhere': if generation still fails with a dense
    reader row, access at the reader row was never the binding constraint.
    """
    if selected.dim() != 5:
        raise ValueError(
            f"selected must be [B,L,G,qb,K], got {tuple(selected.shape)}")
    B, L, G, qb, K = selected.shape
    kb = qb
    width = max(K, kb)
    device = selected.device
    out = torch.full((B, L, G, qb, width), pad_value,
                     dtype=selected.dtype, device=device)
    out[..., :K] = selected
    out[:, :, :, qb - 1, :] = torch.arange(
        kb, device=device, dtype=selected.dtype)
    return out


def dense_evidence_routes(selected, needle_block, pad_value=PAD_VALUE,
                          neighborhood=0):
    """Dense reader row + dense evidence-block query row + evidence everywhere.

    Ceiling diagnostic for the 32K failure: if the evidence block's own tokens
    were contextualized densely (their query row attends every causal block)
    AND the reader row is dense AND every row can reach the evidence, does
    generation recover? ``neighborhood`` additionally densifies the query rows
    of the blocks within +-N of the evidence (semantic-boundary straddle
    control). Uses oracle knowledge of the needle — not deployable, purely
    attribution.
    """
    widened = force_needle_routes(selected, needle_block, pad_value=pad_value)
    out = dense_reader_routes(widened, pad_value=pad_value)
    B, L, G, qb, width = out.shape
    needle_block = int(needle_block)
    dense_row = torch.arange(qb, device=out.device, dtype=out.dtype)
    low = max(0, needle_block - int(neighborhood))
    high = min(qb - 1, needle_block + int(neighborhood))
    for block in range(low, high + 1):
        causal_width = block + 1
        row = torch.full((width,), pad_value, dtype=out.dtype,
                         device=out.device)
        row[:causal_width] = dense_row[:causal_width]
        out[:, :, :, block, :] = row
    return out


def candidate_span(candidate_blocks, qb, neighborhood=0):
    """Blocks actually densified for ``candidate_blocks`` at a given radius.

    Each candidate expands to the clamped window [b-w, b+w]. Overlapping
    windows collapse, so the count is NOT len(candidates)*(2w+1) — read the
    real densified-row cost off this list rather than computing it.
    """
    if int(neighborhood) < 0:
        raise ValueError(f"neighborhood must be >= 0, got {neighborhood}")
    span = set()
    for block in candidate_blocks:
        block = int(block)
        if not 0 <= block < qb:
            raise ValueError(f"candidate block {block} outside [0,{qb})")
        low = max(0, block - int(neighborhood))
        high = min(qb - 1, block + int(neighborhood))
        span.update(range(low, high + 1))
    return sorted(span)


def dense_candidate_routes(selected, candidate_blocks, pad_value=PAD_VALUE,
                           neighborhood=0):
    """Deployable retrieve-then-re-encode routing: no oracle knowledge.

    Dense reader row plus dense query rows for ``candidate_blocks`` — the
    gate's own top-scoring reader-row blocks. Repairs the K/V of likely
    evidence regions (the 32K corruption mechanism) at O(L) per densified row,
    ~(M+1)/qb of dense prefill extra. Everything else keeps its learned routes.

    ``neighborhood`` densifies +-N blocks around each candidate instead of the
    candidate alone. This separates two readings of the inverted M curve (M=4
    beats M=32): whether repair must be CONTIGUOUS around the evidence, or
    whether scattered high-scoring blocks are equally good per row spent. Same
    knob as ``dense_evidence_routes``, but driven by gate scores rather than
    needle metadata, so it stays deployable.
    """
    out = dense_reader_routes(selected, pad_value=pad_value)
    B, L, G, qb, width = out.shape
    dense_row = torch.arange(qb, device=out.device, dtype=out.dtype)
    for block in candidate_span(candidate_blocks, qb, neighborhood):
        causal_width = block + 1
        row = torch.full((width,), pad_value, dtype=out.dtype,
                         device=out.device)
        row[:causal_width] = dense_row[:causal_width]
        out[:, :, :, block, :] = row
    return out


def teacher_topk_routes(target, k_selected, local_window=1):
    """Pack the teacher's top-``k_selected`` blocks per row into routes.

    ``target`` is the [L,G,qb,kb] teacher marginal from
    ``selector.targets.load_teacher``. Local-window policy and PAD handling
    come from ``selected_ids_to_blocks`` unchanged.
    """
    if target.dim() != 4:
        raise ValueError(f"target must be [L,G,qb,kb], got {tuple(target.shape)}")
    kb = target.shape[-1]
    # The packer prioritizes forced local blocks and then keeps the SMALLEST
    # candidate ids, not the highest-mass ones. Hand it exactly the number of
    # free nonlocal slots so no high-mass teacher block is dropped by id order.
    free = max(int(k_selected) - (int(local_window) + 1), 1)
    width = min(free, kb)
    ids = target.topk(width, dim=-1).indices                    # [L,G,qb,width]
    return selected_ids_to_blocks(
        ids, k_selected=int(k_selected), local_window=int(local_window),
        key_blocks=kb)


def teacher_dual_top_p_routes(
        target, top_p, block_size=64, sink_tokens=128,
        recency_tokens=256, k_min_tokens=512, pad_value=PAD_VALUE,
        unify_groups=True):
    """Build an optimistic SpotAttention-inspired prefill oracle.

    SpotAttention's published long-context accuracy path keeps prefill dense
    and applies sparse selection at decode. This diagnostic deliberately
    adapts its dual-top-p rule to every prefill query block, using the exact
    dense teacher marginal instead of a learned selector. It is therefore a
    construction ceiling, not a deployable SpotAttention reproduction.

    ``target`` is the row-normalized dense teacher marginal [L,G,qb,kb].
    SpotAttention predicts one head-averaged distribution shared by the
    attention heads, so ``unify_groups=True`` averages SPRUCE's equal-sized
    GQA groups and repeats that route back across the selected-block group
    axis.
    Token-sized paper defaults are converted to this run's block size:
    a reserved absolute sink prefix, a reserved causal recency suffix, and a
    minimum total selected-token floor. Nucleus selection runs only over the
    residual distribution. Output is validator-compatible
    [1,L,G,qb,K_max], with variable row budgets padded by ``pad_value``.
    """
    if target.dim() != 4:
        raise ValueError(
            f"target must be [L,G,qb,kb], got {tuple(target.shape)}")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0,1], got {top_p}")
    if int(block_size) < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    for name, value in (
            ("sink_tokens", sink_tokens),
            ("recency_tokens", recency_tokens),
            ("k_min_tokens", k_min_tokens)):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains non-finite values")
    if bool((target < 0).any()):
        raise ValueError("target contains negative mass")

    L, G, qb, kb = target.shape
    if unify_groups:
        target = target.mean(dim=1, keepdim=True).expand(L, G, qb, kb)
    if qb != kb:
        raise ValueError(
            "prefill dual-top-p currently requires equal query/key blocks, "
            f"got qb={qb}, kb={kb}")
    device = target.device
    sink_blocks = min(
        kb, (int(sink_tokens) + int(block_size) - 1) // int(block_size))
    recency_blocks = min(
        kb, (int(recency_tokens) + int(block_size) - 1) // int(block_size))
    k_min_blocks = min(
        kb, (int(k_min_tokens) + int(block_size) - 1) // int(block_size))

    query_ids = torch.arange(qb, device=device)[:, None]
    key_ids = torch.arange(kb, device=device)[None, :]
    causal = key_ids <= query_ids
    sink = (key_ids < sink_blocks) & causal
    recency_start = (query_ids - recency_blocks + 1).clamp_min(0)
    recency = (key_ids >= recency_start) & causal if recency_blocks else (
        torch.zeros_like(causal))
    reserved = sink | recency
    residual = causal & ~reserved

    expanded_residual = residual[None, None].expand(L, G, qb, kb)
    negative = torch.full_like(target, -1.0)
    residual_scores = torch.where(expanded_residual, target, negative)
    order = residual_scores.argsort(dim=-1, descending=True)
    ordered_mass = target.gather(-1, order)
    ordered_valid = expanded_residual.gather(-1, order)
    ordered_mass = torch.where(
        ordered_valid, ordered_mass, torch.zeros_like(ordered_mass))

    residual_count = ordered_valid.sum(dim=-1)
    residual_mass = ordered_mass.sum(dim=-1)
    cumulative = ordered_mass.cumsum(dim=-1)
    threshold = residual_mass * float(top_p)
    nucleus_count = (cumulative < threshold[..., None]).sum(dim=-1) + 1
    nucleus_count = torch.where(
        residual_count > 0, nucleus_count, torch.zeros_like(nucleus_count))
    reserved_count = reserved.sum(dim=-1)[None, None].expand(L, G, qb)
    floor_count = (k_min_blocks - reserved_count).clamp_min(0)
    selected_residual_count = torch.maximum(
        nucleus_count, floor_count).minimum(residual_count)

    ranks = torch.arange(kb, device=device)
    prefix = ranks < selected_residual_count[..., None]
    residual_selected = torch.zeros(
        (L, G, qb, kb), dtype=torch.bool, device=device)
    residual_selected.scatter_(-1, order, prefix)
    selected_mask = (
        reserved[None, None].expand(L, G, qb, kb) | residual_selected)
    selected_mask &= causal[None, None]

    row_counts = selected_mask.sum(dim=-1)
    width = max(1, int(row_counts.max().item()))
    ids = torch.arange(kb, device=device, dtype=torch.int64)
    ids = ids.view(1, 1, 1, kb).expand(L, G, qb, kb)
    sentinel = kb + 1
    packed = ids.masked_fill(~selected_mask, sentinel).sort(dim=-1).values
    packed = packed[..., :width]
    packed = packed.masked_fill(packed == sentinel, int(pad_value))
    return packed.to(torch.int32).unsqueeze(0)
