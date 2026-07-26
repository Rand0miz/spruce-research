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


def dense_candidate_routes(selected, candidate_blocks, pad_value=PAD_VALUE):
    """Deployable retrieve-then-re-encode routing: no oracle knowledge.

    Dense reader row plus dense query rows for ``candidate_blocks`` — the
    gate's own top-scoring reader-row blocks. Repairs the K/V of likely
    evidence regions (the 32K corruption mechanism) at O(L) per densified row,
    ~(M+1)/qb of dense prefill extra. Everything else keeps its learned routes.
    """
    out = dense_reader_routes(selected, pad_value=pad_value)
    B, L, G, qb, width = out.shape
    dense_row = torch.arange(qb, device=out.device, dtype=out.dtype)
    for block in candidate_blocks:
        block = int(block)
        if not 0 <= block < qb:
            raise ValueError(f"candidate block {block} outside [0,{qb})")
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
