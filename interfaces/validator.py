import torch

PAD_VALUE = -1

def validate_selected_blocks(t: torch.Tensor) -> None:
    """
    Checks selected_blocks against the spec. Raises AssertionError on the
    first violation found, with a message identifying exactly where.
    Shape: [batch, layer, kv_head_group, query_block, K_selected_blocks]
    """
    assert t.dim() == 5, (
        f"Expected 5D tensor [batch, layer, kv_head_group, query_block, "
        f"K_selected_blocks], got {t.dim()}D with shape {tuple(t.shape)}"
    )
    assert t.dtype == torch.int32, f"Expected int32, got {t.dtype}"

    batch, layer, group, qblocks, k = t.shape

    for b in range(batch):
        for l in range(layer):
            for g in range(group):
                for q in range(qblocks):
                    row = t[b, l, g, q].tolist()
                    real_ids = [x for x in row if x != PAD_VALUE]

                    # Rule: causal (no block ID beyond the query block itself)
                    for block_id in real_ids:
                        assert block_id <= q, (
                            f"Causal violation at [b={b}, l={l}, g={g}, q={q}]: "
                            f"block {block_id} is in the future (q={q})"
                        )
                    
                    # Rule: sorted, strictly increasing
                    assert real_ids == sorted(real_ids), (
                        f"Not sorted at [b={b}, l={l}, g={g}, q={q}]: {real_ids}"
                    )

                    # Rule: no duplicates (sorted+unique check combined)
                    assert len(real_ids) == len(set(real_ids)), (
                        f"Duplicate block IDs at [b={b}, l={l}, g={g}, q={q}]: {real_ids}"
                    )