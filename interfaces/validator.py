import torch

PAD_VALUE = -1
LOCAL_WINDOW = 1


def validate_selected_blocks(t: torch.Tensor, local_window: int = LOCAL_WINDOW) -> None:
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
    assert local_window >= 0, f"local_window must be >= 0, got {local_window}"

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

                    # Rule: local-window blocks near q must be present unless non-causal.
                    local_start = max(0, q - local_window)
                    required = list(range(local_start, q + 1))
                    missing = [block_id for block_id in required if block_id not in real_ids]
                    assert not missing, (
                        f"Missing local-window block(s) at [b={b}, l={l}, g={g}, q={q}]: "
                        f"missing {missing}, row={real_ids}"
                    )

                    # Rule: fixed-width rows use PAD_VALUE only after all real IDs.
                    if PAD_VALUE in row:
                        first_pad = row.index(PAD_VALUE)
                        trailing = row[first_pad:]
                        assert all(x == PAD_VALUE for x in trailing), (
                            f"Padding is not trailing-only at [b={b}, l={l}, g={g}, q={q}]: "
                            f"{row}"
                        )


def validate_residual_summary_nodes(*args, **kwargs) -> None:
    """Lazy compatibility export for the residual partition validator."""
    from interfaces.residual_summaries import (
        validate_residual_summary_nodes as _validate,
    )

    _validate(*args, **kwargs)
