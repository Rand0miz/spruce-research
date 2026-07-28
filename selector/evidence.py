"""Reader-row evidence ranking shared by compact-context diagnostics."""
from collections.abc import Iterable

import torch


def rank_reader_candidate_blocks(
        reader_scores: torch.Tensor, top_m: int,
        allowed_blocks: Iterable[int], *,
        block_ids: Iterable[int] | None = None) -> tuple[list[int], list[float]]:
    """Max-pool layer/group scores and rank only source-document blocks.

    ``reader_scores`` may be [L,G,1,K] or [L,G,K]. Returned block IDs are in
    descending score order; compilation later restores source order.
    """
    if int(top_m) < 1:
        raise ValueError(f"top_m must be >= 1, got {top_m}")
    if reader_scores.dim() == 4:
        if reader_scores.shape[2] != 1:
            raise ValueError(
                "4-D reader_scores must have one reader query row")
        reader_scores = reader_scores[:, :, 0, :]
    if reader_scores.dim() != 3:
        raise ValueError(
            f"reader_scores must be [L,G,K] or [L,G,1,K], got "
            f"{tuple(reader_scores.shape)}")
    if not bool(torch.isfinite(reader_scores).all()):
        raise ValueError("reader_scores contains non-finite values")

    key_blocks = reader_scores.shape[-1]
    if block_ids is None:
        source_ids = list(range(key_blocks))
    else:
        source_ids = [int(block) for block in block_ids]
        if len(source_ids) != key_blocks:
            raise ValueError(
                "block_ids length must match reader_scores key width")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("block_ids must be duplicate-free")
    allowed = sorted(set(int(block) for block in allowed_blocks))
    if not allowed:
        raise ValueError("allowed_blocks must contain at least one block")
    allowed_set = set(allowed)
    positions = [
        position for position, source_id in enumerate(source_ids)
        if source_id in allowed_set
    ]
    if not positions:
        raise ValueError(
            "none of the scored block_ids overlap allowed_blocks")

    pooled = reader_scores.amax(dim=(0, 1))
    position_tensor = torch.tensor(
        positions, dtype=torch.long, device=pooled.device)
    allowed_scores = pooled.index_select(0, position_tensor)
    width = min(int(top_m), len(positions))
    values, indices = allowed_scores.topk(width)
    chosen_positions = position_tensor.index_select(0, indices)
    chosen_blocks = [
        source_ids[int(position)]
        for position in chosen_positions.detach().cpu().tolist()
    ]
    return (
        chosen_blocks,
        [float(value) for value in values.detach().cpu().tolist()],
    )
