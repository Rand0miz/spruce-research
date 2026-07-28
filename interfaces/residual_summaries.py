"""Residual-tree routing for compressed summaries of omitted key blocks.

The existing ``selected_blocks`` tensor remains the exact-token route.  This
module assigns stable, leaf-first IDs to a binary key tree and constructs the
maximal tree frontier covering every causal block omitted by that route.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from interfaces.validator import PAD_VALUE, validate_selected_blocks


@dataclass(frozen=True)
class ResidualTreeLayout:
    """Deterministic leaf-first binary-tree layout.

    ``level_counts[0]`` is the number of leaf blocks.  Each following level
    has ``ceil(previous / 2)`` nodes, matching ``selector.tree``'s odd-tail
    rollup.  Node IDs are ``level_offsets[level] + index``.
    """

    num_leaves: int
    level_counts: tuple[int, ...]
    level_offsets: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    levels: tuple[int, ...]
    indices: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.starts)

    @property
    def root_id(self) -> int:
        return self.level_offsets[-1]

    def node_id(self, level: int, index: int) -> int:
        if not 0 <= level < len(self.level_counts):
            raise IndexError(f"tree level {level} is out of range")
        if not 0 <= index < self.level_counts[level]:
            raise IndexError(f"node index {index} is out of range for level {level}")
        return self.level_offsets[level] + index

    def range(self, node_id: int) -> tuple[int, int]:
        if not 0 <= node_id < self.num_nodes:
            raise IndexError(f"tree node ID {node_id} is out of range")
        return self.starts[node_id], self.ends[node_id]

    def children(self, node_id: int) -> tuple[int, ...]:
        """Return existing children in the immediately lower level."""
        if not 0 <= node_id < self.num_nodes:
            raise IndexError(f"tree node ID {node_id} is out of range")
        level = self.levels[node_id]
        if level == 0:
            return ()
        index = self.indices[node_id]
        child_level = level - 1
        first = index * 2
        return tuple(
            self.node_id(child_level, child_index)
            for child_index in (first, first + 1)
            if child_index < self.level_counts[child_level]
        )


def build_residual_tree_layout(num_leaves: int) -> ResidualTreeLayout:
    """Build the stable tree-node ID and leaf-range table."""
    if num_leaves < 1:
        raise ValueError(f"num_leaves must be >= 1, got {num_leaves}")

    counts = [int(num_leaves)]
    while counts[-1] > 1:
        counts.append((counts[-1] + 1) // 2)

    offsets = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count

    starts: list[int] = []
    ends: list[int] = []
    levels: list[int] = []
    indices: list[int] = []
    previous_ranges: list[tuple[int, int]] = []
    for level, count in enumerate(counts):
        if level == 0:
            ranges = [(index, index + 1) for index in range(count)]
        else:
            ranges = []
            for index in range(count):
                child_start = index * 2
                child_end = min(child_start + 2, len(previous_ranges))
                ranges.append(
                    (previous_ranges[child_start][0], previous_ranges[child_end - 1][1])
                )
        for index, (start, end) in enumerate(ranges):
            starts.append(start)
            ends.append(end)
            levels.append(level)
            indices.append(index)
        previous_ranges = ranges

    return ResidualTreeLayout(
        num_leaves=int(num_leaves),
        level_counts=tuple(counts),
        level_offsets=tuple(offsets),
        starts=tuple(starts),
        ends=tuple(ends),
        levels=tuple(levels),
        indices=tuple(indices),
    )


def residual_frontier_for_row(
    selected_block_ids: Sequence[int] | Iterable[int],
    query_block: int,
    layout: ResidualTreeLayout,
) -> list[int]:
    """Return maximal nodes covering the omitted causal blocks in one row."""
    if not 0 <= query_block < layout.num_leaves:
        raise ValueError(
            f"query_block must be in [0, {layout.num_leaves}), got {query_block}"
        )
    selected = sorted({int(block_id) for block_id in selected_block_ids if block_id >= 0})
    if selected and (selected[0] < 0 or selected[-1] > query_block):
        raise ValueError("selected block IDs must be causal and non-negative")

    causal_end = query_block + 1
    frontier: list[int] = []

    def contains_selected(start: int, end: int) -> bool:
        position = bisect_left(selected, start)
        return position < len(selected) and selected[position] < end

    def visit(node_id: int) -> None:
        start, end = layout.range(node_id)
        if start >= causal_end:
            return
        if end <= causal_end and not contains_selected(start, end):
            frontier.append(node_id)
            return
        for child_id in layout.children(node_id):
            visit(child_id)

    visit(layout.root_id)
    # Stable interface rows are ID-sorted even though traversal is range-first.
    return sorted(frontier)


def build_residual_summary_nodes(
    selected_blocks: torch.Tensor,
    *,
    layout: ResidualTreeLayout | None = None,
    validate_selected: bool = True,
) -> torch.Tensor:
    """Build ``[B,L,G,Q,S]`` residual frontiers without a fixed budget.

    ``S`` is the largest complete frontier in this tensor.  Shorter rows use
    trailing ``-1`` padding; an all-exact route therefore has ``S == 0``.
    """
    if validate_selected:
        validate_selected_blocks(selected_blocks)
    if selected_blocks.dim() != 5:
        raise ValueError("selected_blocks must be [B,L,G,Q,K]")
    if selected_blocks.dtype != torch.int32:
        raise ValueError(f"selected_blocks must be int32, got {selected_blocks.dtype}")

    batch, layers, groups, query_blocks, _ = selected_blocks.shape
    if layout is None:
        layout = build_residual_tree_layout(query_blocks)
    elif layout.num_leaves != query_blocks:
        raise ValueError(
            f"tree has {layout.num_leaves} leaves but selected_blocks has "
            f"{query_blocks} query blocks"
        )
    selected_cpu = selected_blocks.detach().to(device="cpu")
    rows: list[list[int]] = []
    largest = 0
    for batch_index in range(batch):
        for layer_index in range(layers):
            for group_index in range(groups):
                for query_block in range(query_blocks):
                    selected = [
                        int(value)
                        for value in selected_cpu[
                            batch_index, layer_index, group_index, query_block
                        ].tolist()
                        if value != PAD_VALUE
                    ]
                    frontier = residual_frontier_for_row(
                        selected, query_block, layout
                    )
                    rows.append(frontier)
                    largest = max(largest, len(frontier))

    result = torch.full(
        (batch, layers, groups, query_blocks, largest),
        PAD_VALUE,
        dtype=torch.int32,
    )
    row_index = 0
    for batch_index in range(batch):
        for layer_index in range(layers):
            for group_index in range(groups):
                for query_block in range(query_blocks):
                    frontier = rows[row_index]
                    row_index += 1
                    if frontier:
                        result[
                            batch_index, layer_index, group_index, query_block, : len(frontier)
                        ] = torch.tensor(frontier, dtype=torch.int32)
    return result.to(device=selected_blocks.device)


def validate_residual_summary_nodes(
    selected_blocks: torch.Tensor,
    residual_summary_nodes: torch.Tensor,
    *,
    require_maximal: bool = True,
    validate_selected: bool = True,
) -> None:
    """Prove exact blocks plus residual nodes partition every causal prefix."""
    if validate_selected:
        validate_selected_blocks(selected_blocks)
    assert residual_summary_nodes.dim() == 5, (
        "Expected residual_summary_nodes [batch, layer, kv_head_group, "
        f"query_block, S], got {tuple(residual_summary_nodes.shape)}"
    )
    assert residual_summary_nodes.dtype == torch.int32, (
        f"Expected residual_summary_nodes int32, got {residual_summary_nodes.dtype}"
    )
    assert residual_summary_nodes.shape[:4] == selected_blocks.shape[:4], (
        "selected_blocks and residual_summary_nodes must have identical "
        f"[batch, layer, group, query] dimensions, got "
        f"{tuple(selected_blocks.shape[:4])} and "
        f"{tuple(residual_summary_nodes.shape[:4])}"
    )

    batch, layers, groups, query_blocks, _ = selected_blocks.shape
    layout = build_residual_tree_layout(query_blocks)
    selected_cpu = selected_blocks.detach().to(device="cpu")
    residual_cpu = residual_summary_nodes.detach().to(device="cpu")
    for batch_index in range(batch):
        for layer_index in range(layers):
            for group_index in range(groups):
                for query_block in range(query_blocks):
                    location = (
                        f"b={batch_index}, l={layer_index}, g={group_index}, "
                        f"q={query_block}"
                    )
                    exact = [
                        int(value)
                        for value in selected_cpu[
                            batch_index, layer_index, group_index, query_block
                        ].tolist()
                        if value != PAD_VALUE
                    ]
                    padded_row = [
                        int(value)
                        for value in residual_cpu[
                            batch_index, layer_index, group_index, query_block
                        ].tolist()
                    ]
                    residual = [value for value in padded_row if value != PAD_VALUE]
                    if PAD_VALUE in padded_row:
                        first_pad = padded_row.index(PAD_VALUE)
                        assert all(value == PAD_VALUE for value in padded_row[first_pad:]), (
                            f"Residual padding is not trailing-only at [{location}]"
                        )
                    assert residual == sorted(residual), (
                        f"Residual node IDs are not sorted at [{location}]: {residual}"
                    )
                    assert len(residual) == len(set(residual)), (
                        f"Duplicate residual node IDs at [{location}]: {residual}"
                    )

                    coverage = [0] * (query_block + 1)
                    for block_id in exact:
                        assert 0 <= block_id <= query_block, (
                            f"Non-causal exact block {block_id} at [{location}]"
                        )
                        coverage[block_id] += 1
                    for node_id in residual:
                        assert 0 <= node_id < layout.num_nodes, (
                            f"Unknown residual node ID {node_id} at [{location}]"
                        )
                        start, end = layout.range(node_id)
                        assert end <= query_block + 1, (
                            f"Residual node {node_id} range [{start}, {end}) is "
                            f"partially or wholly future at [{location}]"
                        )
                        for block_id in range(start, end):
                            coverage[block_id] += 1
                    assert all(count == 1 for count in coverage), (
                        f"Exact and residual routes do not form a disjoint complete "
                        f"causal partition at [{location}]: coverage={coverage}"
                    )
                    if require_maximal:
                        expected = residual_frontier_for_row(
                            exact, query_block, layout
                        )
                        assert residual == expected, (
                            f"Residual frontier is complete but not maximal at "
                            f"[{location}]: expected {expected}, got {residual}"
                        )
