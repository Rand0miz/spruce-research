# residual_summary_nodes interface spec

## Shape

1. `residual_summary_nodes`: `int32`
2. Shape: `[batch, layer, kv_head_group, query_block, S]`
3. `S` is the largest complete residual frontier in the tensor. It is not a
   truncation budget and may be zero when every causal block is exact.

## Tree-node IDs

The binary key tree uses the same odd-tail rollup as `selector.tree`.
Leaf blocks are level 0. Level `l + 1` contains `ceil(nodes_l / 2)` parents.
IDs are assigned leaf-first using the cumulative node counts as level offsets.
Each node covers a half-open interval of leaf block IDs.

## Rules for every row

1. Exact IDs remain exclusively in `selected_blocks`.
2. Residual node IDs are sorted, duplicate-free, and use trailing `-1` padding.
3. A residual node is wholly causal; partially causal nodes are split.
4. Residual nodes do not overlap each other or exact selected blocks.
5. Exact blocks plus residual nodes cover every block in the causal prefix
   exactly once.
6. The frontier is maximal: a parent is used whenever its complete range is
   causal and contains no exact block.

