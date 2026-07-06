# selected_blocks interface spec

## Shape
1. selected_blocks : int32 tensor
2. shape: [batch, layer, kv_head_group, query_block, K_selected_blocks]

## Rules for every row (selected_blocks[b, l, g, q, :])
1. Causal: every block ID must be <= q (no attending to a future block).
2. Sorted: block IDs must be in strictly increasing order.
3. No duplicates: no block ID may appear twice in the same row.
4. Local window: the row must include a fixed set of blocks near q
   (defined by LOCAL_WINDOW below) unless they'd be non-causal, in which
   case they're simply absent (never padded with an invalid ID).
5. Padding: rows are fixed-width (K_selected_blocks). If a row has fewer
   real selections than K, remaining slots are filled with PAD_VALUE (-1).
   PAD_VALUE is not a real block and is skipped by rules 1-3.

## Constants
LOCAL_WINDOW = 1   # include query block itself, plus 1 block before it
PAD_VALUE = -1