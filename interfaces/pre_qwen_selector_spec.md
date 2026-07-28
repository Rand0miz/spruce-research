# Pre-Qwen hierarchical selector contract

## Purpose

The evidence compiler needs source locations before Qwen reads the compact
packet. This selector is deliberately independent of the Qwen backbone and the
learned attention selector: it consumes only the tokenizer IDs of the source
document and final question.

It is a retrieval front end for exact-text compilation. It is not a replacement
for `selected_blocks` inside the sparse-attention kernel and does not change
that frozen interface.

## Representation

For every 64-token source-document block:

- Hash tokenizer-ID unigrams into the first half of a fixed-width Boolean
  sketch.
- Hash adjacent tokenizer-ID pairs into the second half.
- Compute block document frequency for every bucket.
- Build a deterministic radix-2 tree by Boolean max/union over child sketches.
- Carry odd tail nodes into a smaller final parent without padding them into
  the source-block address space.
- Assign stable node IDs with leaf-first level offsets.

Boolean parent union is intentional. A rare evidence term remains present at
every ancestor instead of being attenuated by an average over unrelated text.

The locked development configuration is:

```
block_size = 64
feature_dim = 512
unigram_fraction = 0.5
radix = 2
```

## Query and traversal

The final question uses the same unigram/bigram hash. Query buckets are
weighted by squared block IDF, so document-specific phrases dominate generic
instruction words. Starting at the root, the selector scores child coverage
and retains a fixed beam at each level. The locked compiler configuration is:

```
top_m = 4
beam = 4
block_radius = 1
boundary = "paragraph"
```

Returned IDs are absolute source-prompt block IDs. The evidence compiler
restores source order, clips every span to the document, expands by one block,
repairs paragraph boundaries, and performs a fresh ordinary dense Qwen read.

## Complexity

- Prompt tokenization: O(n).
- Leaf sketch construction: O(n) feature insertions.
- Fixed-width tree construction: O(number_of_blocks * D), with fixed D=512.
- Query construction: O(question length).
- Traversal: O(beam * log n * D), with fixed beam and D.
- Dense read: quadratic only in the compact packet length, not the original
  document length.

The total pre-Qwen preprocessing remains O(n). No “sub-linear total cost” or
“linear selection” claim is made.

## Fully charged timing boundary

`benchmarks/benchmark_pre_qwen_e2e.py` starts both paths from prompt/question
strings already in memory.

Dense charges:

1. Full-prompt tokenization.
2. Host-to-device input transfer.
3. Dense prefill.
4. Dense decode.

Compiled charges:

1. Full-prompt tokenization with offsets.
2. Leaf and tree construction.
3. Question hashing and IDF weights.
4. Top-down traversal.
5. Exact-text expansion, boundary repair, and stitching.
6. Compact-prompt tokenization.
7. Host-to-device input transfer.
8. Dense prefill.
9. Dense decode.

Manifest reads, model/tokenizer loading, and one backend warm-up are excluded
from both paths. The index is rebuilt on every measured repeat; no cache is
assumed. The primary speed statistic is the ratio of summed per-case median
request times, not the median of per-case speedups.

The frozen 25-case deployability gate requires:

- At least 24/25 compiled exact overall.
- 11/11 compiled exact at 32K.
- Sum-weighted fully charged speedup greater than 1.0x.
