# Paper Notes — verified from source

Status: all four verified.

---

## HiP (arXiv 2406.09827) — verified

**Mechanism.** Two stages: mask estimation, then sparse attention over that mask. Neither requires training. Mask estimation is a greedy binary tree search per query: divide the key sequence into k ranges ("nodes"); each iteration splits every retained node in half, takes the first key of each branch as its representative, scores the query against those representatives, and keeps the global top-k branches. Repeats until each range is length 1 — log₂T iterations. Survivors are the estimated top-k keys.

**Complexity.** Mask estimation O(T log T) (log₂T iterations × constant work × T queries). Sparse read O(kT) = O(T) for fixed k. Space O(T).

**Frozen vs. trained.** Fully training-free. Reuses pretrained attention scores to locate top-k; no gate is learned. This is the gap you target — HiP's selector is greedy, not learned.

**Eval.** Llama2, Qwen1.5. WikiText2 (perplexity), MMLU, LongBench, BookSum. Up to 36.9× attention decode speedup vs FlashAttention2, 3.3× end-to-end vs PagedAttention, quality roughly maintained.

**Two confirmed caveats that feed your novelty:**
- HiP keeps some dense (unpruned) layers for quality in its main config; only removes that via an appendix ensemble method. Your baseline may include dense layers — note it.
- Confirms the tensor-core problem: per-query tree search scores a different key matrix per query, so it can't use tensor cores naturally; HiP adds block approximation to fix this. Direct evidence that the naive tree is not automatically fast — i.e. your kernel work is a real contribution.

**Not yet confirmed.** The late-level reliability caveat (greedy tree prunes the true target when sibling scores are close) is a real property of greedy top-k tree search and follows from representative-token scoring, but I did not see HiP explicitly state it in the sections read. Do not attribute the admission to HiP without reading its Limitations section.

---

## SeerAttention-R (arXiv 2506.08889) — verified

**Mechanism.** A lightweight learnable "AttnGate" added to a pretrained model predicts which attention blocks matter. In the gate, K tensors are pooled along the sequence dimension per block, then passed through two added linear layers (the only learnable params). It is a flat block scorer — scores blocks to decide activation. This is the "flat learned gate" baseline in your plan, not a hierarchical selector.

**Frozen vs. trained.** Original model parameters untouched — gate is a plug-in, no change to base weights. Only the AttnGate linear layers train. Signal is self-distillation: gate imitates the full model's own attention (pooled attention mass over blocks is the target). Trained on just 0.4B tokens.

**Complexity / scope — correction to the task note.** SeerAttention-R targets the **decoding** phase, not prefill. It extends the original SeerAttention (which handled prefill) by removing query pooling to accommodate auto-regressive decoding. So it is the decode-side recipe. Relevance to your prefill-first plan: use the *original* SeerAttention for the prefill flat-gate baseline; SeerAttention-R matters if/when you extend to decode. The self-distillation gate design carries across both.

**"GQA-aligned shared sparsity."** Confirmed in direction: removing query pooling is the decode adaptation, and the gate shares sparsity decisions across heads in a group. Exact per-head-group output granularity not fully verified from sources read — confirm against the paper's method section before you rely on the precise shape.

**Eval.** Near-lossless reasoning accuracy at 4K token budget on AIME under large block sizes (64/128). TileLang sparse-decode kernel, up to 9× over FlashAttention-3 on H100 at 90% sparsity. Code: github.com/microsoft/SeerAttention.

---

## SpotAttention (arXiv 2606.22874) — verified

**Mechanism.** A lightweight selector attaches to a frozen pretrained transformer and learns by KL distillation to estimate the full model's attention distribution. It picks the top-K keys per query. Because the estimate is a calibrated probability distribution (not just a score), a dual top-p rule reads a variable per-query, per-layer budget directly from it, with sink and recency blocks reserved separately so they don't dominate. No separate pruning stage.

**Confirmed: it does not avoid the full-prefix scan.** The selector attaches to every full-attention layer and is trained alone by KL distillation against dense attention. This is a flat, per-query scorer — same category as SeerAttention and your task's baseline description. It's the flat-scan baseline your hierarchical selector needs to beat, not a sub-quadratic method itself.

**Frozen vs. trained.** Backbone fully frozen. Only the selector is trained, via KL distillation (matching the full probability distribution, not just regressing to pooled attention mass like SeerAttention).

**Complexity.** Selector runs as a single tensor-core matmul — hardware-friendly, but per-query cost still scales with prefix length (flat scan). Same big-O category as SeerAttention: O(L) scoring per query block, O(L²) across prefill.

**Eval.** Five backbones across two families: Qwen3 (dense attention) and Qwen3.5 (hybrid linear/full), 4B-32B. Matches dense accuracy at contexts up to 128K, 8x the training length. Decode at 128K: 3.9x faster than FlashAttention, 1.8x faster than Twilight (strongest training-free baseline). K-cache quantization to INT4/FP4 gives 3.5x more shrinkage at no accuracy cost.

**Relevance to your plan.** Confirms the flat-scan bottleneck is a live, current problem others are still solving with flat methods — strengthens your motivation for a hierarchical selector. Also raises the baseline bar: SpotAttention's dual top-p dynamic budget is stronger than a plain fixed top-K flat gate, so your comparison table should note whether you're beating a fixed-budget flat gate (easier) or a calibrated dynamic-budget one like SpotAttention (harder, more current).

---

## HISA (arXiv 2603.28458) — verified

**Mechanism.** Starts from the same problem SpotAttention and SeerAttention share: a lightweight indexer (as in DeepSeek Sparse Attention) still scores every historical key for every query, which is itself O(L²) per layer even though the downstream sparse attention is cheap. HISA replaces that flat indexer with a two-stage hierarchical search: (1) block-level coarse filtering — score pooled block representations, discard irrelevant regions; (2) token-level refinement — apply the original fine-grained indexer only within the retained candidate blocks. Plug-and-play: it rewrites the search path without changing the final sparse attention pattern.

**Confirmed: exactly the two-stage block-then-token cost your task expected.** Coarse stage cost is proportional to the number of blocks (not tokens); refinement stage cost is proportional to tokens inside retained blocks only. This is a genuine reduction from the flat O(L²) indexer bottleneck, but it is two-stage, not recursive/tree-based — one coarse pass, one fine pass, not log-depth traversal. This confirms your read: HISA is a strong systems baseline and an important implementation reference, but a two-level filter is not automatically your O(n log n) recursive-tree claim.

**Frozen vs. trained.** Training-free at inference — a drop-in replacement for the indexer, no retraining of the indexer or backbone required. The authors note as future work that jointly training the block-scoring stage could improve coarse-filter accuracy, particularly at semantic-boundary cases — currently untrained.

**Complexity.** Reduces the indexer from O(L²) per layer to sub-quadratic scaling by cutting the search space at the coarse stage before the expensive fine-grained scoring runs. Reported as replacing a flat full-prefix scan; the exact asymptotic form is two-stage rather than log-depth (consistent with your prior algebra: O(n²/B + nrB), not O(n log n)).

**Known limitation, stated by the authors.** The coarse stage represents each block with a single pooled vector, which can fail when a block crosses a semantic boundary and the pooled representation doesn't reflect the block's most important token. They suggest overlapping blocks, adaptive boundaries, or max-pooling as mitigations — none implemented in the base method.

**Eval.** Built on DeepSeek Sparse Attention (DSA) as the underlying indexer being replaced. Evaluated for indexing cost reduction and selection fidelity preservation relative to the flat DSA indexer.

**Relevance to your plan.** Confirms your framing exactly: HISA is the systems baseline showing a two-stage filter beats a flat scan in real terms, but it is not the recursive/log-depth structure your OpenHiP-style claim needs. Its stated boundary-crossing weakness (single pooled vector per block) is a concrete failure mode your tree-based, multi-level approach could be measured against directly.

---

## Summary — how the four map onto your plan

| Paper | Role in your plan | Selector shape | Frozen backbone? |
|---|---|---|---|
| HiP | Structural template (tree search) | Recursive, O(T log T), training-free | Yes |
| SeerAttention(-R) | Training recipe (self-distilled gate) | Flat, O(L) per query | Yes, gate only trained |
| SpotAttention | Flat-gate baseline, current SOTA | Flat, KL-distilled, dynamic budget | Yes, selector only trained |
| HISA | Systems baseline (two-stage, not recursive) | Two-stage coarse+fine, not log-depth | Yes, training-free |

Your recursive, learned selector sits in the gap none of these four fill: HiP is recursive but not learned; SeerAttention/SpotAttention are learned but flat; HISA is two-stage but not recursive. That gap is the paper.