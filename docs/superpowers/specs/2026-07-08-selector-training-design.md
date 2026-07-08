# Stage 2 — Selector Training Design

**Date:** 2026-07-08
**Scope:** Train the relevance-scoring selector to imitate the frozen teacher's
pooled attention. Prefill-only. Backbone stays frozen; only the selector learns.
Prototype on Qwen2.5-Coder-1.5B-Instruct at 16K–32K on the laptop.

---

## 1. What is trained

The tree is fixed structure (index math, no parameters). The **selector** is the
only thing that learns: it maps `(pooled query-block, pooled key-node)` to a
relevance score. The base model is frozen — no gradients into Qwen.

Architecture (SeerAttention form, the "single pooled vector" start):

```
score(q_block, k_node) = softmax_over_keys( (Wq · pooledQ[q]) · (Wk · pooledK[k])^T / sqrt(d) )
```

`Wq`, `Wk` are the only learnable parameters, **per layer** (each Qwen layer gets
its own gate, matching SeerAttention). SpotAttention's multi-head-ReLU-with-learned-
mixing selector is a later upgrade, not this step.

## 2. Verified loss form (against the papers, not memory)

Both anchor papers use **forward KL with the teacher as the reference distribution**:

- SeerAttention (2410.13276): `loss = D_KL(gt || score)`, where
  `gt = MaxPool2D(softmax(QKᵀ/√d))`, `score = AttnGate(Q,K)`.
- SpotAttention (2606.22874): `L_KL^Dense = (1/BL) Σ_{b,q} Σ_t pᵗ log(pᵗ/pˢ)`,
  teacher `pᵗ = (1/H_q) Σ_h softmax_t(s_t[h,q,·])`, student `pˢ = softmax_t(s_s[q,·])`.

**Neither paper is hierarchical.** Both softmax over the *entire* candidate set
(all key blocks / all keys) at a single granularity. They fix the *per-level* loss;
they do **not** dictate a tree decomposition. The recursive tree is our extension.

**Our per-level loss (paper-faithful):**
`loss_level = D_KL( pᵗ_level || softmax(student_scores_level) )`, forward KL,
teacher-normalized, summed over rows and layers, causal-masked.

## 3. Teacher target — already the right thing

`teacher/chunked_extract.chunked_pool` produces **sum-pooled softmax mass**
`[1, L, H, qb, kb]`. Per query-block row, normalizing that mass to sum 1 gives the
**block-marginal attention probability** — identical to SpotAttention's `pᵗ`
(softmax-then-sum-per-block == per-block marginal). SeerAttention's MaxPool2D target
is looser; we keep the marginal (better). No new extraction of the *target* needed.

Normalization and GQA aggregation live in `selector/targets.py`:
- Row-normalize teacher mass to a distribution over key-blocks (`pᵗ`).
- Aggregate the H query-heads into `kv_head_group` groups (Qwen2.5-Coder-1.5B: 12
  query heads → 2 kv groups) by averaging the per-head mass within each group, so the
  target matches the `selected_blocks` `kv_head_group` axis.

## 4. The gap this design closes: selector inputs

The saved `.pt` holds only the *target* (attention mass), not the *inputs* the
selector reads. The selector needs **pooled Q per query-block** and **pooled K per
key-block**. Decision (locked): **dump them offline, once**, by extending
`teacher/chunked_extract.py` to also mean-pool the post-RoPE Q and K it already has
in hand and save:

- `pooledQ [L, H, qb, d]`
- `pooledK [L, kv, kb, d]`

Training then reads tensors only — no Qwen in the loop, cheap multi-epoch on 8 GB.

**Known deviations (log, not blockers):**
1. We pool **post-RoPE** Q/K. SeerAttention rescales the RoPE angle to block level
   (θ' = θ/B) before pooling to avoid intra-block rotation washing the signal out.
   Flag; revisit only if leaf recall is low.
2. GQA: average the 12 query-head masses into 2 kv-groups for the target (§3).

## 5. Build order (de-risked)

CLAUDE.md mandates "flat gate first, then recursive tree." The flat gate **is the
leaf level of the tree**, so this is not throwaway work.

**Step 1 — Flat gate.** Score every key-block directly (leaf level only). Forward-KL
against the row-normalized teacher marginal. This is the SeerAttention baseline and
exactly what KS1 measures. **Gate: must recover >95% of the teacher's block choices
(and dense retrieval quality) at 16K–32K before any tree work.** If it fails, fix
data/recipe — do not build the tree on a broken base.

**Step 2 — Tree.** Reuse `teacher/pool.rollup` (quadtree, parent = sum of 2×2
children, branch factor 2) and the `ks1_lite` ancestor math (`ancestor = block //
2^level`). Add coarser levels and sum the per-level loss. Open choice, decided at
step 2: **hierarchical conditional** `P(child|parent)` (matches inference descent —
recommended) vs **flat per-level** (paper-faithful but the descent never uses a
global level ranking).

## 6. Recall metric (separate from loss)

Loss falling ≠ ranking correct. `selector/recall.py` periodically checks: does the
selector's predicted top-r at each level contain the teacher's true top-r? Reuse the
`ks1_lite` hit@k and traversal-safe checks. This overlap is the real KS1 signal.

## 7. Data plan

Two distinct datasets — do not conflate:

| Purpose | Content | Count |
|---|---|---|
| Pipeline smoke test (now) | existing needle, depth 0.5 × {16K, 32K} | 2 (done) |
| Real training | **diverse real code documents** (it is a code model) | ~20–50 × {16K, 32K} |
| KS1 eval / recall | needle-in-haystack, **depth sweep** 0→1 | 5 depths × 2 len = 10 |

Rationale: needle prompts are repeated filler — teacher attention on them is
degenerate (diagonal + sink + needle block). A selector trained only on them learns
nothing transferable. Training needs *content diversity* (real code) more than
*volume*; one 29K doc already yields ~460 query-block rows × 28 layers of signal.
Needle prompts are for **eval only**, and there the depth sweep is required to expose
lost-in-the-middle (depth 0.5 alone hides it).

**Ops:** ~142 MB per 29K doc (fp16). 50 docs ≈ 7 GB — store fp16, weight toward 16K,
watch disk. Extract the real set *after* the training loop runs on the existing 2 —
never extract 50 docs before the loop works.

## 8. Module layout (separation of concerns)

```
teacher/chunked_extract.py   +dump pooled Q/K alongside mass (small add; re-run 2 lengths)
selector/targets.py         load .pt, GQA-aggregate H→kv-group, row-normalize → pᵗ
selector/gate.py            SeerAttention flat gate: per-layer Wq, Wk + block dot-product score
selector/loss.py            forward KL(pᵗ || student), per-row, causal-masked, summed over layers
selector/train.py           loop; Adam on Wq/Wk only; frozen backbone; loads dumped features + targets
selector/recall.py          top-r recall + traversal-safe (reuse ks1_lite logic)
selector/tree.py            STEP 2: reuse teacher/pool.rollup + per-level (conditional) loss
```

Selector code never reaches into kernel internals; the eventual hand-off to the
kernel is through `selected_blocks` only.

## 9. Checkpoint 2 (= Kill Switch 1)

Flat selector recovers **>95%** of the teacher's block choices and dense retrieval
quality at 16K–32K. Pass → add the tree (step 2). Fail → fix data/recipe, do not
scale a bad selector.
