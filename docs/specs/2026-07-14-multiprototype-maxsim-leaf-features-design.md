# Multi-Prototype MaxSim Leaf Features — Design Spec

**Date:** 2026-07-14
**Status:** Approved, pre-implementation
**Scope:** Replace mean-pooled block features with multi-prototype (mean + outliers)
MaxSim features in teacher extraction and the flat selector gate, so a single-token
"needle" survives block summarization.

---

## 1. Problem

The flat gate scores blocks with one mean-pooled K vector and one mean-pooled Q vector
per 64-token block:

```
score(q,k) = (Wq · pooledQ[q]) · (Wk · pooledK[k])^T / sqrt(p)
```

Dense attention is a **max** over tokens (softmax is dominated by the single
best-matching token), but mean-pooling answers "what is this block about on average."
A retrieval-critical **needle** — one matching token among ~63 filler tokens — is
averaged away before the gate ever sees it.

**Evidence (1.5B held-out, block=64):**
- `coverage@8 / oracle = 0.738 / 0.740 = 99.7%` — gate nails bulk attention mass.
- `needle_hit@8 = 0.27` (15K) / `0.04` (30K) — gate drops the needle.
- Teacher-oracle `teacher_needle@8 = 0.73 / 0.84` — the teacher target *keeps* the needle,
  so the signal exists; the gate's inputs cannot represent it.
- Adding a top-k membership loss term (`lambda` 0.5 then 4.0) left **training-set**
  `needle_hit@8` stuck at 0.00–0.09. A model that cannot fit its own training target is
  feature-limited, not loss-limited. **Confirmed: features, not recipe.**

Root cause: mean-pooling destroys the single-token spike. No loss weight can teach what
the features do not contain.

## 2. Goal

Give the gate a block representation that preserves per-token spikes, so:
1. Training-set `needle_hit@8` climbs off ~0 (proves features now carry the needle).
2. Held-out `needle_hit@8` >> 0.04.
3. Bulk `coverage` stays near-oracle.

Accuracy is the priority (this is the foundation the tree selector will stand on), but
**inference selector cost is a hard constraint** — the whole paper claim is a cheap
O(log L) selector, so prototype count must stay small and bounded.

## 3. Key constraint: inference reproducibility

Prototype selection must run from Q/K vectors **alone**. At training we have the teacher's
attention, but the real sparse-prefill run does not. Selecting prototypes by "tokens the
teacher attended" is **not reproducible at inference** and is therefore rejected.
Selection uses only the block's own token vectors (mean + distance-from-mean).

## 4. Locked decisions

| Decision | Choice | Reason |
|---|---|---|
| Which sides get prototypes | **Both** query and key | Accuracy-max; catches a diluted query token as well as the key-side needle |
| Selection rule | **mean + (P-1) outliers** (farthest L2 from block mean) | Needle is by definition an outlier; mean covers bulk mass; reproducible at inference |
| Prototypes per block, P | **8** (1 mean + 7 outliers) | 8×8=64 dots/block-pair — bounded selector cost; strong odds the needle lands in 7 outlier slots |
| Scoring | **MaxSim** = max over prototype pairs | Mirrors attention's max-over-tokens behavior |

Rejected: **keep-all-64** — not just storage. Selector scores P_q×P_k pairs per
block-pair; 64×64 = 4096 dots inflates the exact O(log L) selector cost the paper
minimizes, and balloons prototype sets up the tree. Bounded P re-caps at each level.

## 5. Changes by file

Only feature production changes. Everything downstream of raw `scores [L,G,qb,kb]`
(`loss.py`, `recall.py`, `train.py`, `eval_gate.py`, the `selected_blocks` interface) is
**untouched** — the seam is the `scores` tensor, whose shape is unchanged.

### 5.1 `teacher/chunked_extract.py`

Replace `_mean_pool_seq(x, block)` with `_proto_pool_seq(x, block, P)`:
- Per block, emit **P vectors** along a new axis:
  - prototype 0 = block mean (over real rows only; ragged last block averages real rows,
    as the current mean-pool already does).
  - prototypes 1..P-1 = the P-1 tokens with the **largest L2 distance from the block mean**.
- Blocks with fewer than P real tokens: fill unused prototype slots with the block mean
  (harmless under `max`).

Feature capture in `_capture_attention`:
- **K side** (already per kv-group, pre-GQA-repeat): proto-pool directly →
  `pooledK [1, kv, kb, P, d]`.
- **Q side**: average the `rep = H/G` query heads **per token** into their kv-group
  *first*, then proto-pool → `pooledQ [1, G, qb, P, d]`. This is cleaner than today's
  post-pool head average and is required so each prototype is a real per-group token.
- **Target** `pooled [1, H, qb, kb]` — **unchanged**, still the exact chunked attention
  mass, still averaged H→G inside `targets.py`.

Output stacks in `get_pooled_targets`:
- `pooledQ_stack [1, L, G, qb, P, d]`
- `pooledK_stack [1, L, G, kb, P, d]`
- `pooled_stack  [1, L, H, qb, kb]` (unchanged)

Add a `proto` (= P) field to the saved `.pt` metadata so consumers can detect format.

### 5.2 `selector/targets.py`

- Load the new P axis. `q_feat [L,G,qb,P,d]`, `k_feat [L,G,kb,P,d]`.
- Q side no longer needs head→group averaging (done in extraction). Simplify accordingly.
- Target / `cmask` / normalization logic **unchanged**.
- **Format guard:** if a loaded `.pt` lacks the `proto` field / P axis (old single-vector
  dump), raise a clear error telling the user to re-extract. No silent fallback.

### 5.3 `selector/gate.py`

`FlatGate.forward(q_feat, k_feat)` with `q_feat [L,G,qb,P,d]`, `k_feat [L,G,kb,P,d]`:
- Project: `qp = einsum(q_feat, Wq) [L,G,qb,P,p]`, `kp = einsum(k_feat, Wk) [L,G,kb,P,p]`.
- `score(q,k) = max over (i in P_q, j in P_k) of qp[q,i] · kp[k,j]` → `[L,G,qb,kb]`,
  divided by `sqrt(p)` as now.
- **Parameter count unchanged** — same per-layer `Wq, Wk`.
- **Memory:** do NOT materialize `[L,G,qb,kb,P,P]` (≈1.5GB fp16 transient at 1.5B/32K).
  Compute a **running max looped over the P key-prototypes** (8 iterations), each step a
  `[L,G,qb,kb,P_q]` tensor, keeping the transient at ~the current single-vector size.
  Offline/training path — the loop is acceptable.

### 5.4 `selector/loss.py`, `selector/recall.py` — unchanged

Both operate on `scores [L,G,qb,kb]`. Keep `combined_loss` (KL + top-k membership). The
membership term previously could not bite because features could not represent the needle;
now they can. Re-tune `lambda_topk` after observing training-set `needle_hit`.

### 5.5 Tree rollup — recorded, NOT built in this change

For the Stage-3 tree selector: a parent node's prototypes = re-run **mean + outliers over
the union of its children's prototype vectors**, re-capped to P=8. `max` composes up the
levels, so the needle survives to the root; a mean rollup would destroy it at every level.
Flat gate ships first; this note prevents a mean-based rollup being introduced later.

## 6. Validation (run by the user)

The user runs extraction and retraining; this change only lands the code.

1. Re-extract 1.5B teacher targets (train + held-out) with the new proto features.
2. Retrain the flat gate (`combined_loss`).
3. `scripts/eval_gate.py` on held-out.

**Pass signal (in order):**
- **Training-set** `needle_hit@8` climbs clearly off 0 — proves features now carry the
  needle (the decisive check; if this fails, features still insufficient → escalate P or
  revisit selection rule).
- Held-out `needle_hit@8` >> 0.04.
- `coverage@8 / oracle` stays ~near 1.0 (no bulk regression).

## 7. Non-goals / YAGNI

- No keep-all-64, no learned prototype selector, no k-means centroids (centroids re-average
  → re-dilute).
- No change to block size (stays 64).
- No tree implementation here — flat gate only.
- No `selected_blocks` interface change.

## 8. Risks

- **Needle not an L2 outlier from the block mean.** Possible but unlikely — a planted fact
  is distinct from surrounding filler. Mitigation if training-set `ndl` stays low: raise P,
  or switch the selection metric (e.g. distance in the projected `Wk` space). Decide from
  the training-set signal, not guesses.
- **Q-head→group per-token averaging** is a new approximation on the query side. If it
  regresses bulk `coverage`, revisit (e.g. keep more query prototypes). Watch coverage in
  validation.
