# SPRUCE — Lab Log

One dated entry per experiment. Format:

```
## YYYY-MM-DD — <short title>
**Question:** what this run is meant to answer.
**Config:** backbone / length / block_size / P / loss / epochs / gate proj_dim / data (train vs heldout) / commit.
**Number:** the metric(s) that came out. recall@k, coverage/cov_ratio@k, needle_union@k, KL, TTFT, peak mem — whatever the run produced.
**Conclusion:** one line. What it means and what it gates (which kill switch / next step).
```

Rules (from CLAUDE.md):
- Numbers reported on held-out documents only. `eval_gate.py --targets` must point at docs the gate did NOT train on, or the number is overfit and meaningless.
- `recall@k` is a **block-level proxy**, not the KS1 ">95% of dense RULER" number. True RULER needs the sparse generate path (kernel + Qwen swap), which does not exist yet.
- All latency numbers on ONE fixed GPU (one A100-80GB or H200). Laptop timings are development signal, not paper numbers.
- Never write "linear selection" or "sub-linear total." Selector is O(log L) per query block → O(n log n) total prefill selection, O(n·k) attention read.

---

## Status snapshot — 2026-07-21

**Stage:** Stage 3.2 started. The tree export already produces validated `selected_blocks`; a correctness-first PyTorch sparse-prefill attention backend is now registered through Transformers' custom-attention interface. It is prefill-only and has unit coverage, but has not yet been exercised with an installed Qwen checkpoint. Stage 3.3 (Triton kernel) remains unstarted, so no true RULER number yet — recall/coverage proxy only.

**Backbone:** Qwen2.5-Coder-1.5B (H=12, G=2 kv-groups). 3B = laptop ceiling; 7B = ARC only.

**Built + validated:** chunked teacher extraction (P-prototype Q/K, offloading, GPU budget), chunked-vs-eager validator, `selected_blocks` frozen + validator, needle harness (detected lost-in-the-middle), flat gate (memory-safe MaxSim autograd), tree (`build_key_tree` / `build_target_tree` / `tree_kl_loss`), `eval_tree_traversal.py` (top-down beam descent, simulated), repo-index parser, test suite.

**Open checkpoints / kill switches:**
- KS1 (Stage 2): >95% teacher top-r recall at 16-32K, measured as recall not loss. Proxy measurable now; true version needs kernel.
- Stage 2b: does a selector trained short (16-32K) generalize to long (64-128K) via needle harness? Not yet run.
- Quantization checkpoint: does quantizing the backbone hurt needle recall at target length? Not yet run.
- KS2 (post-benchmark): tree beats/matches HiP at equal budget AND beats flat routing in real prefill cost. Blocked on Stage 3.

---

## Entries

<!-- Newest first. Paste eval_gate.py / eval_tree_traversal.py output into Number, distill to one line in Conclusion. -->

## 2026-07-21 — Sparse-prefill reference seam
**Question:** Can a validated `selected_blocks` tensor constrain Qwen-style grouped-query attention without materializing a full sequence-by-sequence score tensor?
**Config:** PyTorch reference backend; blockwise query/KV-group loop; causal-mask + selected-block mask; unit tensors including Qwen-style 2:1 Q-to-KV heads.
**Number:** 13/13 focused validator, tree, sparse-attention, and teacher-prompt-reconstruction tests passed; legacy held-out target reconstruction reproduced 30,573 tokens and needle block 74.
**Conclusion:** The offline-routing-to-sparse-attention seam is correct at unit-test scale, and replay reconstructs its exact extraction prompt from target metadata. A Qwen sparse replay is the next integration check; this is not a latency result.

## 2026-07-21 — Log started
**Question:** —
**Config:** —
**Number:** —
**Conclusion:** Log created. Backfill the best trained flat-gate eval here: run `python scripts/eval_gate.py --gate <best>.pt --targets teacher_targets/heldout_*.pt` and record recall@k / cov_ratio@k / needle_union@k on held-out docs as the first real KS1-proxy entry.
