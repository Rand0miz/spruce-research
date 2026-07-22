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

**Stage:** Stage 3.3 started. Stage 3.2's correctness-first PyTorch sparse-prefill path has completed an offline held-out Qwen replay. The Stage 3.3 kernel path is being adapted from SeerAttention's public block-sparse Triton implementation, beginning with conversion of SPRUCE `selected_blocks` indices into the kernel block-mask layout. No kernel parity or speed number exists yet.

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

## 2026-07-22 — Colab direct-index Nova benchmark
**Question:** Does direct iteration over K selected block IDs remove the first mask-scanning kernel's latency regression while preserving retrieval?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; direct-index Triton sparse prefill + dense SDPA decode versus dense SDPA; 32 generated tokens.
**Number:** Sparse exact=true, fuzzy=1.0, 7.2454s; dense exact=true, fuzzy=1.0, 4.3593s; answers_match=true. Sparse/dense ratio=1.66×. Previous mask-scan sparse time=7.9525s, so direct indexing improved total time by 8.9% but remains slower than dense.
**Conclusion:** Direct indexing helps but does not clear the speed bar. The next kernel revision must remove per-layer GQA K/V expansion and split prefill/decode timings before further tuning.

## 2026-07-22 — Direct-index Triton local parity
**Question:** Does the replacement direct-index kernel remain numerically consistent with the PyTorch sparse reference before long-context benchmarking?
**Config:** Laptop CUDA; `RUN_TRITON_TESTS=1`; `tests/test_triton_sparse_prefill.py`; direct selected-ID kernel with block=64.
**Number:** CUDA parity suite passed (user-run local verification).
**Conclusion:** The direct-index kernel clears the local correctness gate. Next: rerun the long held-out dense-vs-sparse Colab benchmark to measure whether it removes the first kernel's latency regression.

## 2026-07-22 — Colab Nova generated-answer benchmark
**Question:** Does Triton sparse prefill plus dense decode preserve the held-out Nova answer, and does it improve wall-clock generation time versus dense SDPA?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; sparse Triton prefill + greedy SDPA decode versus dense SDPA prefill + decode; 32 generated tokens.
**Number:** Sparse exact=true, fuzzy=1.0, 7.9525s; dense exact=true, fuzzy=1.0, 4.4338s; answers_match=true. Sparse/dense time ratio=1.79× (sparse slower).
**Conclusion:** Retrieval preservation passes on this held-out example, but the first Triton implementation fails the speed objective. Its dense block-mask scan is a correctness kernel, not yet an efficient selected-index kernel; do not claim a latency win.

## 2026-07-22 — Colab held-out Nova Triton vs dense SDPA
**Question:** Does the end-to-end 30K SPRUCE Triton prefill execute and how far do its final next-token logits differ from dense SDPA on held-out Nova?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; `flat_gate_lamt0.75_lamn0.25_lr5e4_e300.pt`; sparse Triton then dense SDPA, sequential model loads.
**Number:** Triton sparse replay completed; final-logit difference vs dense SDPA: max absolute 4.5625, mean absolute 0.389694.
**Conclusion:** The full Stage 3.3 execution path works on a long held-out prompt. These are not numerical-parity values—sparse deliberately removes dense attention edges—so answer-token agreement and generated-answer correctness must be checked next.

## 2026-07-22 — Triton kernel parity test
**Question:** Does the SeerAttention-derived Triton kernel match the Stage 3.2 PyTorch sparse reference on a causal 64-token block?
**Config:** CUDA; fp16; Qwen-style 2 query heads to 1 KV head; block=64; all causal blocks selected; `RUN_TRITON_TESTS=1`.
**Number:** 3/3 Triton adapter/parity tests passed in 13.46 seconds.
**Conclusion:** The kernel path is numerically consistent with the reference at the one-block parity case. The 30K replay did not begin because Windows checkpoint mapping hit WinError 1455 before model execution; the runner now disk-offloads transient checkpoint state during loading.

## 2026-07-22 — Triton sparse-prefill adapter scaffold
**Question:** Can SPRUCE compact selections be converted into SeerAttention-style per-query-head block masks without corrupting padding or KV-group routing?
**Config:** SeerAttention MIT block-sparse Triton forward kernel adapted under `kernels/`; SPRUCE index adapter; block=64; Qwen GQA head expansion; CPU adapter tests plus opt-in CUDA parity test.
**Number:** 8 focused tests passed; 1 CUDA Triton parity test is present but not run by default (`RUN_TRITON_TESTS=1`).
**Conclusion:** The Stage 3.3 adapter and replay backend are wired. Kernel numerical parity and GPU performance remain unmeasured until the opt-in CUDA test and replay are run.

## 2026-07-22 — Held-out Nova sparse replay completed
**Question:** Can Qwen2.5-Coder-1.5B complete a 30K prefill using exported SPRUCE block routes after the custom-attention layout correction?
**Config:** Qwen/Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; `flat_gate_lamt0.75_lamn0.25_lr5e4_e300.pt`; Python blockwise sparse-prefill reference path; layer-27/KV-group-0 routing plot.
**Number:** Sparse replay completed successfully on the held-out target after correcting the attention output layout to `[batch, tokens, heads, head_dim]`.
**Conclusion:** Stage 3.2 now has a working offline-routing → Qwen sparse-prefill integration seam. This confirms execution only; answer quality and latency still require dedicated evaluation.

## 2026-07-22 — Held-out Nova sparse replay routing check
**Question:** Does the exported tree route retain the known evidence block for the final question block on a held-out 30K target?
**Config:** `flat_gate_lamt0.75_lamn0.25_lr5e4_e300.pt`; beam=16, K=18, block=64; held-out `code_nova_4816`, length 30,573, needle block 74, final query block 477.
**Number:** Needle block 74 is selected for the final query block in 48/56 layer×KV-group routes; neither KV group selects it in layers 0–1, while at least one group does in layers 2–27.
**Conclusion:** Routing generally preserves the Nova evidence into late layers, but the layer-0 plot alone is misleading. The first sparse next-token plot is invalid as a quality signal: it exposed a `[B,H,T,D]` vs `[B,T,H,D]` attention-output layout bug that was corrected and regression-tested before rerunning replay.

## 2026-07-21 — Sparse-prefill reference seam
**Question:** Can a validated `selected_blocks` tensor constrain Qwen-style grouped-query attention without materializing a full sequence-by-sequence score tensor?
**Config:** PyTorch reference backend; blockwise query/KV-group loop; causal-mask + selected-block mask; unit tensors including Qwen-style 2:1 Q-to-KV heads.
**Number:** 15/15 focused validator, tree, sparse-attention, teacher-prompt-reconstruction, and replay-plot tests passed; legacy held-out target reconstruction reproduced 30,573 tokens and needle block 74.
**Conclusion:** The offline-routing-to-sparse-attention seam is correct at unit-test scale, replay reconstructs its exact extraction prompt from target metadata, and the routing/logit diagnostic plot is available. A Qwen sparse replay is the next integration check; this is not a latency result.

## 2026-07-21 — Log started
**Question:** —
**Config:** —
**Number:** —
**Conclusion:** Log created. Backfill the best trained flat-gate eval here: run `python scripts/eval_gate.py --gate <best>.pt --targets teacher_targets/heldout_*.pt` and record recall@k / cov_ratio@k / needle_union@k on held-out docs as the first real KS1-proxy entry.
