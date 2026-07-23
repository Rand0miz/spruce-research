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

## Status snapshot — 2026-07-22

**Stage:** Stage 3.3 active. The direct-index Triton sparse-prefill kernel is numerically matched to the PyTorch reference. Compact-ID candidate-only traversal, batched FP16 selector scoring, and K sweeps are implemented. A six-case 15K/30K K sweep preserves all generated answers down to K=10 and reaches 1.408× sum-weighted live-prefill speedup, but the new tiled/GQA kernel regresses K=18 kernel-only efficiency versus the prior comparable six-case run; isolate or revert the kernel regression before expanding the benchmark.

**Backbone:** Qwen2.5-Coder-1.5B (H=12, G=2 kv-groups). 3B = laptop ceiling; 7B = ARC only.

**Built + validated:** chunked teacher extraction (P-prototype Q/K, offloading, GPU budget), chunked-vs-eager validator, `selected_blocks` frozen + validator, needle harness, flat gate, compact-ID candidate-only recursive traversal, PyTorch sparse reference, query/head-tiled GQA-native direct-index Triton sparse prefill, K-sweep live-tree benchmark harness, repo-index parser, test suite.

**Open checkpoints / kill switches:**
- KS1 (Stage 2): >95% teacher top-r recall at 16-32K, measured as recall not loss. Proxy measurable now; true version needs kernel.
- Stage 2b: does a selector trained short (16-32K) generalize to long (64-128K) via needle harness? Not yet run.
- Quantization checkpoint: does quantizing the backbone hurt needle recall at target length? Not yet run.
- KS2 (post-benchmark): tree beats/matches HiP at equal budget AND beats flat routing in real prefill cost. Blocked on Stage 3.

---

## Entries

<!-- Newest first. Paste eval_gate.py / eval_tree_traversal.py output into Number, distill to one line in Conclusion. -->

## 2026-07-23 — Run3 six-case K=10–18 optimization sweep
**Question:** After compact route packing, batched selector scoring, and the tiled/GQA kernel changes, which K gives the best efficiency while preserving held-out retrieval, and did the K=18 implementation itself improve?
**Config:** Colab benchmark artifact (GPU model not serialized); Qwen2.5-Coder-1.5B-Instruct; six held-out targets, three at 14,745–15,209 tokens and three at 30,419–30,573; two depths for Bob/Nova and two lengths for Elena; block=64; radix=2; K={10,12,14,16,18}; effective beam={8,10,12,14,16}; FP16 selector; layer chunk=4; three-repeat medians; feature extraction and feature-file loading excluded.
**Number:** Every K produced sparse exact=6/6, dense exact=6/6, answers_match=6/6, sparse fuzzy=1.0, and the exact same sparse generated answer across all five K values for each case. Sum-weighted live-prefill speedup was K10=1.4076×, K12=1.3569×, K14=1.3165×, K16=1.3013×, K18=1.2786×; median live-total speedup was respectively 1.2457×, 1.2140×, 1.1958×, 1.1833×, and 1.1720×. K10 tree-inclusive prefill was 1.180× over dense for the three ~15K cases and 1.519× for the three ~30K cases. As K fell from 18 to 10, mean needle layer/group hit rate fell from 0.8720 to 0.8363 and any-group/all-layer rate from 0.9583 to 0.9286 despite unchanged answers. At K18, summed selector time was 0.8691s and summed kernel prefill 11.6047s; the prior comparable six-case run reported sum-weighted kernel speedup 1.4627× and live speedup 1.1804×, whereas run3 K18 reports 1.3743× kernel and 1.2786× live.
**Conclusion:** K10 is the best observed speed/accuracy point on these six cases, and compact/batched selection more than offsets a real K18 kernel-only regression. Keep the selector changes, isolate the query/head tiling or related kernel change against the previous kernel, and rerun the full 32-length/multi-depth held-out set before adopting K10 as the default or making a paper claim.

## 2026-07-23 — Tiled GQA kernel and compact selector implementation
**Question:** Can the remaining six identified implementation optimizations be added without breaking selector policy, the frozen `selected_blocks` interface, or sparse-attention numerics?
**Config:** Compact leaf-ID traversal and direct route packing; four-layer candidate-score batching by default; configurable K sweep with K-dependent effective beam; Triton query-tile autotuning over 16/32/64 tokens; GQA head-tile autotuning over 1/2/4 query heads; one-time Q prescaling; full-causal earlier-block specialization and triangular diagonal-block masking. Local RTX 4070 CUDA parity uses 64-token single-block and 128-token two-block GQA FP16 cases; no Qwen weights or long prompt loaded.
**Number:** Compact-ID packing matches the former full-mask policy in unit tests. Full normal suite: 56 passed, 3 skipped. Explicit CUDA Triton suite: 5 passed, including both sparse-reference parity cases. `git diff --check` reports no whitespace errors. No long-context latency was measured.
**Conclusion:** The optimization pass clears local correctness gates and is ready for the held-out L4 K sweep. Do not claim a speed improvement until the same 1K–32K benchmark measures it; the new kernel autotuner may choose unfused head tiles when those are faster on a given GPU.

## 2026-07-23 — Optimized FP16 selector run1/run2 comparison
**Question:** Does preprojected FP16 live-tree scoring improve end-to-end prefill efficiency over the original FP32 live traversal without changing retrieval behavior?
**Config:** Same 32 held-out targets from 948 to 30,419 tokens; Qwen2.5-Coder-1.5B-Instruct; block=64; beam=16; radix=2; K=18; three-repeat medians; run1 original FP32 selector versus run2 optimized FP16 selector. Feature extraction and feature-file loading excluded in both runs.
**Number:** Across all cases, summed selector time fell from 11.0754s to 3.3040s (70.17% reduction); summed live sparse prefill fell from 51.4049s to 43.6409s (15.10% reduction), while summed sparse-kernel time was effectively unchanged at 40.3265s versus 40.3307s and summed dense prefill at 54.1061s versus 54.1290s. Sum-weighted live-prefill speedup improved from 1.0525× to 1.2403×; median live-prefill speedup from 0.9877× to 1.1489×; median live-total speedup from 0.9968× to 1.0922×. Sparse exact remained 32/32; every sparse generated answer, sparse fuzzy score, dense exact result, needle layer/group hit rate, and needle any-group rate matched run1 exactly.
**Conclusion:** Run2 is a real selector-side improvement, not a regression. The graph's unchanged sparse-kernel curve is expected because this optimization targets live tree selection; the tree-inclusive curve and speedup improved substantially with no observed retrieval change on these cases.

## 2026-07-22 — Selector projection/tree optimization microbenchmark
**Question:** Can live tree-selection overhead be reduced without changing FP32 routes, and what additional gain is available from lower-precision selector scoring?
**Config:** Laptop RTX 4070 development microbenchmark; held-out Nova target; 30,573 tokens; block=64; beam=16; radix=2; K=18; selector only, no Qwen model. Same-process comparison of the former per-candidate projection scorer against query-once/key-once projection; vectorized parent construction; optional CUDA FP16 autocast.
**Number:** FP32 preprojection produced exactly the same leaf mask as the former scorer. Across two alternating measurements, former traversal median=1.170s and preprojected traversal median=0.711s (1.645×, 39.2% less time). In a separate warmed live-route comparison, FP16 selector autocast reduced selector time from 0.880s to 0.426s (2.06×) but changed 253/481,824 compact route slots (0.0525%); FP16 is therefore opt-in pending generated-answer validation. Final local suite 53 passed, 2 skipped.
**Conclusion:** The precision-preserving selector optimization is ready for the next L4 sweep. FP16 offers a larger possible gain but must be evaluated as an accuracy ablation, not enabled silently or treated as free.

## 2026-07-22 — Held-out 1K–32K live-tree scaling sweep
**Question:** Across a broad context-length sweep, where does tree-inclusive sparse prefill cross dense SDPA, and does sparse routing preserve retrieval?
**Config:** Colab benchmark artifact (GPU model not serialized); Qwen2.5-Coder-1.5B-Instruct; 32 held-out targets from 948 to 30,419 tokens; one target per length; block=64; beam=16; radix=2; K=18; three repeats with medians; selector uses stored pooled Q/K; feature extraction and feature-file loading excluded.
**Number:** Sparse exact=32/32 (100%), dense exact=23/32 (71.875%), answers_match=23/32, sparse mean fuzzy=1.0, dense mean fuzzy=0.7527. Manual inspection of the nine dense/sparse mismatches finds two semantically correct dense birth-year paraphrases rejected by the exact-sentence metric and seven dense outputs that omit the city/warehouse evidence. Overall median kernel-only prefill speedup=1.235×, median tree-inclusive prefill speedup=0.988×, median tree-inclusive total speedup=0.997×; sum-weighted kernel and live-prefill speedups=1.342× and 1.053×. The first sampled tree-inclusive prefill point above 1× is 17,834 tokens. On the 15 cases at 16,590–30,419 tokens, median kernel speedup=1.420×, median live-prefill speedup=1.145×, and median live-total speedup=1.106×; sparse exact=100% and dense exact=53.3%.
**Conclusion:** The sweep shows the intended length crossover and perfect sparse exact retrieval on these 32 held-out samples. Because there is only one target per length, the crossover and apparent sparse-over-dense accuracy advantage require repeated cases/depths per length before becoming paper-level evidence; feature extraction remains outside the timed path.

## 2026-07-22 — Multi-target extraction cleanup verification
**Question:** Does teacher extraction release layerwise capture buffers and completed target tensors before beginning the next target?
**Config:** Synthetic capture-stack success/failure tests; extraction/save-path regression suite; explicit capture clearing, sequential dtype replacement, per-job tensor deletion, Python GC, and CUDA cache release; no long model extraction run.
**Number:** Final local suite 52 passed, 2 skipped; capture dictionaries are empty after both successful and failed stacking.
**Conclusion:** The known cross-target tensor overlap is removed in code. A multi-target Colab run should confirm that host RAM now stabilizes near a high-water mark; this entry does not claim a measured RSS reduction.

## 2026-07-22 — Six-case live-tree held-out benchmark
**Question:** When candidate-only tree construction, traversal, and route packing are included, does SPRUCE preserve held-out retrieval and outperform dense SDPA prefill?
**Config:** Colab run (GPU model not serialized in the report); Qwen2.5-Coder-1.5B-Instruct; six held-out targets at 14,745–30,573 tokens; block=64; beam=16; radix=2; K=18; three repeats with medians; live tree from stored pooled Q/K features; feature extraction and feature-file loading excluded.
**Number:** Sparse exact=100%, dense exact=100%, answers_match=100%, mean fuzzy=1.0. Median kernel-only prefill speedup=1.433×; median tree-inclusive prefill speedup=1.153×; median tree-inclusive total-generation speedup=1.104×. Sum-weighted kernel prefill speedup=1.463× and live prefill speedup=1.180×. The two 14,745-token Bob cases were slightly slower tree-inclusive (0.959× and 0.953×); the three ~30.5K cases were 1.248×–1.303× faster tree-inclusive.
**Conclusion:** Retrieval preservation holds across all six cases, and the expected length crossover is visible: selector overhead can outweigh sparse savings near 15K but is amortized by ~30K. Expand to the planned multi-length held-out sweep before treating the crossover or aggregate speedup as stable.

## 2026-07-22 — Multi-length efficiency/accuracy report validation
**Question:** Can the live-tree benchmark turn multiple held-out lengths into reproducible efficiency, speedup, accuracy, and selector-overhead curves without manual result processing?
**Config:** Synthetic repeated-length report tests; headless Matplotlib PNG; aggregated CSV; benchmark `--plot` integration; no model latency run.
**Number:** Final local suite 50 passed, 2 skipped; plot and CSV artifact tests passed.
**Conclusion:** The 1K–32K held-out sweep can now emit its scaling figure and tabular data directly. This is tooling validation only; real curve values must come from the forthcoming held-out run.

## 2026-07-22 — Zero-copy kernel and live candidate-tree verification
**Question:** Do direct token-major Triton output, strided zero-copy Q/K/V reads, base-2 online softmax/autotuning, candidate-only recursive scoring, and vectorized route packing preserve existing numerical and routing behavior?
**Config:** Laptop RTX 4070 CUDA parity at one 64-token block; Triton configurations cover 4/8 warps and 2/3 stages; CPU/reference tests include candidate-score equivalence to the flat gate, candidate-only traversal equivalence to the former full-level simulation, selected-block validation, live-route timing schema, and punctuation-normalized retrieval scoring.
**Number:** Triton CUDA parity 4/4 passed; final full local suite 48 passed, 2 skipped.
**Conclusion:** The optimized kernel and true candidate-only live traversal clear the correctness gate. Next measurement is the multi-prompt held-out Colab benchmark including tree build, traversal, and route packing; no speed claim is attached to these implementation tests.

## 2026-07-22 — Warmed GQA-native Bob-color prefill benchmark
**Question:** Does the warmed GQA-native Triton prefill speedup and retrieval behavior reproduce on a second held-out prompt at a shorter length?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `color_green_bob`; 14,745 tokens; block=64; beam=16; K=18; 64-token untimed warm-up; GQA-native direct-index Triton sparse prefill + dense SDPA decode versus dense SDPA; 32 generated tokens; one measured run.
**Number:** Sparse prefill=1.1215s, decode=0.9859s, total=2.1074s; dense prefill=1.3554s, decode=0.9126s, total=2.2679s; answers_match=true. Sparse prefill is 1.209× faster (17.25% less time); sparse total is 1.076× faster (7.08% less time). Both outputs answer "Bob's favorite color is green"; the current scorer gives exact=false and fuzzy=0.8 against the punctuation-stripped needle `bobs favorite color is green.`
**Conclusion:** The prefill speedup reproduces on a second held-out case, while identical dense/sparse outputs show no sparse-specific retrieval loss. Fix scorer possessive normalization separately; repeat timings and longer contexts before making a general latency claim.

## 2026-07-22 — Warmed GQA-native Nova prefill benchmark
**Question:** After excluding first-call backend initialization, does GQA-native direct-index Triton sparse prefill beat dense SDPA while preserving held-out retrieval?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; 64-token untimed warm-up; GQA-native direct-index Triton sparse prefill + dense SDPA decode versus dense SDPA; 32 generated tokens; one measured run.
**Number:** Sparse exact=true, fuzzy=1.0, prefill=2.9643s, decode=1.0955s, total=4.0597s; dense exact=true, fuzzy=1.0, prefill=3.4848s, decode=0.9306s, total=4.4154s; answers_match=true. Sparse prefill is 1.176× faster (14.94% less time); sparse total is 1.088× faster (8.05% less time). Sparse decode measured 0.1649s slower despite both paths using dense SDPA decode.
**Conclusion:** The optimized kernel clears the Stage 3.3 development speed-and-correctness bar on this single held-out L4 case. Repeat runs and additional held-out targets are required; investigate the decode-time difference as noise or cache/backend-state overhead before treating total latency as stable.

## 2026-07-22 — GQA-native direct-index Nova benchmark
**Question:** Does indexing K/V at their native grouped-query heads, instead of expanding them to every query head, close the Triton sparse-prefill latency gap while preserving held-out retrieval?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct; held-out `code_nova_4816`; 30,573 tokens; block=64; beam=16; K=18; GQA-native direct-index Triton sparse prefill + dense SDPA decode versus dense SDPA; 32 generated tokens.
**Number:** Sparse exact=true, fuzzy=1.0, 4.9804s; dense exact=true, fuzzy=1.0, 4.4137s; answers_match=true. Sparse/dense ratio=1.128×, a 0.5667s (12.84%) deficit. Sparse total time improved 31.26% from the prior direct-index result (7.2454s).
**Conclusion:** Native GQA indexing preserves the held-out answer and removes most of the prior regression, but total generation remains slower than dense. Profile prefill and decode separately before changing kernel math again.

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
