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

## Status snapshot — 2026-07-24

**Stage:** Stage 3.3 kernel/integration work is technically active, but selector data/recipe repair now takes priority. The direct-index Triton sparse-prefill kernel is numerically matched to the PyTorch reference. Run8's repeated-filler needle harness was positive, but Run9's first diverse-prose bucket failed decisively: at 64K the unchanged selector retrieved 0/30 versus dense 11/30 and routed the evidence block in only 6.67% of layer/group decisions. This invalidates any general retrieval-preservation or sparse-accuracy claim from Run8 and blocks 7B scaling. A dense-screening pipeline and disjoint 12-case natural training bank are ready to identify solvable 16K/32K prompts and extract their exact full teacher targets for mixed-data retraining. Screening now uses Qwen's instruction chat template, stops generation at EOS, and requires the entire decoded answer to match a concise alias. The 2.06× L4 prefill speedup is not useful at zero sparse accuracy. Feature extraction and feature-file loading remain outside timing; live per-layer feature production is not yet integrated into the deployed forward.

**Backbone:** Qwen2.5-Coder-1.5B (H=12, G=2 kv-groups). 3B = laptop ceiling; 7B = ARC only.

**Built + validated:** chunked teacher extraction (P-prototype Q/K, offloading, GPU budget), chunked-vs-eager validator, `selected_blocks` frozen + validator, needle harness, flat gate, compact-ID candidate-only recursive traversal, PyTorch sparse reference, direct-index Triton sparse prefill with isolated causal/prescale/query-tile ablations, prefill-only CUDA profiler, K-sweep live-tree benchmark harness, repo-index parser, test suite.

**Open checkpoints / kill switches:**
- KS1 (Stage 2): >95% teacher top-r recall at 16-32K, measured as recall not loss. Proxy measurable now; true version needs kernel.
- Stage 2b: does a selector trained short (16-32K) generalize to long (64-128K) via needle harness? Not yet run.
- Quantization checkpoint: does quantizing the backbone hurt needle recall at target length? Not yet run.
- KS2 (post-benchmark): tree beats/matches HiP at equal budget AND beats flat routing in real prefill cost. Blocked on Stage 3.

---

## Entries

<!-- Newest first. Paste eval_gate.py / eval_tree_traversal.py output into Number, distill to one line in Conclusion. -->

## 2026-07-25 — First mixed natural gate: natural gain, legacy regression
**Question:** Does the first mixed original-plus-natural gate improve diverse-prose routing without losing the original repeated-needle behavior?
**Config:** Laptop structural evaluation; original `flat_gate_lamt0.75_lamn0.25_lr5e4_e300.pt` versus `natural_gate_lamt075_lamn025_k10_lr5e4_e300.pt`; live radix-2 tree; natural eval K10/beam8/local1 on dense-accepted chat-formatted held-out 16K/32K targets; legacy eval on six held-out repeated-needle targets at beams 8 and 16. One copied Atlas natural target was accidentally included twice, giving 25 rows/24 unique targets; this has negligible effect but must be deduplicated before the final report. Selector timings are laptop diagnostics and not paper numbers.
**Number:** On the 25-row natural report, natural versus original gate mean recall@10 was 0.6506 versus 0.4618 (+0.1888), coverage/oracle was 0.8789 versus 0.7427 (+0.1362), and any-group-per-layer needle preservation ratio was 0.8309 versus 0.2016 (+0.6292). Improvements held at both lengths: recall@10 0.6536/0.6469 natural versus 0.4727/0.4480 original at 16K/32K. On the six legacy held-out targets, however, beam-8 mean/worst recall@8 fell from 0.9073/0.8980 to 0.7363/0.7143, mean coverage from 0.7141 to 0.6455, and mean needle hit from 0.8363 to 0.5833. At beam 16, mean/worst recall@8 fell from 0.9898/0.9860 to 0.9062/0.8892 and mean needle hit from 0.8720 to 0.6815.
**Conclusion:** The new data fixes the diverse-prose failure but the gate catastrophically regresses the original distribution, so it cannot replace the old gate. Root-cause audit found fixed document ordering: every epoch processed all original targets and then all natural targets, creating a repeated natural-last recency bias. Retrain from the original gate with deterministic per-epoch target shuffling, then rerun both held-out suites. The natural evaluator now skips copied duplicate artifacts automatically.

## 2026-07-25 — Multi-gate natural held-out structural evaluator
**Question:** Can the original and mixed natural selector gates be compared on the same dense-solvable natural held-out prompts using the deployed live tree route rather than training-set inline metrics?
**Config:** Added `scripts/eval_natural_gates.py`; accepts multiple labeled checkpoints and full held-out teacher targets; runs candidate-only recursive traversal plus K-width/local route packing; reports teacher top-block recall, teacher-mass/oracle ratio, per-group and any-group-per-layer evidence preservation, selector time, per-length aggregates, JSON, and a three-panel comparison plot. It rejects feature-only, raw-completion, and dense-rejected artifacts. No gate evaluation was run locally.
**Number:** New evaluator unit tests: 3 passed; focused evaluator/target tests after metadata enforcement: 7 passed. Full suite before the final metadata guard: 90 passed, 11 skipped; compile and whitespace checks passed.
**Conclusion:** Once the mixed gate finishes, screen the disjoint natural-heldout bank at native 16K/32K, extract full targets only from its dense-accepted chat prompts, and compare old versus natural gates at K10. This is the structural proxy; the winning checkpoint still requires sparse generation evaluation.

## 2026-07-25 — Mixed natural-gate epoch-1 plotting failure
**Question:** Does the first corrected mixed original-plus-natural gate run enter training successfully, and why did it stop after epoch 1?
**Config:** User-reported Colab run; mixed original and verified-natural 1.5B teacher targets; 16K/32K; K10; lambda_topk=0.75; lambda_needle=0.25; lr=5e-4; intended 300 epochs. These are training-set diagnostics, not held-out evaluation.
**Number:** Epoch 1 completed in about 89 seconds with KL=1.3469, top-k loss=0.6074, and printed eligible needle loss=0.0000. Per-document training recall@8 was roughly 0.46–0.47 at 16K and 0.39–0.41 at 32K. The run then raised a Matplotlib dimension error because repeated sequence-length tags appended several document values against one eval epoch.
**Conclusion:** Optimization itself did not fail; optional plotting did. Aggregate evaluation by requested 16K/32K bucket, reset partial history on fresh runs, tolerate legacy per-document history in plotting, and rerun from epoch 1. Do not treat epoch-1 training recall as KS1 or held-out evidence.

## 2026-07-25 — Natural-gate training loss wiring correction
**Question:** Are the intended mixed-data natural-gate flags actually applied by the selector trainer?
**Config:** Audited the existing `lambda_topk=0.75`, `lambda_needle=0.25`, K10 training path before launching the mixed original-plus-natural run. The CLI parsed and serialized `--lambda-needle`, but the training loop did not add `needle_topk_loss` to the optimized loss or its metric accumulators. Wired the loss using `--needle-topk` (falling back to `--topk`) and added a loss-direction regression test. No gate training was run locally.
**Number:** Full local suite: 86 passed, 11 skipped; compile and whitespace checks passed.
**Conclusion:** Rebuild/reload the source before natural-gate training. The corrected `lamn0.25` run will genuinely optimize reader-row evidence retention; prior checkpoint names alone are not proof that this term was active.

## 2026-07-25 — Chat-formatted dense screening result
**Question:** Does the corrected dense screener produce enough instruction-following natural prompts at native 16K/32K to support full teacher extraction?
**Config:** User-reported Colab screening result; Qwen2.5-Coder-1.5B-Instruct; disjoint 12-case natural training bank; lengths={16,384,32,768}; depths={0.1,0.5,0.9}; one variant; chat-formatted prompts; EOS-aware greedy decoding; strict concise-answer acceptance. GPU identity and per-case distribution not yet reported.
**Number:** 60/72 accepted = 83.33% overall. At 16,384 tokens, 33/36 accepted = 91.67%, with all 12 cases represented and depth counts {0.1:10,0.5:12,0.9:11}. At 32,768 tokens, 27/36 accepted = 75.00%, with all 12 cases represented and depth counts {0.1:4,0.5:12,0.9:11}; every 32K case has at least one accepted prompt, but the beacon case has only one.
**Conclusion:** Case coverage is sufficient, but 32K shallow evidence is underrepresented. Run a targeted 32K/depth-0.1 multi-variant screen before full extraction, then use both accepted manifests for the first mixed-data selector retraining pass.

## 2026-07-25 — Dense screening instruction-format correction
**Question:** Why did dense Qwen ignore the screening prompt's final answer-format instruction, and can screening distinguish concise compliance from merely containing the answer somewhere in a continuation?
**Config:** Replaced raw document completion with Qwen's system/user chat template plus assistant-generation marker; included the wrapper inside length calibration; preserved the exact formatted prompt and unwrapped user content in the manifest; stopped the shared greedy generation loop on EOS; added strict whole-answer alias equality while retaining retrieval-substring diagnostics. No dense model forward was run locally.
**Number:** Cached Qwen tokenizer smoke test: 2,040 tokens for a 2,048-token request, question and needle preserved, assistant suffix present. Focused tests: 17 passed. Full local suite: 85 passed, 11 skipped.
**Conclusion:** The original screener conflated retrieval with raw-text continuation and could keep decoding after EOS. Existing screening manifests must not be reused; rerun into new output files with the rebuilt source before extracting teacher targets.

## 2026-07-25 — Dense-screened natural teacher pipeline implementation
**Question:** Can natural selector-training prompts be admitted only when dense Qwen retrieves the known fact, then passed verbatim into full teacher extraction without regeneration or held-out leakage?
**Config:** Added a disjoint 12-case `natural_train` bank; dense greedy screening defaults to native-RoPE lengths={16,384,32,768}, depths={0.1,0.5,0.9}, one deterministic prose variant per case, concise accepted-answer aliases, and 16 generated tokens. The resumable manifest records every exact prompt, seed, dense answer/score/timing, token length, evidence block, model, and RoPE settings. An optional accepted-only manifest feeds `extract_teacher_targets.py --verified-manifest`; extraction rejects model/RoPE/block/token/evidence mismatches and runs full teacher mass unless the caller explicitly requests feature-only. No model screening or teacher extraction was run locally.
**Number:** Default screening matrix is 12×2×3=72 dense candidates. Natural training and held-out case IDs are disjoint in tests. Focused manifest/generator/replay/extraction suite: 22 passed. Full local suite: 82 passed, 11 skipped; compile, both CLI-help checks, and whitespace checks passed.
**Conclusion:** The data-quality gate needed after Run9 is implemented. Run dense screening first and inspect per-length acceptance; extract and train only from the accepted manifest. These are implementation checks, not evidence that the present natural generator reaches an adequate dense acceptance rate.

## 2026-07-25 — Run9 diverse-prose 64K selector failure
**Question:** Does the selector trained on repeated-filler needles preserve query-to-evidence retrieval when nearly every background sentence is distinct and several near-miss facts are present?
**Config:** NVIDIA L4; Qwen2.5-Coder-1.5B-Instruct; static YaRN factor=4.0; 64K only; six strictly held-out fictional natural-prose cases × five depths={0.1,0.3,0.5,0.7,0.9}; block=64; K10/beam8; FP16 candidate-only selector, layer chunk=4; single-head Triton sparse prefill; three-repeat medians; concise accepted-answer scoring. Tree work is included; offline feature extraction/loading are excluded.
**Number:** Sparse exact retrieval was 0/30=0% versus dense 11/30=36.67%; paired outcomes were 0 sparse-only, 11 dense-only, 0 both, and 19 neither. Mean sparse/dense fuzzy scores were 0.1167/0.5722, and sparse/dense answers matched in 0/30 cases. Dense was correct on Council 5/5, Aquifer 2/5, Observatory 2/5, Atlas 1/5, Gallery 1/5, and Turbine 0/5; sparse was 0/5 for every case. Mean evidence-block layer/group hit rate was 0.0667 and mean any-group/all-layer rate was 0.1262; by depth, hit rates remained only 0.04–0.14. Mean dense/kernel/live sparse prefill was 11.1298s/5.1003s/5.4106s, giving 2.1822× kernel and 2.0570× live speedup; selector mean was 0.3118s. Median live-total speedup was 1.9573×. Peak allocated memory was 9.660GB sparse versus 9.200GB dense.
**Conclusion:** The current selector does not generalize from repeated filler to diverse prose; its speed comes from pruning the evidence. Dense accuracy is itself low on several cases, but the Council 5/5 dense versus 0/5 sparse result and the 6.67% routing-hit rate isolate a clear selector failure. Stop the remaining long sweep and do not scale this gate recipe to 7B. Next run native-RoPE natural prose at 16K/32K to separate content-distribution failure from length extrapolation, then train on a mixed diverse corpus while retaining Run8 as a regression set.

## 2026-07-24 — Diverse-prose retrieval harness implementation
**Question:** Did the 16K–32K selector learn general query-to-evidence relevance, or can its Run8 result be explained by repeated filler making the needle an obvious novelty outlier?
**Config:** Added a strictly held-out `natural_prose_v1` bank with six fictional single-fact QA cases, concise accepted-answer aliases, three near-miss distractors per case, and deterministic essay-style background generation in which every sentence differs in entities, dates, quantities, subject, and conclusion. Default evaluation is five lengths={64K,80K,96K,112K,128K} × five depths={0.1,0.3,0.5,0.7,0.9} × six cases = 150 paired sparse/dense trials, K10/beam8, three-repeat medians. The current selector gate is used unchanged. Exact prompts are saved inside feature artifacts for tokenizer-stable replay; no model forward was run locally.
**Number:** Cached Qwen tokenizer calibration produced 63,955 tokens for a 64,000-token request (45-token gap, observed evidence fraction 0.0995 for requested depth 0.1) and 127,956 for 128,000 (44-token gap, observed fraction 0.8993 for depth 0.9), both within one 64-token block. The generator unit counts were 1,087 and 2,174 respectively. An 80K depth=0.7 case initially missed by 95 tokens; deterministic editorial-tail calibration reduced it to 79,997 tokens (3-token gap) while preserving observed depth=0.6984. Full local suite before the calibration fix: 76 passed, 11 skipped; post-fix focused natural/replay/extraction suite: 16 passed; compile, CLI-help, and whitespace checks passed.
**Conclusion:** The controlled distribution-shift test is ready for Colab. It cannot establish performance on real corpora by itself, but it directly tests the suspected repetition/novelty shortcut before investing in 7B extraction. Passing warrants RULER/repository QA; failing means diversify the training corpus and retrain while retaining the original needle suite as a regression test.

## 2026-07-24 — Run8 full 64K–128K YaRN held-out benchmark
**Question:** Does the selector trained at 16K–32K preserve retrieval throughout 64K–128K, and how does live-tree-plus-sparse prefill scale against dense SDPA on a fixed paper-class GPU?
**Config:** NVIDIA A100-SXM4-80GB; Qwen2.5-Coder-1.5B-Instruct; static YaRN factor=4.0; nine requested lengths from 64K through 128K in 8K increments; six held-out semantic cases; five needle depths={0.1,0.3,0.5,0.7,0.9}; 270 paired cases; block=64; K10/beam8; FP16 candidate-only selector with layer chunk=4; single-head Triton kernel (autotuned to 4 warps/2 stages); three-repeat medians; dense decode for both paths. Tree construction/traversal/packing are included in live-prefill. Offline Q/K feature extraction and feature-file loading are excluded.
**Number:** Sparse exact retrieval was 262/270=97.04% versus dense 228/270=84.44%, a +12.59-point exact-sentence difference; sparse/dense mean fuzzy scores were 0.9714/0.8559. Paired outcomes were 37 sparse-only correct, 3 dense-only, 225 both, and 5 neither. Manual failure audit found that two dense “failures” correctly stated Sara's birth year in a paraphrase despite the prompt requesting the exact sentence; crediting those gives dense factual retrieval 230/270=85.19% and leaves an 11.85-point sparse advantage. The remaining sparse failures output filler rather than the fact. Sparse/dense exact by semantic template was Elena 45/45 vs 45/45, Sara 41/45 vs 43/45, Jules 43/45 vs 29/45, Nova 45/45 vs 45/45, Bob 45/45 vs 45/45, and Sigma 43/45 vs 21/45; 36/37 sparse-only wins therefore came from Jules and Sigma. By depth from 0.1→0.9, sparse exact was 52,53,54,52,51 of 54 versus dense 49,48,47,46,38. Sum-weighted kernel/live-prefill speedups were 3.6722×/3.3573×; median case kernel/live-prefill/live-total speedups were 3.5457×/3.2445×/2.4658×; sum-weighted live-total speedup was 2.5415×. At 64K, mean dense/kernel/live prefill was 2.8250s/1.0511s/1.1451s (2.4671× live); at 128K it was 9.3905s/2.1155s/2.3149s (4.0565× live). Across all cases, selector time was 40.108s of 467.979s live sparse prefill (8.57%). Mean traversal rose from 0.0854s at 64K to 0.1889s at 128K while tree build stayed near 0.008s. Peak allocated memory at 128K was 16.219GB sparse versus 15.292GB dense. Mean needle layer/group hit rate was 0.8030 and mean any-group/all-layer rate was 0.8976.
**Conclusion:** Stage 2b passes for the 1.5B synthetic needle harness: short-trained tree selection preserves 97.04% exact retrieval through 128K and produces a widening measured prefill gap on A100-80GB. The run also shows a real benchmark-specific sparse accuracy advantage after paraphrase audit, but it is not yet a general “sparse is more accurate” claim because trials reuse only six semantic templates and most wins cluster in two; the report's row-level McNemar p-value is correspondingly optimistic under within-template correlation. Next gates are broader RULER/repo-QA held-out evaluation, flat/HiP matched-budget baselines, and in-forward per-layer selector integration.

## 2026-07-24 — Full 64K–128K held-out suite implementation
**Question:** Can Stage 2b be tested across enough lengths, cases, and needle positions to distinguish length scaling from retrieval accuracy and resume safely across Colab sessions?
**Config:** Qwen2.5-Coder-1.5B-Instruct; static YaRN factor=4.0; nine requested lengths from 64K through 128K in 8K increments; six held-out cases; depths={0.1,0.3,0.5,0.7,0.9}; block=64; K10/beam8; three repeats. Added tokenizer-calibrated prompt construction, exact filler-count replay, per-length feature-only extraction and reports, optional post-report feature cleanup, paired exact-accuracy counts/McNemar statistic, runtime/GPU metadata, peak CUDA memory, and requested-length plot grouping. No long model forward was run locally.
**Number:** The planned matrix contains 270 paired sparse/dense cases and 810 timing samples per path at three repeats. Cached-tokenizer calibration produced 63,993 tokens for a 64,000-token request and 127,992 for 128,000, both within one 64-token block. Full local suite: 71 passed, 11 skipped; focused full-suite tests: 17 passed; compile, CLI-help, and whitespace checks passed.
**Conclusion:** The Colab suite is ready to measure whether short-trained sparse selection preserves or improves retrieval relative to dense across long lengths and needle locations. The implementation does not assume sparse wins; paired accuracy counts and the full curve determine that result.

## 2026-07-24 — Run7 preliminary YaRN length-generalization benchmark
**Question:** Does the selector trained at 16K–32K preserve retrieval and widen the dense-versus-SPRUCE prefill gap when applied unchanged near 64K and 128K under matched 4x YaRN?
**Config:** Colab benchmark artifact (GPU model not serialized); Qwen2.5-Coder-1.5B-Instruct; static YaRN factor=4.0, original/max positions=32,768/131,072; two held-out feature-only targets at actual lengths 58,666 and 117,329; block=64; K10/beam8; single-head Triton kernel (8 warps, 2 stages); FP16 selector; layer chunk=4; one repeat; 64-token warm-up. Tree construction/traversal/packing are included in live-prefill; offline Q/K feature extraction and feature-file loading are excluded.
**Number:** Sparse and dense exact retrieval were both 2/2, mean fuzzy=1.0, and sparse/dense generated answers matched in 2/2 cases. At 58,666 tokens, dense/kernel/live prefill were 9.5149s/4.9688s/5.8410s, giving 1.9149× kernel and 1.6290× live speedup; live prefill saved 38.61% versus dense. At 117,329 tokens they were 31.6251s/9.0678s/9.6713s, giving 3.4876× kernel and 3.2700× live speedup; live prefill saved 69.42%. Live total speedups were 1.5229× and 2.9145×. Selector time was 0.8722s and 0.6035s; the smaller case's 0.3057s tree build versus 0.0296s on the larger case indicates first-case warm-up noise. Needle layer/group hit rates were 0.7500 and 0.8214; any-group/all-layer rate was 0.8571 for both despite correct answers. Across the two different prompts, doubling actual length increased dense prefill 3.324× and sparse kernel prefill 1.825×.
**Conclusion:** Stage 2b is preliminarily positive: the short-trained selector preserved both held-out needles at up to 117K and the expected long-context efficiency gap appeared. Do not freeze the 3.270× number or close Stage 2b yet: rerun with three repeats, serialized GPU identity, multiple cases/depths per length, and eventually live in-forward feature extraction.

## 2026-07-23 — 128K YaRN and feature-only extraction path
**Question:** Can the existing Transformers extraction/replay/benchmark pipeline use Qwen2.5-Coder's documented 4x YaRN configuration at 64K–128K without generating a quadratic dense teacher-mass artifact for the Stage 2b traversal test?
**Config:** Central static-YaRN override; factor=4.0; original max positions=32,768; configured max positions=131,072; model RoPE theta preserved at 1,000,000; Transformers 5 `rope_parameters` with legacy `rope_scaling` compatibility. Added selector-feature-only Q/K prototype capture through fused SDPA, exact prompt metadata, target/config factor checks, and matching dense/sparse model configuration. No 128K model forward was run locally.
**Number:** Cached Qwen2.5-Coder-1.5B config smoke test resolved `max_position_embeddings=131072` and `{rope_type: yarn, factor: 4.0, original_max_position_embeddings: 32768, rope_theta: 1000000.0}`. Full local suite: 64 passed, 11 skipped; four CLI smoke checks and `git diff --check` passed.
**Conclusion:** The code path is ready for a Colab 64K/128K feature extraction and matched dense-versus-SPRUCE run. This is implementation validation only; YaRN retrieval quality, 128K memory fit, and latency remain unmeasured.

## 2026-07-23 — Run6 full 1K–32K query-tile benchmark
**Question:** Does the 9.08% attention-kernel improvement seen in the single 30K profile translate into repeated whole-prefill latency improvement across the full 32-case K10 sweep?
**Config:** Colab benchmark artifact (GPU model not serialized); same 32 held-out targets from 948 to 30,419 tokens as run4; Qwen2.5-Coder-1.5B-Instruct; block=64; K10/beam8; FP16 selector; layer chunk=4; `single_head_qtile`; query tile=32, 8 warps, 2 stages; three-repeat medians. Compared against run4 `single_head` K10 control; feature extraction and feature-file loading excluded.
**Number:** Query tiling preserved sparse exact=32/32, sparse fuzzy=1.0, dense exact=23/32, and all 32 sparse generated answers matched run4. Run6 summed kernel/live times were 38.0279s/40.0046s versus 37.6836s/39.6658s control: query tiling increased kernel latency 0.91% and live-prefill latency 0.85%. Sum-weighted kernel/live speedups fell from 1.4161×/1.3453× to 1.3948×/1.3259×; median live-prefill fell from 1.2560× to 1.2381× and median live-total from 1.1497× to 1.1382×. Query tiling was faster on only 3/32 paired cases and slower on 29/32, with median per-case kernel latency +1.24%. Bandwise kernel latency changes were +1.53% below 8K, +1.55% at 8–16K, +1.21% at 16–24K, and +0.22% above 24K. Stable sampled live-prefill crossover remained 4,771 tokens.
**Conclusion:** Reject `single_head_qtile` as a production optimization despite its isolated profiler result. The attention-kernel event is too small a share of the full forward for that micro-improvement to survive system-level overhead/noise; keep the original `single_head` K10 path as default.

## 2026-07-23 — Run5 30K K10 isolated-kernel profiles
**Question:** Which single-head-only kernel change reduces the Triton attention kernel's device time without changing the model's next token?
**Config:** Colab profiler artifacts (GPU model not serialized); Qwen2.5-Coder-1.5B-Instruct; held-out Elena target; 30,419 tokens; block=64; K10/beam8; FP16 selector; layer chunk=4; one profiled prefill per variant after warm-up; control versus causal-only, Q-prescale-only, and query-tile-only. Profiler wall time includes profiler overhead and is not a benchmark latency.
**Number:** All variants produced next-token ID 68575. Control Triton kernel self-device time was 119.10ms across 28 layer calls with autotune choosing 8 warps/2 stages. Causal-only was 123.27ms (+3.50%); prescale-only 119.53ms (+0.36%); query-tile-only 108.29ms (-9.08%), choosing query tile=32, 8 warps, 2 stages. Profiled wall times were 2.779s control, 2.824s causal, 2.789s prescale, and 2.653s query-tile, but these single profiler-instrumented passes are not used as latency claims. The control attention kernel accounts for roughly 4.3% of profiled wall time; `aten::mm` alone records about 1.083s self-device time, showing that non-attention model work dominates the remaining prefill.
**Conclusion:** Drop causal specialization and Q prescaling as speed candidates. `single_head_qtile` is the only surviving kernel ablation and now needs a normal repeated K10 benchmark against `single_head`; even a real 9% attention-kernel gain has a small whole-prefill ceiling because the attention kernel is only a few percent of the profiled forward.

## 2026-07-23 — Isolated kernel ablations and prefill profiler
**Question:** Can the next kernel optimizations be tested one at a time, without reintroducing GQA head fusion or obscuring which change affects latency?
**Config:** Preserved `single_head` as the unchanged control. Added `single_head_causal` (only causal-mask specialization), `single_head_prescale` (only one-time Q scaling), and `single_head_qtile` (only 16/32/64 query-tile autotuning, one head per program). Retained `tiled_gqa` solely as the prior run3 ablation. Added a prefill-only profiler that exports Chrome trace, operator table, JSON top-op summary, separate selector timing, and the Triton autotuner's chosen configuration. No model or long-context profiling run locally.
**Number:** Explicit CUDA parity matrix 13/13 passed across all five variants at 64-token single-block and 128-token two-block GQA cases. Autotune metadata was non-empty for every variant after execution. Full guarded suite 58 passed, 11 skipped; CLI and whitespace checks passed.
**Conclusion:** The control and three single-change variants are ready for matched K10 profiling/benchmarking. No variant is called faster until the same held-out L4 inputs measure it; GQA head fusion remains excluded from the new optimization path.

## 2026-07-23 — Run4 full 1K–32K single-head K10/K18 sweep
**Question:** Does restoring the run2 single-head kernel recover K18 efficiency on the same 32 cases, and does K10 preserve retrieval while improving the full length curve?
**Config:** Colab benchmark artifact (GPU model not serialized); Qwen2.5-Coder-1.5B-Instruct; same 32 held-out targets from 948 to 30,419 tokens as run2; one target per sampled length; block=64; radix=2; single-head kernel; K10/beam8 versus K18/beam16; FP16 selector; layer chunk=4; three-repeat medians; feature extraction and feature-file loading excluded.
**Number:** K10 and K18 both produced sparse exact=32/32, sparse fuzzy=1.0, dense exact=23/32, and answers_match=23/32; every sparse generated answer was identical between K10 and K18. K10 summed selector/kernel/live-prefill times were 1.9795s/37.6836s/39.6658s versus dense 53.3643s, giving sum-weighted kernel/live speedups 1.4161×/1.3453×, median live-prefill 1.2560×, and median live-total 1.1497×. K18 times were 3.0737s/39.9859s/43.0603s, giving 1.3346×/1.2393×, median live-prefill 1.1495×, and median live-total 1.0908×. Run2 K18 was 1.3421× kernel, 1.2403× live, 1.1489× median live, and 1.0922× median total, so run4 K18 is within about 1% on all aggregate speed metrics. K10 reduces live-prefill time 7.88% versus K18 and moves the stable sampled live-prefill crossover from 7,604 to 4,771 tokens. K10 bandwise sum speedups were 1.04× below 8K, 1.21× at 8–16K, 1.35× at 16–24K, and 1.47× above 24K. Mean needle layer/group hit rate declined from 0.8862 at K18 to 0.8516 at K10; any-group/all-layer rate declined from 0.9420 to 0.9275.
**Conclusion:** The kernel rollback recovered the run2 K18 baseline, confirming the tiled path caused the regression. K10 is the strongest current development configuration across the full sampled length range with no observed answer loss, but its lower routing-hit proxy and one-case-per-length design require repeated cases/depths and harder retrieval tasks before it becomes the default paper configuration.

## 2026-07-23 — Restore measured-fast kernel as default
**Question:** Can the run3 kernel-only regression be removed without discarding the successful compact-selector changes or losing the tiled kernel needed for a controlled A/B?
**Config:** Restored the run2 direct-index kernel shape—one query head and one 64-token query block per Triton program, native GQA KV indexing, original three-config warp/stage autotune—as `single_head`, now the default. Preserved the run3 query/head-tiled implementation as opt-in `tiled_gqa`. Added `--kernel-variant {single_head,tiled_gqa}` to the live benchmark and serialized the choice in every sweep report. No model or long-context latency run locally.
**Number:** Full normal suite 56 passed, 5 skipped. Explicit CUDA suite 7 passed, covering both kernel variants at 64-token single-block and 128-token two-block GQA sparse-reference parity. CLI/help and whitespace checks passed.
**Conclusion:** Correctness is preserved and the empirically faster pre-run3 kernel is again the production default. Run the same six-case K10/K18 Colab benchmark with `single_head`; a speed recovery is expected from prior evidence but is not claimed until measured.

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
