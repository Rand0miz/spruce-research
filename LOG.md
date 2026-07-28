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

## 2026-07-28 — Unscreened 16K–128K compiler beats dense by 15.6 points at 9.58x
**Question:** On the first run of the sealed, unscreened natural paper bank, does the frozen
beam-16 evidence compiler remain more accurate than full dense Qwen as context grows from
16K to 128K, and does it retain a fully charged speed and memory advantage?
**Config:** Colab run `natural_yarn_beam16_paper_v1`, completed
`2026-07-28T07:13:10.625806+00:00`, from source archive SHA-256
`1BA99CBA3681467B35F056E72C337E85DE8E50B0671C6C355FC8598F2F208595`
and sealed prompt-bank SHA-256
`74ACE23201F9FA73D3EE7AE633583215E44D37F4E6A84D82000B9B6B366DF5CE`.
Hardware/software was one NVIDIA A100-SXM4-40GB (reported 42.406 GB), compute capability
8.0, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Transformers 5.13.1, and
Triton 3.6.0.

Qwen2.5-Coder-1.5B-Instruct FP16 was frozen in both modes with matched static YaRN factor
4.0 (32,768 original / 131,072 configured positions). The first-run matrix contained 12
new semantic cases in 12 genres, requested lengths {16, 32, 48, 64, 80, 96, 112, 128} Ki
tokens, depths {0.1, 0.5, 0.9}, and 288 paired prompts. The compiler used D=512 lexical
features, radix 2, M=4 final selected blocks, beam=16 traversal, radius-1 paragraph expansion,
and exact-text stitching. Each mode ran three alternating-order repeats with 32 greedy decode
tokens; answers were stable across all repeats. There was no dense screening, prompt
resampling, training, selector tuning, or index caching.

The request boundary is fully charged from in-memory prompt/question strings. Dense includes
full-prompt tokenization, transfer, full dense prefill, and dense decode. Compiled includes
full-prompt offset tokenization, hierarchy construction, question features, top-down
traversal, exact span expansion/stitching, compact tokenization, transfer, compact dense
prefill, and dense decode. Deterministic synthetic prompt construction, model/tokenizer load,
and one backend warm-up are reported separately and excluded symmetrically.
**Number:** Overall dense exact was **192/288 (66.667%, Wilson 95% 61.034–71.860%)**.
Compiled exact was **237/288 (82.292%, Wilson 95% 77.466–86.267%)**, an absolute
**+15.625-point** improvement. Paired outcomes were 174 both correct, **63 compiler-only**,
18 dense-only, and 33 neither; exact McNemar **p=5.2044e-7**. The compiled raw rate exceeded
dense at all eight lengths:

| Requested length | Dense | Compiler | Delta | Compiler-only / dense-only | McNemar p | Fully charged speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 28/36 (77.8%) | 30/36 (83.3%) | +5.6 pt | 4 / 2 | 0.6875 | 2.485x |
| 32K | 27/36 (75.0%) | 30/36 (83.3%) | +8.3 pt | 5 / 2 | 0.4531 | 4.112x |
| 48K | 24/36 (66.7%) | 29/36 (80.6%) | +13.9 pt | 7 / 2 | 0.1797 | 5.778x |
| 64K | 23/36 (63.9%) | 29/36 (80.6%) | +16.7 pt | 8 / 2 | 0.1094 | 7.421x |
| 80K | 20/36 (55.6%) | 30/36 (83.3%) | +27.8 pt | 12 / 2 | 0.01294 | 9.183x |
| 96K | 22/36 (61.1%) | 29/36 (80.6%) | +19.4 pt | 10 / 3 | 0.09229 | 10.617x |
| 112K | 24/36 (66.7%) | 31/36 (86.1%) | +19.4 pt | 9 / 2 | 0.06543 | 12.431x |
| 128K | 24/36 (66.7%) | 29/36 (80.6%) | +13.9 pt | 8 / 3 | 0.2266 | 14.083x |

Across the five 64K–128K cells, dense scored 113/180 versus compiler 148/180
(62.78% versus 82.22%, **+19.44 points**), with 47 compiler-only versus 12 dense-only
(paired p=5.126e-6) and 11.163x sum-weighted speedup. The distance-from-query effect is the
strongest accuracy result: at depth 0.1, dense scored 46/96 (47.92%) versus compiler 81/96
(84.38%), +36.46 points, 40 compiler-only versus 5 dense-only (p=7.878e-8). Restricted to
depth 0.1 at 64K–128K, dense was 24/60 (40.0%) versus compiler 50/60 (83.33%),
**+43.33 points** (30 compiler-only / 4 dense-only, p=6.165e-6). At depths 0.5 and 0.9,
the deltas were smaller: 69/96 versus 75/96 (+6.25 points) and 77/96 versus 81/96
(+4.17 points).

Fully charged summed per-case median request time was **1,368.499s dense versus 142.836s
compiled**, or **9.58093x**. Overall median request time was 4.14547s versus 0.477956s;
median model-prefill component was 3.73215s versus 0.0389413s. Median per-prompt speedup was
8.2919x with IQR 4.9706–11.5383x. At 128K, median request latency was 10.7090s dense versus
0.767158s compiled. Median compiler input was **1,849.5 tokens**, 2.4637% of the original;
at 128K it was 1,842.5 tokens, 1.4061% retained. The compiler mean request components sum to
0.49452s: offset/layout tokenization 53.90%, dense decode 24.54%, exact span stitching 9.77%,
compact prefill 7.78%, hierarchy construction 2.37%, compact tokenization 1.13%, selection
0.47%, and transfer 0.04%. The actual tree selection itself averages only 2.344ms; live
full-prompt offset tokenization is the largest remaining cost.

Median peak allocated GPU memory was 4.658 GB dense versus 3.269 GB compiled at 16K and
15.582 GB versus 3.273 GB at 128K; the 128K allocated reduction is **79.0%**. Do not use
the reserved-memory panel as a path comparison: both lines equal the dense allocator
high-water mark at every length because dense and compiled alternate in one process and
PyTorch retains cached reservations.

Evidence routing explains the remaining errors cleanly. Direct M=4 recall was 232/288
(80.56%); radius expansion raised it to 256/288 (88.89%), repairing 24 direct misses.
When expanded evidence was present, compiler exact was **237/256 (92.58%)**, with 63
compiler-only and zero dense-only rows. When expanded evidence was absent, compiler exact
was **0/32**, and all 18 dense-only rows occurred in this subset. Thus every compiled exact
answer had access to the expanded source evidence, and every dense-only loss is attributable
to selection/expansion missing it rather than Qwen failing to read a selected packet.
Radius repair is load-bearing: 21/24 of the direct-miss/expanded-hit rows became exact.

The two non-routing read clusters are also identifiable. `paper_council_vote19_6` scored
0/24 in both modes; with evidence present the model usually emitted the incomplete answer
`19`, while missed packets selected the distractor `6-to-2`. `paper_astronomy_polaris_field508`
had full expanded recall but compiled 17/24; six failures emitted `Polar Field 508` rather
than the required `Polaris Field 508`, and one selected `Polar Mosaic Five`. The principal
selector failure was `paper_engineering_alloy_r62`: compiled 4/24 versus dense 22/24 because
expanded recall was only 4/24; all four evidence hits produced exact `R-62`.

The predeclared semantic-case guardrail does **not** clear. Eight semantic cases have a
positive compiler delta, three tie, and one (`alloy_r62`) is negative. The seeded 10,000-draw
12-cluster bootstrap has median +15.625 points but 95% interval
**[-5.556, +34.722] points**, so its lower bound crosses zero. The decision correctly records
`compiled_more_accurate_raw_rate=true`, `raw_paired_mcnemar_significant=true`, and
`semantic_case_cluster_bootstrap_supports_positive_delta=false`. The 288 repeated
length/depth rows therefore support a strong result on this controlled distribution, but
they are not 288 independent semantic tasks and do not yet justify a general-superiority
claim.

Synthetic prompt construction took 1,787.464s total outside request timing (overall median
6.083s; 128K median 12.694s). All 288 candidate IDs are unique; every length report is
complete with 36 cases, configuration and GPU metadata match across all eight files, and
all dense/compiled repeat answers are stable. The primary combined report SHA-256 is
`AE1DEB4939FA1D06111D663D80355BD4423D3627AC611F854D9EB827FC5FB35A`.
Local artifacts are under
`benchmarks/outputs/natural_yarn_beam16_full_results/`; the paper overview, length×depth
heatmap, and GPU-memory charts were visually inspected. Twelve figures are present in both
PNG and vector PDF alongside five CSV tables, the complete paired JSON, and `SUMMARY.md`.
**Conclusion:** The central engineering/research mechanism is supported: converting selected
regions back into a short, coherent exact-text document makes the frozen model both
substantially more accurate and much faster than asking it to read the full long context,
especially when the evidence is far from the final query. This is the first untouched,
unscreened 16K–128K result for the frozen beam-16 configuration, and it achieves exactly the
intended context-compilation behavior. Freeze this bank permanently. The next validation
must add independent untouched semantic cases and external long-context tasks without tuning
beam, M, D, radius, or packet formatting on these results; the current case-cluster interval
is the remaining paper-claim blocker, not the row-level effect or the systems result.

## 2026-07-28 — Unscreened 16K–128K natural YaRN paper suite packaged
**Question:** Can the frozen beam-16 live evidence compiler be compared with full dense
Qwen across the entire 16K–128K YaRN range, without dense-screening the generated prompts,
while preserving enough paired detail and uncertainty reporting to test whether compilation
can outperform dense reading at long context?
**Config:** Implementation and harness verification only; no Qwen accuracy or latency result
was produced locally. Added a new sealed prompt bank,
`scripts/prompt_banks/natural_paper_untouched.json`, SHA-256
`74ACE23201F9FA73D3EE7AE633583215E44D37F4E6A84D82000B9B6B366DF5CE`,
with 12 previously unused semantic cases in 12 genres and three close distractors per case.
The locked first-run matrix is Qwen2.5-Coder-1.5B-Instruct FP16, static YaRN factor 4.0
(32,768 original / 131,072 configured positions), requested lengths
{16,384, 32,768, 49,152, 65,536, 81,920, 98,304, 114,688, 131,072},
evidence depths {0.1, 0.5, 0.9}, seed 20260728, D=512, M=4 final blocks, beam=16,
radix 2, radius-1 paragraph repair, 32 generated tokens, and three alternating-order
repeats. The dense and compiled modes use the same generated prompt and frozen backbone;
there is no dense acceptance filter, prompt resampling, selector tuning, or training.

Prompt synthesis is deterministic harness setup and is timed separately, not hidden. Both
request timers exclude model/tokenizer load and one backend warm-up. Dense charges
full-prompt tokenization, transfer, dense prefill, and dense decode. Compiled charges
full-prompt offset tokenization, lexical hierarchy construction, question features,
top-down traversal, exact-span expansion and stitching, compact-prompt tokenization,
transfer, compact dense prefill, and dense decode. Index caching is disabled.
**Number:** The complete matrix contains **288 paired prompts** (12 semantic cases × three
depths × eight lengths) and **1,728 measured model requests** (two modes × three repeats).
Every paired case is atomically checkpointed; every completed length regenerates partial
paper artifacts, and a length is considered resumable-complete only after its expected case
count and final prompt-build summary are present.

The final reporter writes the complete paired JSON, a Markdown summary, and five CSV tables
(raw cases, by length, by depth, by semantic case, and length×depth). It produces **12
paper figures in both 220-DPI PNG and vector PDF**: accuracy with Wilson intervals, paired
outcomes, latency/speedup, compiled latency components, context compression, direct/expanded
evidence recall, length×depth heatmaps, semantic-case accuracy, a four-panel overview, peak
GPU memory, per-prompt speedup median/IQR, and excluded prompt-construction time. Statistical
controls include exact paired McNemar and a seeded 10,000-draw semantic-case cluster bootstrap
of the compiler-minus-dense accuracy delta. The notebook only marks a positive accuracy
effect as cluster-supported when that bootstrap's lower 95% bound exceeds zero; even then its
decision record limits the interpretation to this controlled retrieval distribution.

A tokenizer-only construction control on the first sealed case at depth 0.5 produced 16,346
prompt tokens inside the 16,352-token budget at requested 16K (needle block 127; 2.654s
harness construction), and 131,027 prompt tokens inside the 131,040-token budget at requested
128K (needle block 1,022; 30.264s construction). No model read or answer screening was run on
either control.

Full local verification passed **203 tests / 13 skipped** with the single existing top-k-loss
warning. The exact focused Colab control set passed **21/21** with only SWIG deprecation
warnings. The paper-report suite alone passed 6/6; all 12 synthetic figures were rendered and
the heatmap, memory, and speedup figures were visually inspected. Python compileall,
notebook-cell AST parsing, CLI help, and `git diff --check` passed. The notebook has 9 cells /
7 code cells. The refreshed `spruce_colab_train_source.zip` contains **214 entries**, is
**7,365,183 bytes**, has zero Python cache entries, and has SHA-256
`1BA99CBA3681467B35F056E72C337E85DE8E50B0671C6C355FC8598F2F208595`.
The recoverable prior archive is
`spruce_colab_train_source.pre_rebuild_20260728_010730.zip`.
**Conclusion:** The resumable first-run comparison is ready in
`colab/run_pre_qwen_natural_yarn_paper.ipynb`. Its actual accuracy, speed, memory, and
long-context curves remain unmeasured until the Colab run completes. A compiler advantage
must be judged from paired wins/losses and the semantic-case-clustered interval—not from the
288-row raw rate alone—and cannot by itself establish general model-quality superiority.

## 2026-07-28 — Beam-16 candidate slack recovers 25/25 at 7.52x fully charged speed
**Question:** Was beam=M=4 prematurely pruning useful lexical branches, and can wider
traversal recover the two live-compiler errors without increasing the final M=4 Qwen packet
or sacrificing end-to-end speed?
**Config:** Colab follow-up `pre_qwen_beam16_381d4e742ac0`, completed
`2026-07-28T04:36:42.208299+00:00`, from source archive SHA-256
`381D4E742AC01016CFBBAB6B76CE28673ABAF3D62B6AF8FFC6AF26C37549D266`.
The only method change from the completed beam-4 run was traversal beam 4 -> 16.
Qwen2.5-Coder-1.5B-Instruct FP16, the same 25 prompts, D=512, M=4 final blocks,
unigram fraction 0.5, squared block IDF, radix 2, radius-1 paragraph repair, three
alternating-order repeats, 32 generated tokens, and the fully charged timing boundary were
otherwise unchanged. Index caching remained disabled and the hierarchy was rebuilt on every
measured request. The dense baseline was rerun on the same loaded model/GPU rather than
reusing old timing.

This is explicitly a follow-up on an opened set:
`evaluation_status=followup_on_opened_25_cases_not_untouched_gate` and
`untouched_gate_claim=false`. It establishes engineering behavior, not a new untouched
scientific claim.
**Number:** Dense and compiled both achieved **25/25 exact (100%)**. Compiled split
**14/14 at 16K** and **11/11 at 32K**. Relative to beam 4, compiled accuracy rose from
23/25 to 25/25 and radius-expanded evidence recall rose from 24/25 to **25/25**. Direct
M=4 evidence-block recall remained **21/25 (84%)**, so the fixed radius repair continues to
be load-bearing. The unchanged direct recall plus two recovered answers means candidate
slack changed which surrounding/source passages reached the final packet, not merely whether
the exact needle block ranked in the four direct selections.

Median compiled input was **1,833 tokens (8.963% of original)**, only 72 tokens above the
beam-4 median of 1,761 despite the wider traversal because final M remained four. Summed
per-case median fully charged time was **74.0856s dense** versus **9.85087s compiled**,
giving **7.52072x sum-weighted speedup**. Median request time was 1.91307s dense versus
0.398316s compiled; median prefill component was 1.63516s versus 0.123718s. By length,
sum-weighted speedup was **5.10308x at 16K** (26.1276s / 5.11997s) and **10.1372x at
32K** (47.9580s / 4.73090s). The small apparent speed increase versus beam 4 (7.456x ->
7.521x) is ordinary run noise, not evidence that a wider beam is faster.

The supplied decision is preserved at
`eval/pre_qwen_runs/pre_qwen_beam16_381d4e742ac0/decision.user_supplied.json`. Drive
artifacts are
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_beam16_381d4e742ac0/pre_qwen_beam16_e2e_25.json`,
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_beam16_381d4e742ac0/decision.json`,
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_beam16_381d4e742ac0.zip`, and stable pointer
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_beam16_latest.json`.
**Conclusion:** The no-slack diagnosis is supported: widening traversal alone recovers both
answers and complete expanded evidence coverage while retaining the same four-block model
input and essentially identical 7.5x fully charged speed. **Do not retrain a selector.**
Freeze beam=16 as the engineering configuration and validate it on approximately 100 new,
untouched prompts before making a paper/toolkit accuracy claim.

## 2026-07-28 — Beam-16 Colab test-order failure fixed before benchmark
**Question:** Why did the beam-16 notebook stop in its pre-benchmark pytest cell on a fresh
Colab runtime, and can the launcher make the real failure visible if it recurs?
**Config:** User-reported Colab environment: torch 2.11.0+cu128 and transformers 5.13.1.
The subprocess running the focused integration suite exited with status 1 before any model
benchmark. The pasted traceback contained only `CalledProcessError`, not pytest's underlying
stdout. Inspection of the test order found that the notebook ran integration tests before
downloading Qwen, while the real-tokenizer tests intentionally call
`AutoTokenizer.from_pretrained(..., local_files_only=True)`. On a fresh runtime this can fail
solely because the cache is empty, even though the later benchmark would have downloaded the
same tokenizer.
**Number:** No beam-16 accuracy or timing number was produced. Patched
`colab/run_pre_qwen_beam16_followup.ipynb` to download
`Qwen/Qwen2.5-Coder-1.5B-Instruct`'s tokenizer before pytest, capture combined pytest
stdout/stderr, print it, and only then call `check_returncode()`. Notebook JSON remains valid
at 8 cells / 6 code cells; all code cells parse after removing the `%cd` magic. The focused
non-integration controls pass **14/14** locally. The refreshed upload ZIP contains 206
entries, is 7,336,464 bytes, and has SHA-256
`381D4E742AC01016CFBBAB6B76CE28673ABAF3D62B6AF8FFC6AF26C37549D266`.
The recoverable previous archive is
`spruce_colab_train_source.pre_rebuild_20260728_002754.zip`.
**Conclusion:** This was a launcher/test-order problem, not a beam-16 result. Re-upload the
refreshed ZIP and rerun from the first cell. If pytest still fails, its complete output will
now be printed and can be diagnosed directly rather than hidden behind `CalledProcessError`.

## 2026-07-28 — Beam-16 one-variable Colab follow-up packaged
**Question:** Can the no-slack beam=M=4 diagnosis be tested with one isolated selector
change while preserving the same M=4 Qwen packet and the fully charged timing contract?
**Config:** Implementation/packaging only; no beam-16 model result was produced locally.
Added `colab/run_pre_qwen_beam16_followup.ipynb`. The notebook loads the completed beam-4
decision from `MyDrive/SPRUCE_COLAB/outputs/pre_qwen_latest.json`, asserts that the only
changed method parameter is beam 4 -> 16, and reruns the same 25 prompts with
Qwen2.5-Coder-1.5B-Instruct FP16, D=512, M=4, radius-1 paragraph repair, radix 2, three
alternating-order repeats, and 32 generated tokens. The full dense baseline is rerun on the
same loaded model/GPU; no old timing is reused. Index caching remains disabled.

Because the 25 prompts were already opened by the beam-4 decision, the notebook explicitly
records `evaluation_status=followup_on_opened_25_cases_not_untouched_gate` and
`untouched_gate_claim=false`. This run may establish an engineering repair but cannot be
presented as a second untouched accuracy gate. Results use a separate stable pointer,
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_beam16_latest.json`, so the original beam-4 pointer
is not overwritten.
**Number:** Notebook JSON parses as nbformat 4 with **8 cells / 6 code cells**; every code
cell parses after removing the one Colab `%cd` magic. Focused selector/compiler/benchmark
verification passed **16/16** with the same SWIG deprecation warnings. `git diff --check`
passed. The refreshed `spruce_colab_train_source.zip` contains **206 entries**, is
**7,336,260 bytes**, and has SHA-256
`6FC32AB7FA879E1C48EFF7A7C9A9EFE9BC2479246BA9694FC02DB78B0BC248C1`.
The recoverable previous archive is
`spruce_colab_train_source.pre_rebuild_20260728_002240.zip`.
**Conclusion:** Upload the refreshed ZIP and run the beam-16 notebook. A gain from 23/25 to
at least 24/25 with speed still above 1.0x would validate candidate slack as the engineering
repair, but the result must remain labeled a follow-up on an opened set until confirmed on
new untouched prompts.

## 2026-07-28 — Fully charged live pre-Qwen path is 7.46x faster but misses accuracy by one
**Question:** Does the locked tokenizer-only hierarchy preserve the evidence-compiler
accuracy gate while remaining faster than ordinary full-dense Qwen after charging every live
request cost?
**Config:** Colab run `pre_qwen_b9f3748f4287`, completed
`2026-07-28T04:10:06.794184+00:00`, from source archive SHA-256
`B9F3748F42875308B9FE13EA15619DDED42CE4237E56378168D6BDDCBD90E7C2`.
Qwen2.5-Coder-1.5B-Instruct FP16; 25 frozen dense-accepted prompts (14 at 16K,
11 at 32K); D=512 Boolean hashed lexical sketches; half unigram and half adjacent-token
buckets; squared block-IDF question weights; radix-2 max/union tree; M=4 output blocks;
beam=4; radius-1 paragraph repair; 32 generated tokens; three repeats with dense/compiled
execution order alternated. The index was disabled as a cache and rebuilt on every measured
repeat.

Both paths started from prompt/question strings already resident in memory. Model and
tokenizer loading, manifest reading, and one backend warm-up were excluded from both.
Dense charged full-prompt tokenization, transfer, dense prefill, and dense decode. Compiled
charged full-prompt offset tokenization, hierarchy construction, question features,
top-down traversal, exact-text expansion/stitching, compact tokenization, transfer, dense
prefill, and dense decode.
**Number:** Dense retained **25/25 exact**. The live compiled path achieved **23/25 exact
(92.0%)**, split **12/14 at 16K** and **11/11 at 32K**. It therefore missed the predeclared
accuracy gate (>=24/25 overall and 11/11 at 32K) by exactly one overall answer while fully
clearing the long-context cell. Direct M=4 evidence-block recall was **21/25 (84.0%)**;
radius-1 expansion raised evidence recall to **24/25 (96.0%)**. The aggregate payload does
not contain the case-level intersection, so it cannot establish whether the one
expanded-evidence miss was one of the two generation failures, or name the other failure.

Median compiled input was **1,761 tokens**, or **8.454%** of the original prompt. Summed
per-case median fully charged request time was **75.2652s dense** versus **10.0946s
compiled**, a **7.45596x sum-weighted speedup**. Median request time was 1.96549s dense
versus 0.397922s compiled; median model-prefill component was 1.68263s versus 0.120294s.
By length, the sum-weighted speedup was **5.01714x at 16K** (26.5320s / 5.28826s) and
**10.13929x at 32K** (48.7333s / 4.80638s). The speed gate passed; the combined
deployability gate failed only because the accuracy gate failed.

Mechanistically, the locked selector has no retrieval slack: beam width equals final output
width (beam=M=4). A max/union parent score is an admissible but loose lexical upper bound:
different descendant blocks can collectively contain different question features and make
their parent appear strongly relevant even when no leaf contains the complete evidence
phrase. Retaining only four parents can therefore prune the true leaf before final scoring.
The small D=512 hash can add collision-based false matches, and exact subword overlap cannot
distinguish affirmative evidence from lexically similar negated or provisional distractors.
Radius expansion repaired three of four direct block misses, demonstrating that block
boundaries are a secondary issue, but it cannot recover a branch pruned farther away.

The supplied decision is preserved at
`eval/pre_qwen_runs/pre_qwen_b9f3748f4287/decision.user_supplied.json`; Drive artifacts are
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_b9f3748f4287/pre_qwen_e2e_25.json`,
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_b9f3748f4287/decision.json`,
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_b9f3748f4287.zip`, and stable pointer
`MyDrive/SPRUCE_COLAB/outputs/pre_qwen_latest.json`. The browser connection opened, but the
available connector exposes no Drive download or cell execution, so the case-level report
was not independently copied locally.
**Conclusion:** The implementation is not a general failure: it demonstrates a large,
fully charged end-to-end speed signal and perfect 32K accuracy. It fails the strict product
gate by one answer because selection/read robustness is one case below the frozen threshold.
Do not tune beam or D on these 25 cases. First retrieve the case-level report for clean
failure attribution; then develop a wider candidate beam plus exact leaf/paragraph reranking
on separate prompts. With a 7.46x timing margin, visiting more than four candidates is the
highest-probability repair and is unlikely to erase the speed advantage, but it must be
measured rather than assumed.

## 2026-07-27 — Live pre-Qwen selector, charged benchmark, and Colab artifact verified
**Question:** Is the tokenizer-only hierarchy integrated with exact-text compilation, covered
by correctness tests, exposed through a reproducible fully charged benchmark, and packaged
for the frozen 25-case Colab decision run?
**Config:** Implementation/packaging verification only; no successful dense 16K/32K model
comparison was produced in this step. Added `selector/pre_qwen.py`,
`interfaces/pre_qwen_selector_spec.md`,
`benchmarks/benchmark_pre_qwen_e2e.py`, focused selector/aggregate tests, reusable-layout
compilation, and `colab/run_pre_qwen_e2e.ipynb`. The benchmark alternates dense/compiled
order across repeats, reports per-component medians and outer request wall time, rebuilds the
index on every compiled request, and defines the primary speed statistic as summed per-case
median dense request time divided by summed per-case median compiled request time. The Colab
configuration is locked at D=512, M=4, beam=4, radix 2, radius 1, paragraph boundaries,
three repeats, FP16, and 32 decode tokens. It requires an L4/A100-class GPU with at least
20GiB and the exact 25-prompt Drive manifest.
**Number:** Full repository verification passed **197 tests**, skipped 13 opt-in/platform
tests, and retained the one pre-existing `test_topk_loss.py` scalar-conversion warning.
Focused verification with opt-in real-tokenizer integration passed **16/16**; the only
additional warnings were three SWIG deprecation messages at interpreter shutdown.
`colab/run_pre_qwen_e2e.ipynb` parses as nbformat 4 with 11 cells and 7 code cells; the
benchmark CLI parses; Python compileall and `git diff --check` passed.

The refreshed `spruce_colab_train_source.zip` contains **202 entries**, is **7,330,445
bytes**, and has SHA-256
`B9F3748F42875308B9FE13EA15619DDED42CE4237E56378168D6BDDCBD90E7C2`.
All seven required pre-Qwen selector/spec/benchmark/test/notebook entries were independently
verified inside the archive, with zero `__pycache__` or `.pyc` entries. The recoverable
previous archive is
`spruce_colab_train_source.pre_rebuild_20260727_235802.zip`. `LOG.md` remains deliberately
excluded so recording the archive's own hash is not self-referential.
**Conclusion:** The implementation and reproducible Colab decision artifact are ready. The
only remaining work for this request is the actual >=20GiB frozen run. Until its Drive
decision reports both the accuracy and speed gates, the live compiler is a promising
selection result—not a deployable performance claim.

## 2026-07-27 — Fully charged benchmark local smoke reaches the known 8GB ceiling
**Question:** Can one matched 16K dense-versus-compiled request be executed locally as a
runtime smoke before the frozen Colab evaluation?
**Config:** `benchmarks/benchmark_pre_qwen_e2e.py` on the RTX 4070 Laptop GPU (8GB),
Qwen2.5-Coder-1.5B-Instruct FP16, first dense-accepted 16K prompt, one repeat, eight
generated tokens, locked pre-Qwen D=512 / M=4 / beam=4 / radius-1 paragraph configuration.
This was a plumbing smoke only; it was not the frozen 25-case evaluation.
**Number:** Model load and backend warm-up completed. The first full-dense 16K prefill then
failed before producing an answer or JSON report: PyTorch requested an additional 11.99GiB
while only 2.04GiB of the 8.00GiB device remained free (4.55GiB allocated and 0.285GiB
reserved but unallocated at failure). No accuracy or latency number was produced.
**Conclusion:** The laptop cannot run this matched full-dense control, consistent with the
standing separation between 8GB development and long-context benchmark hardware. This is
not evidence for or against the compiler. Keep the Colab notebook's >=20GiB L4/A100 guard
and run the complete matched comparison there; do not weaken the dense baseline to fit 8GB.

## 2026-07-27 — Cheap pre-Qwen lexical hierarchy locks at D=512 / beam=4
**Question:** Can the saved Q/K selector-feature dependency be replaced by a live,
query-independent document representation that is cheap to construct before any Qwen
forward, while retaining the evidence locations needed by the exact-text compiler?
**Config:** Laptop development and held-out selection-only diagnostics; no Qwen model forward
and no generation accuracy or GPU speed claim. The new path in `selector/pre_qwen.py`
tokenizes the exact stored prompt with offsets, clips to the source-document token range,
hashes unigram and adjacent-token IDs into a fixed-width Boolean block sketch, computes
document frequency over 64-token leaves, and constructs a leaf-first radix-2 hierarchy by
Boolean max/union. At request time, the question is hashed through the same mapping and
weighted by squared block IDF; top-down traversal keeps a fixed beam and returns M=4 exact
source blocks. Parent union preserves rare lexical cues instead of averaging them away.
Stable node IDs use leaf-first level offsets. This path has no Qwen hidden states, Q/K,
selector checkpoint, learned encoder, or external embedding model.

Configuration selection used only the four predeclared `natural_train.json` development
cases (`reserve_cedar_spring`, `reading_room_september_1992`,
`planning_alder_framework`, `compressor_alloy_t19`), each at 16K and 32K, depth 0.5 and
seed 20260725. The sweep was D={512,1024,2048,4096} and beam={4,8,16,32}, with fixed M=4,
radix 2, and the already-locked compiler radius of one block. The selection rule was:
maximize expanded evidence-block recall, then choose the smallest D and beam. After locking
D=512 / beam=4, the configuration was run once without tuning on the 60 dense-accepted
natural held-out prompts available in
`benchmarks/outputs/screen outputs/accepted.json` (33 at 16K, 27 at 32K).
**Number:** Every one of the 16 development configurations recovered the evidence directly
on **8/8** prompts and after radius expansion on **8/8**. The tie rule therefore selected
the smallest setting, **D=512 and beam=4**. At this setting the maximum visited-node count
was 63; development median index construction was 0.00941s and median traversal was 0.00115s
on the laptop CPU. Those timings are single-pass engineering diagnostics, not benchmark
numbers.

On the locked 60-prompt held-out selection diagnostic, direct M=4 evidence recall was
**58/60 (96.67%)** and radius-1 expanded recall was **60/60 (100%)**. The 16K split was
31/33 direct and 33/33 expanded; the 32K split was 27/27 direct and 27/27 expanded.
Median full-prompt offset tokenization/layout time was 0.06541s, and median hierarchy
construction plus question encoding and traversal was 0.006953s on the RTX 4070 Laptop
host. No generation was run, so these selection recalls cannot be quoted as answer accuracy.
The exact frozen 25-case manifest on Drive was not touched during tuning and remains the
single final accuracy/speed evaluation.
**Conclusion:** A genuinely live pre-Qwen selector input path now exists and its cheapest
tested setting preserves every held-out evidence location after the compiler's already-fixed
one-block repair. This removes the saved-feature blocker at the representation level.
Deployability is still unproven until the manifest-complete Colab run charges full-prompt
tokenization, hierarchy construction, traversal, stitching, compact tokenization, transfer,
dense prefill, and decode against the matched full-dense path on the same GPU.

## 2026-07-27 — Evidence compiler clears the complete 25-case accuracy gate
**Question:** Does the locked exact-text evidence compiler preserve the strong local result
on the manifest-complete 25 dense-accepted natural prompts, including all eleven 32K cases,
without tuning its packet format or selector budget on that set?
**Config:** Colab run `evidence_compiler_dc8ee8103764`, completed
`2026-07-28T03:35:12.211235+00:00`, from source archive SHA-256
`DC8EE8103764C0F57A8739F516444F9480DC4BDCB22A6E7682A03802D6FFA300`.
Qwen2.5-Coder-1.5B-Instruct and the selector remained frozen. The configuration was locked
from the preceding 24-case laptop diagnostic: M=4 candidate blocks, one-block expansion,
paragraph-boundary repair, radix-2 beam-8 traversal, selector layer chunk 4, CPU selector,
32 generated tokens, one repeat. The accepted manifest supplied 25 exact stored prompts:
14 at 16K and 11 at 32K. Each selected location was mapped back to exact original text,
stitched in source order with provenance labels, and re-encoded through ordinary dense SDPA.
No sparse hidden states, sparse K/V, omitted-region summaries, paraphrasing, or learned
compressor were used.

The three predeclared modes shared the compiler and dense reader. `oracle` used the known
evidence block and is attribution-only. `flat` scored every document leaf at the reader row
and remains an O(n) ceiling. `tree` used the recursive SPRUCE traversal and reranked only its
bounded final-reader leaf union. Selector inputs for flat/tree were saved Q/K features
produced by teacher extraction; feature extraction was not included in the reported prefill
time.
**Number:** Oracle, flat, and tree each achieved **24/25 exact (96.0%)**. Every mode split
**13/14 at 16K (92.86%)** and **11/11 at 32K (100%)**. The locked tree therefore clears the
predeclared accuracy gate of at least 24/25 overall and 11/11 at 32K. Oracle evidence recall
was 25/25; flat and tree direct and radius-expanded evidence recall were each **24/25
(96.0%)**. Matching oracle/tree exact counts despite different evidence recall means the
aggregate decision alone cannot prove they failed on the same case; use the case-level Drive
report before attributing the remaining error.

Median compiled prompt size was **590 tokens (2.987% of original)** for oracle and **1,358
tokens (5.727%)** for flat/tree. Median dense compiled-prefill time on the Colab runtime was
0.04054s oracle, 0.08896s flat, and 0.08664s tree. These are compiled-reader component times,
not end-to-end TTFT: the run used one repeat, selector device was CPU, and live feature
construction plus complete selection/stitching cost is absent. The notebook correctly records
`accuracy_gate_passed=true`, `deployability_gate_passed=false`, with the blocker that no cheap
live pre-Qwen feature path or matched end-to-end speed measurement exists.

The supplied decision is preserved at
`eval/evidence_compiler_runs/evidence_compiler_dc8ee8103764/decision.user_supplied.json`.
Drive artifacts are
`MyDrive/SPRUCE_COLAB/outputs/evidence_compiler_dc8ee8103764/evidence_compiler_25.json`,
`MyDrive/SPRUCE_COLAB/outputs/evidence_compiler_dc8ee8103764/decision.json`,
`MyDrive/SPRUCE_COLAB/outputs/evidence_compiler_dc8ee8103764.zip`, and stable pointer
`MyDrive/SPRUCE_COLAB/outputs/evidence_compiler_latest.json`. The current connector exposes
no Drive download, so the case-level report and ZIP were not independently copied locally;
provenance is recorded beside the preserved aggregate decision.
**Conclusion:** The exact-text evidence-compiler idea works on the complete accuracy gate.
The frozen recursive selector plus dense compact reread recovers the required 24/25 overall
and all 11 long 32K cases while presenting only 5.73% of the original tokens to Qwen. This
supports the diagnosis that the old failure was primarily sparse representation corruption,
not an inability to locate or read the evidence. It does **not** yet establish SPRUCE as a
deployable or faster method. The active engineering problem is now a cheap pre-Qwen
hierarchical document representation and a matched end-to-end benchmark charging feature
construction, tree selection, text stitching, transfer, dense prefill, and decode.

## 2026-07-27 — Evidence compiler recovers 23/24 with a fresh dense read
**Question:** Is SPRUCE's primary failure that Qwen cannot read selected evidence after it
has been contextualized through the unfamiliar sparse attention graph, and can the same
selected locations work when their original text is stitched into a short coherent prompt
and re-encoded densely?
**Config:** Laptop development diagnostic on the RTX 4070 Laptop GPU (8GB; not paper timing),
Qwen2.5-Coder-1.5B-Instruct FP16 with an ordinary SDPA dense read, frozen backbone and frozen
gate `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`. Added the exact-text evidence
compiler in `interfaces/evidence_compiler.py` and diagnostic
`benchmarks/evaluate_evidence_compiler.py`. Source block IDs are mapped back to the original
prompt text, clipped to the document (never the chat wrapper or question), expanded by one
block, repaired to paragraph boundaries, merged in source order, labeled with original
block/token provenance, and placed before the unchanged question in a fresh dense chat prompt.
No omitted-region summaries, sparse hidden states, or sparse K/V are read.

The configuration was locked after one Gallery 16K smoke case: M=4, block radius=1,
paragraph boundaries, 32 generated tokens, one repeat. It was then run unchanged on all **24
unique dense-accepted natural held-out targets available locally** (13 at 16K, 11 at 32K).
This local directory is one target short of the later Colab-re-screened 25-case manifest, so
this is a 24-case diagnostic rather than the complete gate. The set has also already been
used by prior SPRUCE work and is not an untouched claim set.

Three selection controls used the same compiler and dense reader:

- `oracle`: the known evidence block, attribution only.
- `flat`: the frozen gate scores every document leaf at the final reader row and keeps M=4;
  this is an O(n) selection ceiling, not the recursive complexity claim.
- `tree`: radix-2 beam-8 SPRUCE traversal, union the final-reader leaves across layers/groups,
  then score only that bounded union and keep M=4.

Selector Q/K inputs came from saved teacher-extraction features. Feature load is measured, but
the original feature-extraction pass is absent, so none of these totals is deployable TTFT.
Artifact:
`benchmarks/outputs/evidence_compiler/frozen25_M4_R1_paragraph.json`; console record:
`benchmarks/outputs/evidence_compiler/frozen25_M4_R1_paragraph.log`.
**Number:** All three modes achieved **23/24 exact (95.83%)**, split **12/13 at 16K
(92.31%)** and **11/11 at 32K (100%)**. This is a large change from the prior sparse-prefill
results on the same task family: exact-only routing reached 16/25 and deterministic residual
summaries reached 17/25, with only 5/11 at 32K.

The error attribution is clean. Oracle evidence was present 24/24 but missed one 16K Atlas
case by generating the incomplete path `/atlas.chk` rather than the accepted full path; the
M=4 packets supplied additional context and answered that case correctly. Flat and tree
candidate packets both missed only Aquifer 16K depth 0.1. In that case neither selected nor
expanded blocks contained the evidence, and the answer was the nearby distractor `Lake
Calden`. Conditional on selecting the evidence block, flat/tree dense packets were **23/23
exact**. Flat and recursive tree produced the same final M=4 blocks on **24/24 cases**. The
tree's pre-rerank final-reader union ranged from 49 to 112 unique blocks.

Median compiled prompt size was **590.5 tokens (2.971% of the original prompt)** for oracle
and **1,343 tokens (5.434%)** for flat/tree. Median dense compiled-prefill time was 0.223s
oracle, 0.295s flat, and 0.277s tree; median peak allocated memory was 3.176GB oracle and
3.390GB flat/tree. Median CPU selection time was 0.191s for the flat ceiling and 1.540s for
the current unoptimized recursive tree; median compilation was about 0.10s. These laptop
component timings are diagnostic only: the saved-feature path omits live feature extraction,
the run used one repeat, and no matched full-dense end-to-end comparison was made.

The implementation is covered by 13 passing lightweight compiler/benchmark tests plus one
passing Qwen-tokenizer integration test. Tests prove stable/valid source IDs, document-only
clipping, paragraph-boundary repair, source-order preservation, adjacent-span merging,
provenance labels, no silent span dropping, rejection of wrapper-only selections, flat/tree
document filtering, bounded tree candidates, aggregation, and CLI parsing.

Full repository verification after integration passed **192 tests**, skipped 12 opt-in/platform
tests, and retained the one pre-existing `test_topk_loss.py` scalar-conversion warning.
`colab/run_evidence_compiler_gate.ipynb` parses as nbformat 4 with 13 cells and 8 code cells;
the benchmark CLI parses; and `git diff --check` passed. The refreshed
`spruce_colab_train_source.zip` contains **193 entries**, is **7,311,357 bytes**, and has
SHA-256 `DC8EE8103764C0F57A8739F516444F9480DC4BDCB22A6E7682A03802D6FFA300`.
The rebuild script verified the compiler interface, selector ranker, benchmark, tests, and
notebook inside the archive. The recoverable previous archive is
`spruce_colab_train_source.pre_rebuild_20260727_223828.zip`. `LOG.md` is deliberately
excluded from the ZIP because recording the ZIP's own hash inside it would be
self-referential; the workspace copy remains the record of record.
**Conclusion:** The user's read-distribution hypothesis is strongly supported. When the
selected locations are returned to exact source text and re-encoded together through normal
dense attention, the previous 32K wall disappears on all 11 locally available cases. The
remaining selector-packet failure is selection, not reading: evidence absent implies failure;
evidence present gives 23/23. This does **not** yet establish a deployable speedup or preserve
the original sparse-attention conversion claim because the diagnostic consumes pre-extracted
Q/K features, and the flat ceiling is O(n). Next: run the manifest-complete Colab 25-case
version, then replace offline Q/K features with a genuinely cheap pre-Qwen hierarchical
document selector before any toolkit or paper claim.

## 2026-07-27 — Deterministic residual summaries fail the decision ladder
**Question:** Does complete residual coverage with deterministic live mean-K/V prototypes
repair the frozen Qwen2.5-Coder-1.5B sparse-prefill representation enough to clear the
plug-and-play gate, or at least show the predeclared signal required to justify a learned
compressor?
**Config:** Colab run `residual_4ab66445f234`, completed
`2026-07-28T01:08:20.454012+00:00`, from source archive SHA-256
`4AB66445F2347BA8D934DD712AB9288053A3C37297888741207148462A9ABF1D`.
The backbone and selector remained frozen. Prototype count P was tuned only on four
`natural_train.json` development cases
(`compressor_alloy_t19`, `planning_alder_framework`,
`reading_room_september_1992`, `reserve_cedar_spring`) at 16,384 and 32,768 tokens,
disjoint from the frozen 25-prompt evaluation set. P={1,2,4} used the predeclared rule:
select the smallest P whose sampled attention-output RMSE is no more than 1.05 times the
best P. The selected P was then evaluated exactly once on the frozen 25 prompts. Exact
routes used the existing dense-candidates M=4/K=10/W=0/S=0 policy; residual summaries
covered every omitted causal region. The charged-attention fraction counts one entry per
valid prototype and 64 entries per exact block. This run did not implement or time a Triton
summary loop.
**Number:** Development attention-output RMSE was **0.2856209151 for P=1**,
**0.2851001494 for P=2**, and **0.2842079434 for P=4**. The within-5% threshold was
0.2984183406, so all three settings qualified and the rule correctly selected **P=1**;
P=1 was only about **0.50%** worse than the best P=4. Despite complete residual coverage,
final-hidden relative RMSE versus exact-only sparse attention became worse, not better:
the reported reduction was **-13.985% at 16K** and **-5.614% at 32K**.

On the frozen set, P=1 produced **17/25 exact overall (68.0%)**, split
**12/14 at 16K (85.7%)** and **5/11 at 32K (45.5%)**. Relative to the prior best exact-only
dense-candidates result (16/25, 12/14 at 16K, 4/11 at 32K), deterministic summaries recovered
only one additional 32K case. The median charged-attention fraction was
**0.0754183**, comfortably below the <=0.25 cost ceiling, but accuracy missed the
plug-and-play requirements by **7 cases overall** and **6 cases at 32K**. The deterministic
accuracy-and-cost signal is false. Learned-compressor eligibility is also false: 17/25 is
below the >=20/25 branch, while both length-specific final-hidden improvements are negative
rather than >=30%. No matched live-prefill speedup exists because Triton summary support was
not built; therefore the full product gate is false as well.

The exact decision payload supplied after the run is preserved at
`eval/residual_summary_runs/residual_4ab66445f234/decision.user_supplied.json`. Drive paths
reported by the notebook are
`MyDrive/SPRUCE_COLAB/outputs/residual_4ab66445f234/decision.json`,
`MyDrive/SPRUCE_COLAB/outputs/residual_4ab66445f234.zip`, and the stable pointer
`MyDrive/SPRUCE_COLAB/outputs/residual_summary_latest.json`. The current connector exposed
Colab session connection but no Drive download/cell-execution operation, and no local Drive
sync or authenticated Drive CLI was present, so the full ZIP was not independently copied
or bundle-verified locally; provenance is recorded beside the preserved payload.
**Conclusion:** Deterministic residual summaries fail both the plug-and-play gate and the
predeclared learned-compressor eligibility gate. Per the frozen decision ladder, **stop:
do not train the learned compressor and do not implement the Triton summary loop**. Complete
mean summaries marginally improve frozen exactness from 16/25 to 17/25 at very low charged
attention, but they worsen final-hidden fidelity at both lengths and leave the 32K retrieval
wall essentially intact. The next step requires an explicit contribution decision outside
this residual-summary branch, most plausibly scoping automated sparse-in-the-loop adaptation
or pivoting the claim; it is not more P tuning.

## 2026-07-27 — Residual-summary Colab decision notebook and refreshed upload ZIP
**Question:** Can the deterministic residual-summary experiment be launched from the IDE with
the current local source ZIP, run without copying that ZIP to Drive, preserve the frozen-set
decision order, and leave a predictable Drive result pointer for later agent retrieval?
**Config:** Packaging and harness verification only; no Qwen model forward, development RMSE,
frozen accuracy, or latency measurement was run. Added
`colab/run_residual_summary_gate.ipynb`, a 17-cell notebook with 10 code cells. Its first
interaction uses `google.colab.files.upload()` and requires exactly one local ZIP. It hashes
the uploaded bytes, unpacks them into `/content/SPRUCE`, and verifies residual interface,
pooling, attention, diagnostics, live benchmark, and suite markers before using GPU time.
Drive is mounted only after source upload and is used for the existing selector checkpoint
`MyDrive/SPRUCE_COLAB/selector_ckpt/natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`,
the frozen accepted manifest
`MyDrive/SPRUCE_COLAB/screens/natural_heldout_accepted_20260726.json`, and durable outputs.

The notebook predeclares four disjoint `natural_train.json` development cases, lengths
{16,384, 32,768}, depth 0.5, 64 sampled token positions, dense-candidates M=4/K=10/W=0/S=0,
and P={1,2,4}. Added `benchmarks/diagnose_residual_summaries.py`: dense SDPA runs once per
development prompt; the exact-only PyTorch reference and each P setting then compare the same
sampled positions at every attention-module output and decoder-layer output. Squared errors
are accumulated globally rather than averaging per-layer ratios. P selection is frozen to the
smallest P with attention-output RMSE <=1.05 times the best RMSE. Only that P proceeds to one
frozen 25-prompt PyTorch accuracy/cost run. The notebook explicitly leaves the full product
gate false until conditional Triton summary support supplies matched live-prefill speed.

Results are written under
`MyDrive/SPRUCE_COLAB/outputs/residual_<source-hash-prefix>/`. The final cell writes
`decision.json`, packages the full result directory as a Drive ZIP, and updates the stable
pointer `MyDrive/SPRUCE_COLAB/outputs/residual_summary_latest.json` with the run ID, source
SHA-256, decision path, bundle path, and completion time. This lets the connected Colab
session locate and print the result without requiring the user to browse Drive.
**Number:** Full repository verification passed **179 tests**, with **11 opt-in/platform tests
skipped** and the same one pre-existing `test_topk_loss.py` scalar-conversion warning.
Focused residual/diagnostic/suite verification passed 37/37; notebook JSON parses as nbformat
4 with 17 cells and 10 code cells; both diagnostic and existing benchmark CLIs parse; and
`git diff --check` passed. Rebuilt `spruce_colab_train_source.zip` contains **183 entries**,
is **7,289,503 bytes**, and has SHA-256
`4AB66445F2347BA8D934DD712AB9288053A3C37297888741207148462A9ABF1D`. Required notebook,
diagnostic, residual interface/pooling, and test entries were verified by the rebuild script.
The recoverable prior archive is
`spruce_colab_train_source.pre_rebuild_20260727_130949.zip`.
**Conclusion:** Open `colab/run_residual_summary_gate.ipynb` through the connected Colab
session, run all, select the refreshed local ZIP when the first cell asks, and authorize Drive
once in the second setup cell. After those two interactions the run is unattended. Do not
interpret notebook completion as a full plug-and-play pass unless a later Triton run also
clears the >1.0x matched live-prefill gate.

## 2026-07-27 — Deterministic hierarchical residual summaries implemented
**Question:** Can SPRUCE preserve the existing exact `selected_blocks` route while replacing
every omitted causal block with a deterministic, complete tree frontier of compressed live
K/V summaries, without changing the summary-disabled PyTorch reference behavior?
**Config:** Development implementation only; no held-out prompt, model-forward accuracy run,
or paper latency measurement. The Qwen backbone and selector remain frozen. Exact routing
continues to use the unchanged `selected_blocks: int32 [B,L,G,Q,K]` contract. Added
`residual_summary_nodes: int32 [B,L,G,Q,S]`, with stable leaf-first binary-tree IDs, odd-tail
ceil-halving rollup matching `selector.tree`, trailing `-1` padding, and `S` chosen from the
largest complete frontier in the supplied route tensor rather than from a truncation budget.
The deterministic compressor runs independently inside every decoder-layer attention callback
from that layer's live post-RoPE K/V. Each residual node is divided into P={1,2,4} contiguous
token ranges; each nonempty prototype stores FP32-accumulated mean K, mean V, token count, and
`log(token_count)`. Exact-token logits and prototype logits are concatenated before one
softmax. Padding tokens are excluded from means/counts by reducing the 4D additive attention
mask over query rows. The learned compressor and Triton summary loop were deliberately not
implemented because both are conditional on a deterministic held-out accuracy signal.

**Implementation:**
- Added `interfaces/residual_summaries.py` and
  `interfaces/residual_summaries_spec.md`. `build_residual_tree_layout` assigns stable IDs;
  `build_residual_summary_nodes` recursively splits future/partially causal or exact-overlapping
  nodes and emits the maximal complete omitted frontier; and
  `validate_residual_summary_nodes` proves causality, sorted/unique/trailing padding,
  no exact-summary overlap, no summary-summary overlap, complete causal-prefix coverage, and
  maximality.
- Added `sparse/summaries.py`. Vectorized prefix reductions construct every node/prototype
  table for the current decoder layer without materializing token-by-token attention.
  Empty/tail prototypes carry count zero and `-inf` log-count bias and are removed before the
  mixed softmax. `residual_attention_density` charges one entry per valid prototype and
  `block_size` entries per exact block; PAD slots are free.
- Extended `sparse/attention.py` only behind `residual_summaries=True`. The default path keeps
  the prior exact sparse score/value computation. Mean summaries use
  `q·mean(K)/sqrt(d) + log(count)` and concatenate mean V with exact V for the shared
  probability-value product.
- Added shared CLI/config controls in `sparse/config.py`:
  `--residual-summaries`, `--summary-prototypes {1,2,4}`,
  `--summary-mode {mean,learned}`, and `--summary-checkpoint`. The one-prompt replay, live-tree
  benchmark, and route-control suite accept the controls. Triton plus summaries is rejected
  explicitly. Learned mode requires a checkpoint and then stops with the deterministic-gate
  message rather than silently falling back to means.
- Extended live timing honestly. `live_prefill_seconds` now includes selector/tree routing,
  residual tree-layout construction, maximal-frontier construction, CPU/GPU transfer, and the
  model prefill. Per-layer summary construction and mixed attention are both inside the
  measured model-prefill interval. Reports mark this explicitly and include exact-entry,
  prototype-entry, charged-entry, and maximum causal-entry counts.

**Number:** Full repository verification passed **179 tests**, with **11 opt-in/platform tests
skipped** and one pre-existing `test_topk_loss.py` warning about converting a grad tensor to a
Python scalar. `git diff --check` passed. New controls cover tree layouts corresponding to
16K/32K/64K/128K at block size 64 (256/512/1024/2048 leaves), odd leaf counts, duplicated-range
odd-tail ancestors with stable distinct IDs, partial final blocks, padding masks, GQA
(2 query heads per KV head in the mathematical control), FP16, BF16, and P={1,2,4}. The
multiplicity control uses identical keys and non-identical values and reproduces dense causal
attention after summary pooling, which is stronger than the trivial identical-K/V control.
The all-exact route produces `S=0` and retains dense equivalence. Passing a residual tensor
while `residual_summaries=False` is bitwise identical to the previous summary-disabled output.
These are software/mathematical verification results, not held-out retrieval accuracy or
speed measurements.
**Conclusion:** The deterministic correctness-first residual-summary branch is ready for the
predeclared development P sweep and then one frozen 25-case evaluation. Do not quote an
accuracy or speed improvement yet, do not train the learned compressor yet, and do not extend
the Triton kernel yet. Select the smallest P whose development attention-output RMSE is within
5% of the best P; proceed to learned compression only if deterministic summaries reach at
least 20/25 exact or reduce final-hidden relative RMSE by at least 30% at both 16K and 32K.

## 2026-07-27 — SpotAttention-inspired sparse-prefill oracle stops at the cost gate
**Question:** Did the apparently silent model-forward cell in
`run_spotattention_discriminator.ipynb` fail to execute, or did the route preview intentionally
leave no cost-eligible top-p settings to run?
**Config:** Local reconstruction from the same full dense-teacher targets used by the frozen
natural held-out campaign: 24 unique target files plus the historical duplicate Atlas 16K d0.5
target, reproducing the logged 25-case split of 14 at 16K and 11 at 32K. Qwen2.5-Coder-1.5B
teacher mass; block size 64; exact dense-teacher dual top-p routes at p={0.7,0.8,0.9}; 128 sink
tokens, 256 recency tokens, and 512 minimum selected tokens. Predeclared cost gate: median
charged attention fraction <=0.25. This was a route-only CPU reconstruction; no model forward
or accuracy measurement was run.
**Number:** Median charged attention fraction was 0.366803 at p=0.7, 0.488349 at p=0.8, and
0.663797 at p=0.9 (charged sparsity 0.633197, 0.511651, and 0.336203). Maximum packed route
width was 242, 312, and 397 blocks respectively. Even the minimum per-case charged fraction at
p=0.7 was 0.287414, above the 0.25 ceiling. Eligible settings: 0/3.
**Conclusion:** The third-last cell did run logically, but its loop body was skipped because
`ELIGIBLE_TOP_P` was empty. This branch stops on cost before accuracy: paper-equivalent
SpotAttention-inspired dual-top-p routing is too dense for the declared sparse-prefill product
target, so running the expensive model forwards would violate the experiment's guardrail and
cannot produce a passing result.

## 2026-07-27 — SpotAttention scope correction and sparse-prefill oracle notebook
**Question:** Can SpotAttention serve as the decisive frozen-backbone sparse-prefill baseline, and what is the cheapest valid experiment after the dense-band gate failed?
**Config:** Re-read SpotAttention arXiv:2606.22874v1 rather than reconstructing its mechanism from memory. The paper trains a 4-head, 128-dim SparseKL indexer at block size 16 with a frozen backbone and dual top-p routing; defaults are p={0.7,0.8,0.9}, 128 sink tokens, 256 recency tokens, and a 512-token minimum. Its published downstream long-context accuracy protocol explicitly keeps prefill dense and applies sparse selection only at decode, so those numbers are not evidence for a plug-and-play sparse-prefill conversion. Added `teacher_dual_top_p_routes`: an explicitly nondeployable prefill oracle using the exact dense-teacher head-averaged distribution, paper-equivalent token budgets mapped to SPRUCE block size 64, variable-width PAD-packed routes, and honest charged-density accounting. Added `colab/run_spotattention_discriminator.ipynb`; it previews route cost on the frozen 25-prompt set and runs only p settings whose median charged attention fraction is <=0.25, with oracle gates of >=24/25 overall, 11/11 at 32K, and sum kernel-prefill speedup >1.0x. No model forward or held-out accuracy measurement was run locally.
**Number:** Focused route/live-benchmark/sparse-kernel tests passed 27/27 with 10 CUDA-only cases skipped locally; notebook JSON validates; `git diff --check` reports no whitespace errors. Rebuilt `spruce_colab_train_source.zip` contains 173 entries, is 7,263,560 bytes, and has SHA-256 `5520508A0529B8CBE41B92CB88F757B48FB1D8D1E1BE2D7EF23107659E7F50F9`.
**Conclusion:** Do not cite SpotAttention's accuracy as a sparse-prefill result. Run the oracle notebook first: failure closes this dynamic prefill construction without another selector-training run; success justifies implementing the real SparseKL indexer and then measuring live selector cost. This preserves the frozen-backbone plug-and-play question without conflating it with SpotAttention's dense-prefill/sparse-decode result.

## 2026-07-27 — Four-layer dense-band plug-and-play gate fails
**Question:** Can one charged four-layer dense SDPA band restore the best zero-training sparse policy from 16/25 to the required >=24/25 without surrendering the efficiency gain?
**Config:** Colab L4; Qwen2.5-Coder-1.5B-Instruct FP16; frozen backbone; full 25-prompt dense-accepted natural held-out set; best existing `dense-candidates` M=4, K=10, W=0, S=0 routing; true hybrid dispatch with all other layers retaining compact Triton routes; paired dense timing; bands early={0,1,2,3}, middle={12,13,14,15}, late={24,25,26,27}. Predeclared gate: >=24/25 overall, 11/11 at 32K, <=4 dense layers, median charged attention fraction <=0.25, and sum-weighted live-prefill speedup >1.0x.
**Number:** Early band was best at 17/25 exact: 12/14 at 16K and 5/11 at 32K, charged attention fraction 0.206574 (sparsity 0.793426), sum live-prefill speedup 1.144373x. Late band scored 16/25: 12/14 and 4/11, charged fraction 0.206289 (sparsity 0.793711), speedup 1.203250x. Middle band scored 15/25: 11/14 and 4/11, charged fraction 0.206467 (sparsity 0.793533), speedup 1.219308x. All 3/3 passed the cost gate and 0/3 passed the accuracy gate; best improvement over the 16/25 no-band baseline was +1 prompt.
**Conclusion:** A four-layer dense band does not close the retrieval gap; the best result remains seven prompts short of the >=24/25 gate and six 32K prompts short of 11/11. Early placement helps slightly, but the effect is too small to justify another interpolation ladder toward dense. Strict zero-backbone-training plug-and-play is not supported by the current SPRUCE sparse construction. Do not proceed to LoRA automatically: the next decision is an explicit method redesign versus relaxing the product requirement to an automated adaptation conversion.

## 2026-07-27 — Dense-layer-band hybrid and Colab decision notebook
**Question:** Can the final zero-training rung test dense decoder-layer bands without widening every sparse layer's route tensor or hiding dense work from the reported sparsity?
**Config:** Added optional `--dense-layers` hybrid dispatch: named decoder layers use causal dense SDPA while all other layers retain compact K-wide Triton routes. Added block-level charged attention fraction/sparsity accounting, propagated it through route-control case/summary CSVs, and fixed the PyTorch reference to accumulate QK scores and the probability-value product in FP32 before casting back to model dtype. Added `colab/run_dense_layer_bands.ipynb`, predeclaring three equal four-layer bands (0-3, 12-15, 24-27) on the best `dense-candidates` M=4/K=10 policy, a maximum median charged-attention fraction of 0.25, and the accuracy gate of >=24/25 overall plus 11/11 at 32K. No model forward was run.
**Number:** Focused reference/route/hybrid/suite tests passed 33/33 with 10 Triton-CUDA cases skipped locally; both benchmark CLIs parse successfully; notebook JSON validates. Rebuilt `spruce_colab_train_source.zip` contains 172 entries, including the new notebook and all hybrid sources, is 7,255,601 bytes, and has SHA-256 `BF247D143F667ACAEAB7923576866DF63B7240A43DAA0F26541C121F25A1B3DA`; zero staging directories remain.
**Conclusion:** Upload the rebuilt archive and run `colab/run_dense_layer_bands.ipynb`. Its result is the decision point for strict zero-training plug-and-play; it does not start LoRA automatically if all three bands miss.

## 2026-07-27 — All-causal-block full-path equivalence at 16K and 32K
**Question:** When every causal block is selected, does the complete SPRUCE attention path reproduce dense Qwen at real context lengths, separating sparse-path correctness from selector/pruning quality?
**Config:** Colab NVIDIA L4 (22.03GiB); Python 3.12.13; PyTorch 2.11.0+cu128; Transformers 5.13.1; Triton 3.6.0; Qwen2.5-Coder-1.5B-Instruct FP16; native RoPE; block size 64; `single_head` Triton kernel; one matched dense-accepted held-out natural prompt (`aquifer_lake_orison`, depth 0.5, seed 20260725) at requested lengths 16,384 and 32,768; every causal block selected for all 28 layers and both KV groups. Predeclared acceptance: finite outputs, same top-1, top-10 overlap >=0.9, logits cosine >=0.999, final-hidden relative RMSE <=0.02.
**Number:** Triton passed 2/2 lengths. At 16K (16,383 actual tokens), top-1 matched (`Lake`), top-10 overlap 1.0, logits cosine 0.99999309, logits relative RMSE 0.00144616, and final-hidden relative RMSE 0.00183988. At 32K (32,768 tokens), top-1 matched (`Lake`), top-10 overlap 1.0, logits cosine 0.99999303, logits relative RMSE 0.00155127, and final-hidden relative RMSE 0.00183441. Every captured Triton layer was finite; layer-0 relative RMSE was 0.00035081 at 16K and 0.00037136 at 32K. Dense/Triton forward times were 3.342/7.410s at 16K and 6.027/9.722s at 32K; these are correctness-run timings, not paper speed claims. The PyTorch reference failed 0/2: outputs were non-finite beginning at layer 0 at both lengths, with top-1 changing to token ID 0 and all error metrics becoming NaN.
**Conclusion:** The production Triton all-block path is correct at both real lengths on this held-out prompt, so route construction/consumption, RoPE, masking, and the Triton kernel do not explain the natural-prompt accuracy gap when pruning is removed. The overall report is false only because the PyTorch reference has a separate long-context numerical defect. Its FP16 QK matmul occurs before FP32 softmax, unlike the Triton kernel's FP32 online-softmax state; confirm and fix score accumulation before treating earlier PyTorch-reference diagnostics as kernel-independent evidence. This is a one-prompt equivalence gate, not a full accuracy evaluation.

## 2026-07-27 — Rebuilt Colab source archive for the all-blocks equivalence gate
**Question:** Does the Colab source archive contain the exact runnable notebook and source needed for the 16K/32K dense-vs-all-causal-block correctness test?
**Config:** Rebuilt `spruce_colab_train_source.zip` from the current workspace with `colab/rebuild_source_archive.ps1`; preserved the prior archive's benchmark evidence, refreshed all source/test trees, and included `colab/run_all_blocks_equivalence.ipynb`, `benchmarks/all_blocks_equivalence.py`, the updated PyTorch sparse reference, and its focused tests. No model forward or accuracy experiment was run.
**Number:** 171 archive entries; 7,249,555 bytes; SHA-256 `9FD2B8E808DC9B8517AD8712A1488B015980F9BD4B65FC59CB9BB806C35F4389`; 5/5 required equivalence files verified present; zero leftover staging directories.
**Conclusion:** The refreshed archive is ready to upload to `SPRUCE_COLAB` and run through `colab/run_all_blocks_equivalence.ipynb`; this entry is packaging verification only, not an equivalence result.

## Status snapshot — 2026-07-27

**Stage (current after the all-block and dense-band gates):** The production Triton path is
numerically equivalent to dense when every causal block is selected at both 16K and 32K, so
integration, causal masking, RoPE, and the kernel are not the source of the natural-prompt
gap. The final predeclared zero-training rung also failed: equal-cost four-layer dense bands
scored early 17/25, late 16/25, and middle 15/25 against the required >=24/25; all retained
79.3% charged sparsity and >1.14x sum-weighted live-prefill speedup. The best band recovered
only one case, leaving the method seven cases short overall and six short at 32K. The current
SPRUCE sparse construction therefore does not support the strict plug-and-play claim. LoRA is
not an automatic next step: choose explicitly between redesigning the sparse construction
(while keeping the backbone frozen) and redefining conversion as automated adaptation. The
SpotAttention-inspired frozen-backbone prefill oracle is now also closed on its predeclared
cost gate: exact dense-teacher dual-top-p routes charge median attention fractions of 0.367,
0.488, and 0.664 for p={0.7,0.8,0.9}, so none qualifies for the <=0.25 ceiling and no accuracy
forward is justified.

**Stage (current after the residual-summary decision run):** The frozen-backbone hierarchical
residual-summary redesign is empirically closed. The predeclared development rule selected
P=1 (attention-output RMSE 0.285621, within 0.50% of P=4), then the one allowed frozen-set
evaluation scored 17/25 overall, 12/14 at 16K, and 5/11 at 32K. Median charged attention was
only 0.0754, but the method missed the accuracy gates by seven cases overall and six at 32K.
More importantly, final-hidden relative RMSE was worse than exact-only sparse attention at
both lengths (-13.99% and -5.61% reported reduction). This fails both routes into learned
compression (neither >=20/25 nor >=30% hidden-RMSE improvement at both lengths). Per the
decision ladder, learned compression and the Triton summary loop must not be built. The
frozen-backbone plug-and-play construction claim is not supported; the next contribution
choice is an explicit adaptation/pivot decision, not another summary hyperparameter sweep.

**Stage (current after the complete evidence-compiler gate):** The frozen-backbone
exact-text compiler clears its accuracy target. Use SPRUCE only to locate source regions,
return to their original text, stitch coherent paragraph spans, and re-encode the compact
packet with ordinary dense Qwen attention. On the manifest-complete 25 cases, the locked
M=4/radius-1/paragraph recursive-tree path scores 24/25 overall and 11/11 at 32K while
presenting a median 1,358 tokens (5.73% of the original) to Qwen. Flat and tree match exactly,
and both evidence recall and exactness are 24/25. This strongly supports sparse representation
corruption as the old failure and makes evidence compilation the active branch ahead of
backbone adaptation. It does **not** clear the product/speed gate: selector inputs are still
pre-extracted Q/K features, the reported 0.0866s is only dense compiled prefill, and no matched
end-to-end measurement charges live feature construction, selection, stitching, transfer, and
decode. The active gate is now deployability through a cheap pre-Qwen hierarchical document
representation.

**Stage (current after the beam-16 engineering follow-up):** The saved-Q/K dependency is
removed and the complete live path is fast. D=512 / M=4 / beam=16 scores 25/25 overall,
14/14 at 16K, and 11/11 at 32K while achieving 7.521x sum-weighted request speedup with
every preprocessing, transfer, prefill, and decode cost charged. Expanded evidence recall is
25/25. This repairs the beam-4 result (23/25 at 7.456x) with traversal slack alone and no
selector training. Because beam 16 was tested after the 25 cases were opened, treat it as an
engineering result, not a new untouched gate. Freeze the configuration and move to ~100 new
untouched prompts before a paper/toolkit accuracy claim.

**Stage (current after the 16K–128K paper-suite implementation):** The frozen beam-16
configuration is now packaged for its first unscreened scaling run on a newly sealed bank.
The matrix has 288 paired rows from 12 independent semantic cases, three evidence depths,
and eight exact 16K increments through 128K under matched static YaRN. It checkpoints every
case to Drive and emits paired statistics, a case-clustered bootstrap, five CSV tables, and
12 PNG + 12 vector-PDF figures. No model result exists yet. Treat the 288 rows as repeated
length/depth observations from 12 semantic clusters, not as 288 independent task concepts.
The active next step is the Colab run; do not change beam, D, M, radius, packet formatting,
the prompt bank, or the declared analysis after seeing partial results.

**Stage (current after the completed unscreened 16K–128K run):** The context compiler
achieves its intended behavior on the sealed scaling suite. It scores 237/288 (82.29%)
against dense 192/288 (66.67%), wins 63 paired rows while losing 18, and is faster at every
length (2.49x at 16K to 14.08x at 128K; 9.58x sum-weighted overall). At 64K–128K it is
148/180 versus dense 113/180. The improvement is strongest when evidence is farthest from
the final query: depth-0.1 accuracy is 81/96 versus 46/96 overall and 50/60 versus 24/60
at 64K–128K. The mechanism is now explicit: all 237 compiler successes include expanded
evidence, all 32 expanded-evidence misses fail, and all 18 dense-only rows are evidence
misses. Exact-text recompilation solves representation/readability; remaining accuracy work
is evidence-location recall and independent semantic coverage.

The systems result is strong on one A100-SXM4-40GB: median compact context is 1,849.5 tokens,
median allocated GPU memory stays near 3.27 GB instead of growing from 4.66 to 15.58 GB, and
fully charged selection/compilation plus model execution remains faster at every tested
length. The paper guardrail remains open: a 12-semantic-case clustered bootstrap gives
[-5.56, +34.72] points even though row-level McNemar p=5.20e-7. Do not describe this as
general superiority. Freeze the opened bank and expand validation with new independent
semantic cases and external tasks.

**Stage:** Root cause of the natural-retrieval failure is now measured, and it is NOT fixable by selector loss tuning. Three findings (2026-07-26 diagnostics, PyTorch reference backend on the laptop, kernel-independent): (1) the block-pooled, group-averaged teacher targets erase the evidence — unconditional teacher top-8 eligibility averages 0.64 per layer-group (0.00 worst case), and the teacher's own top-8 routes generate a distractor; the real dense retrieval signal lives in a few question-row TOKENS (mass up to 0.32 at L24) that query-side block pooling destroys. (2) Evidence access is not sufficient: forcing the evidence block into every route (verified hit 1.0) changes nothing. (3) At 16K a dense reader row (body still K=10 sparse) recovers the exact answer at ~0.4% extra cost; at 32K even dense-reader + K=64 fails — body sparsification corrupts document-side representations and close distractors win. Follow-up controls attributed the 32K failures: densifying the evidence block's own query row (plus dense reader) restores Observatory exactly, so the mechanism is evidence-K/V corruption under sparse prefill, repairable at O(L) per densified row; Atlas alone resists all partial densification and needs its own token-level audit. Deployable validation done: `--route-mode dense-candidates` (gate-scored top-8 reader-row blocks densified, no oracle) recovers 2/3 exact at ~2% extra prefill — the gate already ranks the evidence first on all three prompts. Atlas alone still fails every partial densification despite near-saturated dense evidence attention (0.999); next is a sparse-vs-dense differential audit on its layers. Colab Triton parity + full held-out validation of dense-candidates (`benchmarks/run_route_control_suite.py`) remain before paper claims or selector retraining. New tooling landed on branch `selector-diagnostics`: `--route-mode {learned,oracle-needle,teacher-top8,dense-reader}` and `--backend pytorch`/`--skip-dense` in the live benchmark, `scripts/audit_dense_attention.py`, `--needle-eligibility always` in the trainer (implemented, untrained), and union/teacher-ceiling metrics in trainer and natural-gate eval output.

**Superseded by the 2026-07-26/27 Colab runs — kept for the diagnostic history, but read the
newest entries first.** Three claims above are now known to be wrong or incomplete: the
2/3 dense-candidates result was an anecdote (real rate on 25 held-out targets is 16/25); "the
gate already ranks the evidence first" holds only at M>=8 (recall is 0.52 at M=1, 0.96 at M=4);
and Atlas is not a unique outlier but an ordinary case of the depth pattern. Colab Triton
parity is done. The Atlas differential audit was downgraded and never run.

**Stage (current):** Exact-route policy and exact-route construction are **exhausted**.
Sixteen distinct policies measured on the same 25 dense-verified natural held-out targets —
M in {1,2,4,8,16,32},
neighborhood W in {0,1,2}, K in {10,18,32,64}, sink S in {0,1,2} — and none exceeds **16/25**
(`dense-candidates` M=4, S=0). Dense scores 25/25 on the same prompts. The 32K deep-evidence
cell (d0.1) is **0.00 in every one of the sixteen**, and the 32K/d0.5 cell never exceeds 0.20.
Efficiency is settled and is not the problem: every configuration beats dense on prefill
(1.24-1.29x at 16K, 1.48-1.54x at 32K), the densified rows cost 0-3% at the useful budget, and
sink forcing costs nothing measurable. The remaining gap is attributed to the frozen backbone
being fed a representation distribution it never saw in training — the non-monotonic M curve
(dense 25/25, M=4 16/25, M=32 9/25) is a mixture-mismatch signature, not an
information-content one. Deterministic residual summaries tested and rejected the last
within-attention frozen-backbone construction lever: complete coverage moves 16/25 to only
17/25 and worsens final-hidden fidelity at both 16K and 32K. Exact-text evidence compilation,
which restores normal dense reading over a ~5% packet, is now the active non-adaptation branch.
The unbuilt LoRA/QLoRA stage remains a fallback only if a cheap live compiler selector cannot
preserve the 23/24 accuracy signal. Selector retraining is
ruled out quantitatively, not by assertion: conditional exactness given the evidence row is
densified is 0.62, so a perfect top-1 gate projects to ~15-16/25 — what M=4 already delivers.

**Backbone:** Qwen2.5-Coder-1.5B (H=12, G=2 kv-groups). 3B = laptop ceiling; 7B = ARC only.

**Built + validated:** chunked teacher extraction (P-prototype Q/K, offloading, GPU budget),
chunked-vs-eager validator, `selected_blocks` frozen + validator, needle harness, flat gate,
compact-ID candidate-only recursive traversal, PyTorch sparse reference, direct-index Triton
sparse prefill with isolated causal/prescale/query-tile ablations, deterministic hierarchical
residual-tree interface/frontier validator, live mean-K/V prototype pooling with multiplicity
bias, prototype-aware cost accounting, summary-disabled regression controls, prefill-only CUDA
profiler, K-sweep live-tree benchmark harness, exact-text evidence compiler with block/paragraph
boundary modes and source provenance, flat/tree/oracle compiler evaluation harness, repo-index
parser, tokenizer-only lexical block hierarchy, fully charged dense-vs-compiled benchmark,
test suite.

**Landed 2026-07-26/27 (not yet committed):** multi-target route-control suite
`benchmarks/run_route_control_suite.py` (one child process per combination instead of per
target, `--include-dense` so speedups are measured on the routes actually used, `--resume`,
per-case CSV); route knobs `--candidate-blocks` / `--candidate-neighborhood` /
`--sink-blocks` on the live benchmark and `--candidate-block-values` /
`--candidate-neighborhood-values` / `--k-selected-values` / `--sink-block-values` on the suite;
`candidate_span` + `neighborhood` in `scripts/route_overrides.py`; **attention-sink forcing**
`sink_blocks` in `scripts/export_selected_blocks.py`; deterministic residual-summary layout,
frontier, validator, mean-K/V pooling, mixed-softmax PyTorch attention, CLI controls, live
timing, and charged-entry accounting; the residual development diagnostic and
`colab/run_residual_summary_gate.ipynb`; evidence compilation in
`interfaces/evidence_compiler.py`, `selector/evidence.py`,
`benchmarks/evaluate_evidence_compiler.py`, and
`colab/run_evidence_compiler_gate.ipynb`; plus the prior route-control/accuracy/sink notebooks.
The current Colab source archive includes residual summaries, the evidence compiler, the live
pre-Qwen selector/charged benchmark, and the unscreened 16K–128K YaRN paper suite:
214 entries, 7,365,183 bytes, SHA-256
`1BA99CBA3681467B35F056E72C337E85DE8E50B0671C6C355FC8598F2F208595`.
Current workspace verification is **203 passed / 13 skipped**; the focused real-tokenizer
pre-Qwen/compiler/paper integration run passes 21/21.

**Open checkpoints / kill switches:**
- KS1 (Stage 2): **not cleared**, now measured on the full held-out set rather than a smoke set.
  The best frozen-backbone sparse result is deterministic residual P=1 at 17/25 (0.68) against
  dense 25/25 on the same prompts; KS1 asks for >95% of dense retrieval quality. Split by
  length: 16K 12/14 (0.86), 32K 5/11 (0.45). `recall@8` around 0.745 was a block-level proxy
  that never predicted generation — treat it as a selection metric only.
- Stage 2b: does a selector trained short (16-32K) generalize to long (64-128K) via needle harness? Not yet run. Higher risk than previously assumed: the gate is trained on 200 targets (160 natural + 40 replay) at 16K/32K only.
- Quantization checkpoint: does quantizing the backbone hurt needle recall at target length? Not yet run.
- KS2 (post-benchmark): tree beats/matches HiP at equal budget AND beats flat routing in real prefill cost. Blocked on Stage 3.
- Residual-summary gate: **closed — failed.** P=1 scored 17/25 overall and 5/11 at 32K with
  median charged attention 0.0754. Final-hidden relative-RMSE reductions were negative at both
  lengths. Neither the plug-and-play gate nor learned-compressor eligibility cleared; do not
  train the merger or build its Triton loop.
- Evidence-compiler accuracy gate: **passed.** Locked recursive M=4, radius-1 paragraph packets
  score 24/25 overall and 11/11 at 32K on the manifest-complete Colab set. Median compiled
  context is 1,358 tokens (5.73% of original). This is an accuracy/readability result only.
- Evidence-compiler engineering gate: **passed on the opened 25-case follow-up.** Beam 16,
  with final M still four, scores 25/25 and retains 7.521x fully charged sum-weighted speedup;
  expanded evidence recall is 25/25. This confirms candidate slack as the repair. It is not
  an untouched validation because beam was changed after seeing the beam-4 result. Freeze
  beam 16 and require approximately 100 new untouched prompts before a paper/toolkit claim.
- **Adaptation stage: fallback, still unbuilt and unscoped.** No `peft`/LoRA exists in the repo;
  the backbone has been frozen for every result to date. Do not start adaptation while the
  exact-text compiler can still be made deployable without changing the frozen-backbone claim.

**Current challenges (open, in priority order):**
1. **Expand independent semantic validation without touching the completed suite.** The
   frozen beam-16 run is complete and positive on its controlled distribution: 237/288 versus
   dense 192/288 at 9.58x, but its 12-case cluster interval still crosses zero. Do not sweep
   beam, D, M, radius, packet formatting, scoring, prompt generation, or analysis on this now
   opened bank. Add new independent semantic cases and external long-context tasks with the
   exact locked method; report both paired and case-clustered uncertainty before a general
   paper/toolkit accuracy claim.
2. **The sparse 32K wall is now an attributed representation failure.** Exact-route and
   residual-summary sparse prefill topped out at 4/11 and 5/11, while the same source locations
   re-encoded densely score 11/11 locally. Preserve that result on the complete manifest and
   stop treating more within-attention route construction as the active lever.
3. **Sink forcing and dense-candidates conflict.** Each helps alone (learned 3->8;
   dense-candidates 3->16) but together give 11/25. Densified rows already reach block 0, so
   the forced sink is pure budget overhead on the ~500 sparse body rows. Untried control:
   `dense-candidates` S=1 at **K=11**, restoring the slot the sink consumes. This is a cheap
   cleanup control, not a plausible rescue for the failed plug-and-play gate.
4. **Older `learned` numbers understate the method.** The missing attention sink was a
   construction defect, fixed 2026-07-27. Every pre-fix `learned` figure in this log (3/25,
   0/3 smoke, 32K 0.00) was measured with the sink competing for learned slots. Re-quote at
   S=1 before any of them reaches a paper.
5. **Checkpoint provenance.** `selector/train.py:131` saves only
   `{"num_layers", "head_dim", "proj_dim"}` — no flags, no target list, no argv. Recipes
   survive only in filenames, so "which settings produced this gate" is currently
   unfalsifiable. Fix before learned-summary or adaptation training starts.
6. **Timing hygiene.** At `--repeats 2` the first case of each combination carries CUDA warmup
   and can read below 1.0x kernel speedup (worst observed 0.686x). Use `--repeats >= 3` for any
   quoted timing; summary timing must include residual layout/frontier/transfer and live K/V
   construction.
7. **No established memory win.** Sparse peak exceeds dense at both lengths (16K 4.79 vs
   4.66 GB; 32K 6.52 vs 6.22 GB) on the prior exact-only path. Residual summaries add temporary
   FP32 prefix reductions and node tables; profile them before making any memory claim.
8. **Uncommitted work.** All tooling above, including residual summaries, is in the working
   tree only; nothing from 2026-07-26/27 has been committed.

---

## Entries

<!-- Newest first. Paste eval_gate.py / eval_tree_traversal.py output into Number, distill to one line in Conclusion. -->

## 2026-07-27 — Route-control tooling, sink forcing, and Colab archive rebuild (implementation)

**Question:** implementation entry, no measured claim. What was built to run the 2026-07-26/27
Colab campaign (runs 1-7), and what is the verified state of the code and the Colab archive?

**Config:** local Windows dev box for implementation and tests; Colab L4 for execution. All
route work goes through the frozen `selected_blocks` interface and `interfaces/validator.py`.

**Number (verification, not measurement):**
- `benchmarks/run_route_control_suite.py` rewritten: one child process per COMBINATION over all
  targets rather than one per (target, mode, backend), cutting ~240 model loads to 4-9 and
  making the full-set run feasible in one session. Adds `--include-dense`, `--resume`,
  `--repeats`, per-case CSV, and sweeps `--candidate-block-values`,
  `--candidate-neighborhood-values`, `--k-selected-values`, `--sink-block-values`.
- `scripts/route_overrides.py`: `candidate_span` helper plus `neighborhood` on
  `dense_candidate_routes`. Overlapping windows collapse, so densified-row cost is MEASURED and
  reported (`densified_rows`), never derived as M*(2W+1).
- `scripts/export_selected_blocks.py`: `sink_blocks` on `selected_ids_to_blocks`. Sink IDs are
  excluded from the nonlocal candidate set and skipped where the local window already covers
  them, so no slot is double-spent and no duplicate reaches the interface; rejects
  `k_selected < local_window + 1 + sink_blocks` because sink IDs sort lowest and truncation
  would otherwise drop the RECENT blocks. `sink_blocks=0` is byte-identical to the previous
  behaviour, asserted by test, which is what protects every earlier benchmark.
- `benchmarks/compare_dense_sparse_live_tree.py`: `--candidate-neighborhood`, `--sink-blocks`,
  `sink_blocks` threaded through `live_route`, and `densified_rows` / `span_contains_needle` /
  `sink_blocks` recorded per case and in the report's `selector` metadata.
- Tests 131 -> **139 passed, 11 skipped**. New coverage: combo-stem suffixes, span cost from the
  child rather than recomputed, sink default no-op, sink forced into every causal row, sink not
  duplicated inside the local window, budget rejection.
- Notebooks under `colab/`: `spruce_colab_route_control` (runs 1-3),
  `spruce_colab_accuracy_ladder` (runs 4-6), `spruce_colab_sink_test` (run 7). All code cells
  AST-checked with magics and backslash continuations stubbed — 0 syntax errors.
- Colab archive rebuilt from source: 185 -> **160 entries** (25 `__pycache__`/`.pyc` dropped),
  SHA-256 `E5148A2D8F5F98AFDD087043D4ED29F941D84E63B3306403236428DD7525A33A`, previous archive
  retained as `spruce_colab_train_source.prev.zip`. Capability check passes for every file the
  run-7 notebook requires, so that notebook needs no patch cells.

**Conclusion:** The campaign infrastructure is in place and verified, and the one behavioural
change in it — sink forcing — is proven a no-op at its default, so no earlier result is
retroactively altered by the tooling itself. Two process lessons worth keeping. Notebook code
cells must be AST-parsed before being sent (a `\n` collapsed inside a string literal cost a
Colab session), and archive contents must be verified by CONTENT rather than pinned by digest,
since the archive is rebuilt by hand. Nothing here is committed yet.

## 2026-07-27 — Attention sink WAS being dropped: learned 3/25 -> 8/25, but the 32K d0.1 wall stands

**Question:** `scripts/export_selected_blocks.py` forced only the local window
`q-local_window..q`; block 0 competed for learned slots like any other block, and run 5 showed
the gate spending rank 1 on it in 6/25 cases. Does forcing the sink outside the K budget —
standard practice in block-sparse attention, never varied by the run 4-6 ladder — move the 32K
depth wall that ten route policies could not?

**Config:** Colab L4, same 25 dense-verified natural held-out targets, same gate
`natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`, K=10, beam 8, Triton,
`--include-dense`, repeats 2. New `--sink-blocks N` / `--sink-block-values`. S in {0,1,2} x
{`learned`, `dense-candidates` M=4}. Artifacts:
`SPRUCE_COLAB/outputs/sink_test/run7_sink/{summary,cases}.csv`.

**Number:** S=0 reproduces both baselines exactly (`learned` 3/25, `dense-candidates` 16/25),
so the run is trustworthy.

| mode | S=0 | S=1 | S=2 |
|---|---|---|---|
| learned | 3/25 (0.12) | **8/25 (0.32)** | 8/25 (0.32) |
| dense-candidates M=4 | **16/25 (0.64)** | 11/25 (0.44) | 11/25 (0.44) |

By length — `learned`: 16K 3->5, 32K **0->3**. `dense-candidates`: 16K **12->7**, 32K 4->4.
32K by depth, `learned`: d0.9 0.00 -> **0.667**, d0.5 0.00 -> 0.20, d0.1 0.00 -> **0.00**.
32K by depth, `dense-candidates`: unchanged at d0.1 0.00 / d0.5 0.20 / d0.9 1.00 for every S.
Median kernel speedup 1.27-1.32x across all six configurations; S costs nothing measurable.
S=2 is identical to S=1 in every cell — one sink block is the whole effect.

Mechanism, from per-case `needle_hit` (same case, `learned`, S=0 -> S=1): aquifer 32K d0.9
0.589 -> 0.571; observatory 16K d0.9 0.804 -> 0.786; aquifer 16K d0.5 0.411 -> 0.393. The
forced sink consumes one of the ~8 free slots, so evidence coverage drops slightly — and
`learned` still gains 5 cases, meaning the sink is worth more than the slot it costs.
For `dense-candidates` the picture inverts: `needle_hit` is 1.000 at every S (the reader row
and the four candidate rows are dense, so they already attend block 0), so the sink adds
nothing where it matters while still costing a slot on every one of the ~500 sparse body rows.
All five regressions are 16K cases that were exact at S=0, and they fail as near misses
(gallery 16K d0.9 "17 April 1986" -> "17 April 1987"; aquifer 16K d0.5 -> "Lake Vesta").

**Conclusion:** The missing sink was a real construction defect and is now fixed: the pure
sparse path nearly triples (3/25 -> 8/25) and takes its first 32K wins ever (0 -> 3). Any
future `learned` baseline must be quoted at S=1; the S=0 numbers understate the method through
a bug, not a method limitation.
It is NOT the answer to the accuracy gap. Sink forcing and dense-candidates are **substitutes,
not complements** — each helps alone, together they are worse than dense-candidates alone
(16 -> 11), because densified rows already reach the sink and the forced slot is pure overhead
on the sparse remainder. Best configuration is still `dense-candidates` M=4 S=0 at 16/25.
Decisive for strategy: **32K d0.1 is 0.00 in all six configurations**, as it was in all ten
route configurations of runs 4-6. Sixteen distinct route/construction policies, one immovable
wall. Deep evidence at long context is not reachable by route construction.
One control worth running before closing this line: `dense-candidates` S=1 at **K=11**, which
restores the slot the sink consumes. If it returns to ~16/25 the regression is purely budget
and the sink fix is free; if not, the interaction is real. Either way it will not move d0.1 —
adaptation remains the only untried lever for deep evidence.

## 2026-07-27 — Route policy is exhausted at 16/25: M peaks at 4, contiguity loses, K is inert

**Question:** three levers left on the routing side after run 3. (a) Is M=4 the peak, or does
the curve keep rising below it? (b) Must the repaired rows be CONTIGUOUS around the evidence,
as the depth-ordered failures suggest, or are scattered high-scoring rows equally good per row
spent? (c) Does the route budget K still bind once rows are repaired?

**Config:** Colab L4, Qwen2.5-Coder-1.5B-Instruct, block_size 64, beam 8, Triton
`single_head`, same 25 dense-verified natural held-out targets (16K/32K, depths d0.1/d0.5/d0.9)
and same gate `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt` as runs 1-3.
`benchmarks/run_route_control_suite.py` with new `--candidate-neighborhood-values` and
`--k-selected-values`. Run 4: M in {1,2,4}, W=0, K=10, `--include-dense`, repeats 2. Run 5:
M in {1,4} x W in {0,1,2}, K=10, dense skipped, repeats 1. Run 6: M=4, W=0,
K in {10,18,32,64}, `--include-dense`, repeats 2. Artifacts:
`SPRUCE_COLAB/outputs/accuracy_ladder/{run4_small_m,run5_span_vs_scatter,run6_k_sweep}/`.

**Number:**

Run 4 — M curve, both sides now measured (M=8/16/32 from run 3):

| M | 1 | 2 | **4** | 8 | 16 | 32 | dense |
|---|---|---|---|---|---|---|---|
| exact | 11/25 | 14/25 | **16/25** | 12/25 | 9/25 | 9/25 | 25/25 |
| candidate recall | 0.52 | 0.84 | 0.96 | 1.00 | 1.00 | 1.00 | — |

`learned` floor 3/25, median kernel speedup 1.337x; dense-candidates 1.278-1.285x across M.

Run 5 — matched-cost contiguous versus scattered: **M=1/W=2 (5 contiguous rows) 12/25 (0.48)
versus M=4/W=0 (4 scattered rows) 16/25 (0.64)**. Scattered wins at lower cost. M=4 with W=1
(9 rows) and W=2 (14 rows) both give 16/25 — identical to W=0 at 4 rows, so neighbors are pure
cost. Widening W at M=1 lifts span recall 0.52 -> 0.72 but exact only 0.44 -> 0.48.
Six of 25 cases have `densified_rows=2` at M=1/W=1, which means the gate's top-1 block was
block 0 (window clamped): all six score 0 exact, span recall 0.00.

Run 6 — K = 10/18/32/64 at M=4: **16/25 at every K**, identical rates, candidate recall 0.96
throughout. Median kernel speedup 1.272/1.291/1.291/1.282 — raising K 6.4x costs essentially
nothing here and buys nothing.

Conditional accuracy given the evidence row WAS densified, at M=1: 8/13 = 0.62. Given it was
NOT: 3/12 = 0.25.

32K exact rate by depth, identical in all ten route configurations: d0.1 **0.00**,
d0.5 **0.20**, d0.9 **1.00**. By length, every configuration at or below 16K 12/14, 32K 4/11.

**Conclusion:** Route policy is exhausted. Ten distinct policies — M from 1 to 32, W from 0 to
2, K from 10 to 64 — all land at or below 16/25, and the 32K depth wall (d0.1 = 0.00) does not
move by a single case under any of them. This is the pre-declared plateau condition: stop
tuning routes, the remaining gap belongs to the frozen backbone, and the next lever is the
unbuilt adaptation stage.
Three specific results worth carrying into the paper. (1) The M peak is a crossover, not the
optimum of one effect: below M=4 the binding constraint is gate recall (0.52 at M=1), above it
distractor amplification. (2) **Corrects the run-1 claim that the selector is saturated and
retraining would optimize a ceiling metric.** Recall is 1.000 only at M>=8; at the operating
point M=4 it is 0.96 and at M=1 it is 0.52. The conditional rate settles the question
quantitatively instead: even when the evidence row IS densified, only 0.62 of cases are exact,
so a perfect top-1 gate at M=1 projects to roughly 15-16/25 — what M=4 already delivers. The
gate is still not the bottleneck, but the argument is now a measured conditional rate rather
than an oracle anecdote. (3) The contiguity hypothesis is dead: repairing a span around the
evidence is strictly worse per row than repairing scattered high-scoring blocks, so the depth
ordering is not about locality of repair.
Measurement caveat for anyone reading the per-case CSV: the first case of each combination
shows a sub-1.0 kernel speedup (0.686x worst, aquifer 16K d0.1 at M=1) with selector time
0.166s against a 0.104s norm. That is CUDA warmup at repeats=2, not a regression — medians are
unaffected. Use repeats>=3 for any quoted timing.

## 2026-07-26 — M sweep with the dense rows paired: cost is a non-issue, and MORE densification is WORSE

**Question:** two things the first two Colab runs could not answer. (a) With the densified
rows inside the measured routes, does sparse prefill still beat dense — specifically at 16K,
where an earlier K18 run measured 0.887x (slower than dense)? (b) How does exact-retrieval
rate trade against cost as the candidate budget M grows?

**Config:** Colab L4 (not A100), Qwen2.5-Coder-1.5B-Instruct, block_size 64, K=10, beam 8,
`--repeats 3`, `--include-dense` (paired dense run, so every speedup is measured on the same
routes rather than inherited). Triton backend, `single_head`. 25 natural held-out targets
(dense-verified via `screen_natural_prompts.py`), 16K and 32K, evidence depth d0.1/d0.5/d0.9.
`benchmarks/run_route_control_suite.py --modes learned dense-candidates
--candidate-block-values 4 8 16 32`. Outputs:
`SPRUCE_COLAB/outputs/route_control_heldout/run3_triton_cost_msweep/{summary,cases}.csv`.

**Number:**

| mode | M | exact | rate | median kernel speedup | median live prefill |
|---|---|---|---|---|---|
| learned | — | 3/25 | 0.12 | 1.290 | 1.193 |
| dense-candidates | 4 | **16/25** | **0.64** | 1.291 | 1.195 |
| dense-candidates | 8 | 12/25 | 0.48 | 1.283 | 1.188 |
| dense-candidates | 16 | 9/25 | 0.36 | 1.271 | 1.178 |
| dense-candidates | 32 | 9/25 | 0.36 | 1.252 | 1.162 |
| dense (paired) | — | 25/25 | 1.00 | — | — |

Cost, per length, M=4: kernel prefill speedup 1.238–1.294x at 16K, 1.481–1.536x at 32K.
Every one of the 100 dense-candidates cases is >1.0x; the worst case in the whole sweep is
1.186x (council 16K, M=32). Same-case sparse prefill cost versus `learned`: +0.0–3.0% at M=4,
+7.3–8.3% at M=32. M=4 densifies 4 of 256 query blocks at 16K (1.6%) and 4 of 512 at 32K (0.8%).
Peak memory: sparse 4.794 GB vs dense 4.658 GB at 16K (+2.9%), 6.516 vs 6.219 at 32K (+4.8%) —
sparse costs MORE memory here, at every M including `learned`.
Candidate recall 0.96 at M=4, 1.000 at M>=8.
M=4 by length: 16K 12/14, 32K 4/11. By depth at 32K: d0.9 3/3, d0.5 1/5, d0.1 0/3.
At 16K: d0.5 5/5, d0.9 5/5, d0.1 2/4.
Failure text sharpens with M: gallery 16K d0.5 is exact at M=4 and M=8, but answers
"4 September 1987" (a well-formed date from a distractor record) at M=16 and M=32.

**Conclusion:** The efficiency worry is dead — dense rows cost 0–3% of prefill at the useful
budget and the 16K advantage survives (1.24–1.29x), so the claim does NOT narrow to >=32K.
The accuracy result is the opposite of the prediction after runs 1–2: M is an accuracy knob,
and it runs backwards — M=4 gives 16/25 where M=8 gives 12/25 and M=32 gives 9/25, while full
dense gives 25/25. The curve is non-monotonic between the two endpoints, so this is not an
information-content effect. Two candidate mechanisms, not yet separated: (i) *distractor
amplification* — beyond rank ~4 the gate's top-M are the highest-scoring near-miss distractors,
and densifying them restores their representations to full fidelity so they compete with the
evidence (supported by wrong answers becoming sharper and more specific as M grows); (ii)
*mixture inconsistency* — heterogeneous dense/sparse rows in one layer. Discriminating run:
densify ranks 5–8 only, versus ranks 1–4 only, at matched count. Default M drops from 8 to 4,
and M=4 is now the sweep edge — M=1,2 must be measured before 4 is called optimal.
Two further notes for the paper: `candidate_contains_needle` is useless as a predictor
(the one M=4 case that misses the evidence block is exact; many M=8 cases that contain it are
wrong), confirming the failure is downstream representation, not selection; and no memory win
may be claimed at these lengths — the dense baseline already runs a memory-efficient kernel.
`live_total_speedup` dips below 1.0 only on council (0.956 at M=4 → 0.925 at M=32), a
generation-length artifact (sparse emits a long verbose answer, dense a short one), not a
prefill regression.

## 2026-07-26 - Triton/PyTorch parity on the held-out route-control suite: rates identical, two cases discordant

**Question:** Does the Triton sparse-prefill kernel reproduce the PyTorch reference backend's conclusions on the same routes, so the dense-candidates result is a method finding rather than a kernel artifact?

**Config:** Same Colab L4 session, gate, targets (25 re-screened natural held-out), K=10, beam=8, one repeat, 16 new tokens, `--skip-dense`. Only `--backend` differs between the two runs (`benchmarks/run_route_control_suite.py --backends pytorch`, run 2, versus `triton`, run 1). Artifacts: `SPRUCE_COLAB/outputs/route_control_heldout/run2_pytorch_parity/{summary,cases}.csv`.

**Number:** Aggregate rates are identical: learned 3/25 exact on both backends, dense-candidates M=8 12/25 exact on both, candidate recall 1.000 on both. Per-case exactness agrees on 23 of 25 dense-candidates targets, with two discordant cases pointing in **opposite** directions: gallery 16K d0.1 (Triton "17 April 1986" exact 1, PyTorch "1987" exact 0) and observatory 32K d0.5 (PyTorch "Kepler Field 731" exact 1, Triton "Kepler Field 739" exact 0). The learned mode agrees 25/25 on exactness while differing in wrong-answer text on two cases (atlas 16K d0.5: "index/catalog/9882" versus "atlas/chk"; observatory 32K d0.1: "Northern Mosaic C" versus "Newer Interpretation"). Reference-backend prefill was 9.3-10.0s at 16K and 18.6-20.6s at 32K, against 1.28-1.33s and 2.52-2.72s for the Triton kernel on the same routes (a reference-versus-kernel ratio, not a dense speedup claim).

**Conclusion:** Kernel attribution is closed for this finding: both the failure of learned routes (3/25) and the recovery under dense-candidates (12/25), plus the saturated 1.000 candidate recall, reproduce exactly on the plain-PyTorch reference, so none of it is a Triton artifact. The two discordant cases are borderline decodes flipped by ordinary floating-point differences between the two attention implementations - they cancel (one each way), show no systematic direction, and both involve answers already known to sit next to a close distractor ("739"/"731", "1987"/"1986"). The practical rule for the paper: aggregate exact rates are backend-stable, individual case answers are not, so no single-case claim should be made without stating the backend. This also means the depth-ordered failure pattern from run 1 is a property of the routing policy, not of the kernel.

## 2026-07-26 - Dense-candidates on the full natural held-out set: 3/25 -> 12/25, and the residual failure is evidence DEPTH

**Question:** Is deployable retrieve-then-re-encode (`--route-mode dense-candidates`, no oracle) a rate or an anecdote, and does the selector's ranking need retraining?

**Config:** Colab L4 (23.6GB); Qwen2.5-Coder-1.5B-Instruct FP16; gate `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; Triton `single_head` sparse prefill; K=10, beam=8, local window 1, 16 new tokens, one repeat, `--skip-dense`. Held-out targets were not on Drive, so they were re-derived on Colab with the same protocol as the stored set: dense screening of `scripts/prompt_banks/natural_heldout.json` (6 cases x {16384, 32768} x depths {0.1, 0.5, 0.9}, seed 20260725, `screen_natural_prompts.py`) accepted 25 of 36 candidates, then `extract_teacher_targets.py --verified-manifest --features-only` (selector Q/K prototypes only; the modes used here never read teacher mass). This is a 25-target set re-screened under the same rules, not the previously stored 24 - dense verification is by construction (only dense-exact prompts are accepted). Runner: rewritten `benchmarks/run_route_control_suite.py` (one child process per backend x mode x M over all targets, per-case CSV). Archive SHA-256 45F7867DB0444B5570BB627A5526E546D04CC509C47DEFCF05872FB3F98C2559.

**Number:** learned routes: 3/25 exact (council 16K d0.9, gallery 16K d0.9, turbine 16K d0.5). dense-candidates M=8: 12/25 exact. **Candidate recall was 1.000 on all 25 targets** - the gate's own top-8 reader-row blocks contained the evidence block every single time, and per-head/any-group needle hit rates were 1.0 by construction. The residual failures are ordered almost perfectly by evidence depth: d0.9 recovers 8/8, d0.5 recovers 3/10, d0.1 recovers 1/7. By length: 16K 9/14, 32K 3/11; at 32K only the d0.9 cases recover. Wrong answers remain close distractors from the prompt bank ("Kepler Field 739" for 731, "Lake Vesta" for Lake Orison, "cache/atlas/tmp" for state/atlas.chk). Measured sparse prefill cost of densification was small: 1.31s vs 1.28s at 16K and 2.67s vs 2.59s at 32K (about +2-3%, consistent with the (M+1)/qb arithmetic); four first-touch cases show Triton autotune outliers (up to 5.19s) that repeats=3 in the cost run will remove. No dense baseline in this run (`--skip-dense`). Artifacts: `SPRUCE_COLAB/outputs/route_control_heldout/run1_triton_accuracy/{summary,cases}.csv`.

**Conclusion:** Retrieve-then-re-encode is a real effect at scale, not an anecdote - it quadruples exact retrieval (3/25 -> 12/25, 12% -> 48%) with no oracle knowledge, no retraining, and about 2-3% extra prefill - but it does not close the gap to dense (25/25 by construction). Two things are now settled. (1) **The selector's ranking is not the problem and does not need retraining for this failure**: candidate recall is 1.000 across every held-out target, so the evidence is already ranked first-tier at the reader row; any further selector training would optimize a metric that is already saturated. (2) **The binding constraint is the sparse span between evidence and reader**, measured as evidence depth: when the evidence sits near the question (d0.9) densifying the candidate rows recovers it every time, and recovery decays monotonically as the evidence moves earlier (d0.5 30%, d0.1 14%), worsening with length. This is consistent with the earlier Observatory attribution - the evidence block's K/V are repaired by densifying its own row, but every block *between* the evidence and the reader still reads that region through K=10 routes and propagates a corrupted representation forward. Next controls should target that span (densify candidate neighborhoods, or the causal rows between candidate and reader), not the selector loss. Triton/PyTorch parity and the dense-included cost + M sweep are still running; note that with candidate recall already 1.000 at M=8, larger M cannot add the evidence and the M knob is now primarily a cost knob, not an accuracy knob.

## 2026-07-26 — Deployable retrieve-then-re-encode recovers 2/3 natural failures without oracle knowledge
**Amended 2026-07-27 — three claims below are superseded.** (a) 2/3 was three prompts; the rate
on 25 dense-verified held-out targets is 16/25 at M=4 (12/25 at the M=8 used here). (b) "the
gate already ranks the evidence first" holds only at M>=8, where candidate recall is 1.000;
at M=4 it is 0.96 and at M=1 it is 0.52. (c) Atlas is not a sharply-posed unique outlier —
atlas 16K recovers under dense-candidates, and atlas 32K d0.5 fails exactly like every other
deep-evidence case. The differential audit proposed here was downgraded and never run.
**Question:** Does densifying the gate's OWN top-M reader-row blocks (no needle metadata) recover the natural retrieval failures, and does the Atlas audit explain its resistance?
**Config:** New `--route-mode dense-candidates --candidate-blocks 8`: gate leaf scores at the reader row, max-aggregated over layers/KV groups, top-8 blocks' query rows densified plus dense reader row; body keeps learned K=10 routes. Extra prefill cost ≈ (M+1)/qb of dense (≈1.8% at 512 blocks). Same harness (PyTorch reference, RTX 4070, FP16, `--skip-dense`, 16 tokens). Atlas token-level audit via `scripts/audit_dense_attention.py` (dense forward, exact-correct sanity gate passed).
**Number:** Gallery 16K: "17 April 1986", exact 1; the gate self-selected the needle (block 127) as its top candidate. Observatory 32K d0.9: "Kepler Field 731", exact 1; needle (458) again top candidate — recovered from the stable "739" near-miss with zero oracle input. Atlas 32K d0.5: "cache/atlas/tmp", exact 0, even though the needle (256) was the top candidate and blocks 256/257 were both densified. Atlas audit: dense question rows attend the evidence block with far HIGHER mass than Gallery's (max 0.9994 at L19 h3, 0.96-0.99 at L14/16/20/22/23; hundreds of row-head pairs rank it top-8), and dense decodes "state/atlas.chk" exactly. Run artifacts in `benchmarks/outputs/oracle_route_control/`; suite runner `benchmarks/run_route_control_suite.py` packages the full mode-x-backend matrix for the Colab Triton parity rerun.
**Conclusion:** Retrieve-then-re-encode works as a deployable mechanism: the existing gate already ranks the evidence first at the reader row on all three prompts, and densifying its top-8 candidate rows converts 0/3 into 2/3 exact at ~2% extra prefill compute — without retraining anything. Atlas is now a sharply-posed open case: dense attention on it is nearly saturated (0.999) yet no partial densification (evidence rows, +-1, candidates incl. both boundary blocks) recovers it, so the necessary computation involves context outside every densified set — next step is a differential audit (same rows, sparse vs dense forward) to find which layer first diverges, not more densification guesses. Before paper claims: validate dense-candidates on the full natural held-out set and legacy suite with Triton on Colab (`run_route_control_suite.py --backends triton pytorch`), and treat M as a measured knob (candidate-recall vs cost curve).
**Question:** If the evidence block's own query row is contextualized densely (in addition to a dense reader row and evidence forced into every route), does 32K natural retrieval recover — i.e., is the 32K failure specifically corruption of the evidence block's K/V under sparse prefill?
**Config:** New `--route-mode oracle-dense-evidence` (+ `--evidence-neighborhood N`): learned K=10 routes everywhere except (a) reader row dense, (b) evidence block's query row dense over its causal prefix (optionally +-N neighbor rows), (c) evidence block forced into every causal row. Same harness as prior entries (PyTorch reference, RTX 4070, FP16, `--skip-dense`, one repeat, 16 tokens). Oracle knowledge of the needle — attribution only, not deployable.
**Number:** Observatory d0.9 32K: "Kepler Field 731" — exact 1, recovered from a stable "739" near-miss that had survived K=64, oracle-needle, and dense-reader. Atlas d0.5 32K: "cache/atlas/tmp" — exact 0; with `--evidence-neighborhood 1` (evidence±1 rows dense) still "cache/atlas/tmp" — exact 0, rejecting the block-boundary-straddle explanation for this case.
**Conclusion:** The 32K mechanism is now demonstrated for Observatory: sparse prefill corrupts the evidence block's own deep K/V (its 64 tokens attend ~10 of 512 blocks), and repairing just that one row's context — cost O(L) per densified row — restores exact retrieval. Combined with Gallery (dense reader row alone suffices at 16K), two of three failures are fully attributed and repaired by targeted densification totaling <1% extra prefill compute. Atlas resists every partial control (evidence access, dense reader, K=64, dense evidence row, ±1 neighborhood) while dense solves it — its answer is consistently a recombined path fragment, suggesting the required context is distributed beyond the evidence neighborhood; needs its own audit (token-level, per-row) before further densification guesses. Deployable design implied for the paper: sparse prefill + dense reader/question rows + second-pass re-densification of the selector's top evidence candidates ("retrieve then re-encode"), to be validated without oracle knowledge and with Triton parity on Colab.

## 2026-07-26 — K=64 body-budget sweep does not rescue 32K natural retrieval
**Confirmed 2026-07-27 on the full held-out set.** Run 6 swept K = 10/18/32/64 at
`dense-candidates` M=4 over all 25 targets and returned **16/25 at every K**, identical rates,
with median kernel speedup barely moving (1.272/1.291/1.291/1.282). Route budget is not merely
insufficient — it is inert. This entry's conclusion stands as written.
**Question:** Does raising the selected-block budget from K=10 to K=64 (of 512 blocks; 6.4x) repair the two 32K natural failures, with and without a dense reader row?
**Config:** Same harness/backbone/gate as the oracle-route entry (PyTorch reference backend, RTX 4070, FP16, `--skip-dense`); K=64, beam=64 (effective 62), one repeat, 16 new tokens; Atlas d0.5 and Observatory d0.9 32K held-out targets; modes learned and dense-reader.
**Number:** Atlas K64 learned: "state/atlas/tmp" (hit 0.732); Atlas K64 dense-reader: "logs/atlas.chk" (hit 1.0). Observatory K64 learned: "Kepler Field 739" (hit 0.893); Observatory K64 dense-reader: "Kepler Field 739" (hit 1.0). All exact 0/4. Atlas answers drift toward the truth as budget grows ("atlas/audit/..." at K10 -> "state/atlas/tmp"/"logs/atlas.chk" at K64 — correct fragments recombined wrongly); Observatory is a stable near-miss "739" across every intervention (K10/K64, oracle injection, dense reader).
**Conclusion:** Neither budget (to 12.5% of blocks) nor reader coverage nor evidence forcing recovers 32K natural retrieval, while the identical pipeline retrieves Gallery 16K exactly with a dense reader row — so the pipeline is sound and the failure is representational: under sparse prefill the model reconstructs plausible answer fragments instead of reading the evidence, and close distractors win. Do not attempt to fix this with needle-loss weighting. Next decisions are architectural and belong in a fresh design pass: (a) dense/high-budget reader-question rows as standard (fixes the 16K class nearly free), (b) measure at what body budget or with what selective densification (e.g., dense rows for evidence-neighborhood blocks) 32K recovers, (c) rerun this suite on Colab with the Triton kernel to confirm reference/kernel parity of these findings, and (d) only then decide whether KS1's recipe (train selector under corrected labels + new route policy) is still the binding step.

## 2026-07-26 — Oracle-route controls: evidence access is not the binding constraint; a dense reader row is
**Amended 2026-07-27.** The central conclusion — evidence access is not sufficient — stands and
is now quantified: conditional exactness given the evidence row is actually densified is 0.62,
so even a perfect selector projects to ~15-16/25. But every `learned` number in this entry was
measured with the attention sink competing for learned slots (fixed 2026-07-27); the sparse
baseline here is understated by a construction defect. Re-measure at `--sink-blocks 1` before
quoting any `learned` figure from this entry.
**Question:** Is routing the evidence block into `selected_blocks` sufficient for correct sparse generation on the failing natural prompts, and if not, what is?
**Config:** Local RTX 4070 Laptop 8GB; PyTorch sparse reference backend (`sparse.attention`), not Triton — kernel-independent by construction; Qwen2.5-Coder-1.5B-Instruct FP16; `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; K=10, beam=8, local window 1, 16 new tokens, one repeat; `--skip-dense` (a full-length dense SDPA prefill OOMs on 8GB when Transformers materializes the 4D mask — dense answers come from the stored dense-screen verification instead). New `--route-mode` controls in `benchmarks/compare_dense_sparse_live_tree.py`: `learned` (production), `oracle-needle` (learned routes widened one slot, evidence block forced into every causal row of every layer/group), `teacher-top8` (routes packed from teacher mass, no gate), `dense-reader` (learned routes everywhere except the final reader row, which attends every causal block). Gallery 16K d0.5 target (`teacher_gallery_reopened_april_1986_L16384_d0.5...`), dense-verified answer "17 April 1986".
**Number:** learned: answer "1987", exact 0, needle layer-group hit 0.375/any-group 0.643. oracle-needle: injection verified at 1.000/1.000 hit rates, answer still "1987", exact 0. teacher-top8: teacher routes carry the needle in 0.000 of rows (matches the 0.00 unconditional eligibility measured earlier) and answer "4 September 1987" — an explicit distractor date. dense-reader: answer "17 April 1986", exact 1, with the document body still at learned K=10 sparsity. A dense reader row costs one query block × all key blocks ≈ 1/qb of dense prefill (~0.4% extra at 256 blocks).
**Number (32K replication):** Atlas d0.5 (dense answer "state/atlas.chk"): learned "atlas/audit/audit_1994/audit_19" (hit 0.268), oracle-needle "atlas/chk" (hit 1.0 — closer but wrong), dense-reader "logs/atlas.log" (a distractor path, hit 1.0). Observatory d0.9 (dense answer "Kepler Field 731"): learned "Kepler Field 739" (hit 0.554), oracle-needle still "Kepler Field 739" at hit 1.0, dense-reader "Northern Mosaic C" (hit 1.0). All exact 0/6 at 32K.
**Conclusion:** Access to the evidence block is NOT sufficient — forcing it into every route changes nothing — and the teacher's own top-8 mass routes actively select a distractor, closing the label-ceiling argument end-to-end. At 16K the binding constraint is reader-row COVERAGE (dense reader row alone recovers the exact answer with the body still K=10 sparse); at 32K even a fully dense reader row fails, so K=10 body sparsification is corrupting the document-side representations themselves (the evidence block's own tokens attend only ~10 of 512 blocks during prefill, so its deep K/V drift from dense; Observatory answers "739" while directly reading the "731" sentence). The fix is therefore layered: dense/high-budget reader-question rows (near-free, O(L) for one block row) plus a body budget or routing good enough to keep evidence-region representations faithful — needle supervision of the reader row alone cannot repair this. K=64 body-budget sweep running to locate the budget threshold.

## 2026-07-26 — Token-level audit: dense retrieval lives in question-row tokens, killed by query-side block pooling
**Question:** Where does dense Qwen actually attend the evidence on the Gallery case whose block-pooled teacher target ranks the evidence out of top-8 everywhere?
**Config:** New `scripts/audit_dense_attention.py`: registers a recording attention backend (post-RoPE Q/K, output delegated to causal SDPA so the forward stays numerically dense), audits the last 64 prompt tokens (question span; tokenized-subsequence match failed so the fallback last-block window was used) plus all 16 greedy decode steps; Gallery 16K d0.5 target; Qwen2.5-Coder-1.5B-Instruct FP16 on RTX 4070.
**Number:** Sanity gate passed: the audited forward decodes "17 April 1986" (exact 1.0). Prefill question rows show strong evidence attention at mid/late layers: max per-layer max-head evidence-block mass 0.317 (L24 h11, row 45), 0.277 (L19 h3, row 45), 0.179 (L26), 0.113 (L23), with layers 14-27 placing the evidence in some head's block-top-8 for 26-90 of 64x12 row-head pairs. Layers 0-8 are near zero (max 0.014). Decode steps attend the evidence far more weakly: per-layer max mass peaks at only 0.029 (L25), though rank<8 counts rise at L21+ (up to 38). The dense retrieval signal is therefore concentrated in a few specific question-row TOKENS at specific heads/layers during prefill — exactly what pooling attention over the 64-token reader block and averaging heads into KV groups erases in the teacher targets.
**Conclusion:** The label source is recoverable: unpooled question-token rows carry a 0.3-mass retrieval signal the current block-pooled, group-averaged target reduces to rank>8. Combined with the oracle-route result, the fix hierarchy is: (1) give reader/question rows dense or large-budget routes at inference (near-free), (2) if selector supervision of the reader row is still wanted, build its target from question-token rows (max over question tokens and heads, not mean over the block), (3) `--needle-eligibility always` remains useful only on top of (1)/(2).

## 2026-07-26 — Teacher-mass labels themselves erase evidence on natural held-out cases
**Question:** Before redesigning the selector objective, does the group-averaged dense-attention teacher target actually contain the evidence block for the natural cases the sparse path fails, and is head averaging the erasure mechanism?
**Config:** Local CPU diagnostic; no training. Read unconditional `teacher_needle` / `teacher_needle_union` from the existing `benchmarks/outputs/natural_gate_eval/original_vs_natural_K10.json` (25 dense-verified natural held-out targets, K=8 reader row). Then loaded raw per-head pooled mass `[28,12,256,256]` from `teacher_gallery_reopened_april_1986_L16384_d0.5_s20260725_len16375_blk64_d0.5.pt` and computed needle-in-top-8 at the reader row per head versus per KV group.
**Number:** Unconditional teacher eligibility across the 25 held-out targets: per-layer-group mean 0.638 (min 0.000), any-group-per-layer mean 0.731 (min 0.000). The Gallery case — the exact live 0/3 failure that answered "1981" instead of "17 April 1986" — has teacher eligibility 0.00 at 16K/d0.5 and only 0.29–0.46 at its other lengths/depths. On that 0.00 file, per-head analysis shows the erasure is not primarily head averaging: even before group averaging, only 4 of 336 layer-heads put the needle in reader-row top-8 (any-head-per-layer 0.143), the strongest head ranks the needle 13th at 1.1% of row mass, and group averaging drops all four survivors to zero. Dense generation on this same prompt is verified correct.
**Conclusion:** The prefill reader-row block-pooled attention mass is a broken label for answer-critical evidence on natural data: dense Qwen answers correctly while the training target assigns the evidence block near-zero rank, so KL/top-k actively train the selector to exclude the needle, and every needle loss (group and union) is gated on teacher eligibility (`loss.py` `teacher_keeps_needle`) and therefore fires zero exactly on the hardest cases. The label ceiling (0.64/0.73 mean, 0.00 worst) times the student's 0.62–0.85 ratio reproduces the observed 0.48–0.75 live evidence-hit rates; no lambda sweep or tree-level supervision can pass through this ceiling. Before any retraining: (1) run the oracle-route control — force the evidence block plus window into `selected_blocks` for all layers/groups on the three failing prompts to establish whether routed evidence is even sufficient for correct generation; (2) audit token-level (unpooled) dense attention for the Gallery case, including decode-step queries, to find where retrieval attention actually lives; (3) drop the teacher-eligibility gate and treat the known evidence block as a hard positive for the reader row regardless of teacher mass.

## 2026-07-26 — Tree-aware continuation objective diverges from retrieval proxy
**Question:** Does the new all-tree-level boundary/union-needle continuation improve the selector's training-set leaf recall and evidence retention enough to justify completing the 200-epoch overnight run?
**Config:** User-run Colab continuation from `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; Qwen2.5-Coder-1.5B teacher features; 160 natural plus 40 deterministic replay targets per epoch; radix 2; beam/top-k/needle-top-k 8; lambda top-k/boundary/needle 0.5/0.5/1.0; top-k and needle margins 0.25; union needle objective; all tree levels; LR 2e-4; CPU target offload; nominal 200 epochs. The reported inline evaluation is on the training targets and remains a diagnostic only, not held-out evidence.
**Number:** Epoch 1 training loss was KL/top-k/boundary/needle 0.2450/0.3721/1.4102/0.1181 with 16K/32K leaf recall@8 0.761/0.756 and needle@8 0.82/0.81. By epoch 10 the losses were 0.1660/0.3663/1.1905/0.0317 while recall@8 fell to 0.751/0.743 and needle@8 to 0.80/0.76. By epoch 50 the losses were 0.1533/0.3639/1.1284/0.0225 while recall@8 remained 0.751/0.742 and needle@8 fell to 0.73/0.69. By epoch 90 the losses were 0.1515/0.3635/1.1208/0.0213 while recall@8 was 0.753/0.743 and needle@8 was 0.72/0.67. Runtime was approximately 282 seconds per epoch; completing 200 epochs was projected at about 15.5 hours total. A follow-up target-only diagnostic over 24 unique held-out documents found that, conditional on the teacher retaining the needle leaf in top 8, its rolled-up ancestor also remained top 8 for 100.0%, 97.5%, 96.9%, 96.5%, 96.6%, and 97.2% of eligible layer-groups across leaf-through-coarse supervised levels; corresponding any-group layer rates were 100.0%, 98.5%, 97.6%, 97.1%, 96.9%, and 97.7%. The pasted output also contains a second `Starting: fresh training` block. Notebook audit found that the launch cell used a `RESUME_TRAINING` Boolean captured before training, so rerunning only that cell after checkpoints existed could incorrectly start fresh; the local notebook now rechecks `RESUME_OUT.exists()` immediately before every launch.
**Conclusion:** Stop the run and retain the saved epoch checkpoints for diagnosis. The displayed needle diagnostic is per-group while the new loss optimizes a per-layer union, so their inverse movement is not by itself proof that the loss is reversed; actual union and traversal metrics were omitted. Coarse needle eligibility drops only about 2-3% versus leaf eligibility on held-out targets and is not large enough to explain the failure. Evaluate saved checkpoints with real traversal before deciding whether all-level training helped. Do not rerun the old launch cell: it can restart fresh and overwrite rolling checkpoints; use the fixed notebook or an explicit `--resume`.

## 2026-07-26 — Tree-aware natural selector training implementation
**Question:** Can the next overnight run optimize the selector for the actual recursive natural-retrieval failure mode without materializing every tree level's training activations simultaneously?
**Config:** Added opt-in streamed all-level radix-2 supervision to `selector.train`, initialized independently at each level from fixed pooled node features and rolled-up teacher mass. The normalized objective is per-level KL plus broad top-k membership, a new hard weakest-positive/strongest-negative top-k boundary loss, and a new any-KV-group-per-layer evidence loss applied to the needle block's ancestor at each pruning level. `--offload-targets` caches FP16 Q/K and unnormalized grouped teacher mass in CPU RAM, transfers one document at a time, and performs FP32 normalization on the compute device; this prevents all 160+40 targets from occupying VRAM together without storing precision-reduced normalized rankings. New CLI/configuration and resume guards serialize the exact objective; legacy leaf-only training remains the default. Production-shape smoke tests used throwaway copies of the mixed gate and held-out targets solely for tensor-shape/memory validation; no resulting weights were saved or used.
**Number:** Focused hierarchy/loss/resume/cache tests passed 24/24; the full local suite passed 106 tests with 11 skipped. On the RTX 4070 development GPU, one 16K tree-aware training step traversed 8 discriminative levels in 2.219 seconds with 0.867GB peak allocated/1.063GB reserved, while one 32K step traversed 9 levels in 4.815 seconds with 3.141GB peak allocated/3.909GB reserved. The production offload path cached the same 32K document in FP16 on CPU, normalized it to FP32 after transfer, and completed 9 levels in 5.034 seconds with 3.142GB peak allocated. These timings are development diagnostics, not paper latency. The rebuilt `spruce_colab_train_source.zip` is 176,230 bytes with SHA-256 `7B896612C1DC716A96E8599BFCF98B7E1285CD1F9BDCA0F8C1FB4681AABC3F29`.
**Conclusion:** The missing hierarchy-aligned training path is implemented, tested, resumable, and within the 8GB memory ceiling at 32K. Run one mixed 160-natural/40-replay continuation from the current gate with beam-aligned K8, boundary ranking, and union evidence retention; do not claim an improvement until the held-out flat, traversal, and Triton generation evaluations are rerun.

## 2026-07-26 — Beam16/K18 fails to rescue natural retrieval
**Amended 2026-07-27.** Measured with the attention sink unforced (fixed 2026-07-27), so the
0/3 understates the sparse path — `learned` at `--sink-blocks 1` scores 8/25 on the full
held-out set versus 3/25 without. The conclusion that budget is not the root cause is
nonetheless confirmed by run 6 (K=10/18/32/64 all identical). The 0.8872x kernel speedup on the
16K Gallery case quoted here did not reproduce: every configuration in runs 3-7 beats dense at
16K (1.24-1.29x). Treat that figure as a K18-specific or warmup artifact, not a property of the
method.
**Question:** Is the 0/3 K10 natural retrieval failure primarily caused by an overly aggressive beam/selected-block budget?
**Config:** Colab NVIDIA L4 (23.66GB, compute capability 8.9); Linux; Python 3.12.13; PyTorch 2.11.0+cu128; Transformers 5.13.1; Triton 3.6.0; Qwen2.5-Coder-1.5B-Instruct in FP16; `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; the same three dense-verified held-out prompts at 16K shallow, 32K middle, and 32K deep; live radix-2 traversal; beam=16; K=18; local window=1; FP16 selector; four-layer selector chunks; `single_head` kernel with 8 warps/2 stages; one repeat; feature extraction and feature-file loading excluded.
**Number:** Dense exact retrieval remained 3/3 and sparse exact retrieval remained 0/3; answers matched 0/3 and mean sparse/dense fuzzy score was 0.4444/1.0. Sparse produced wrong plausible alternatives: `path/to/atlas/chk` instead of `state/atlas.chk`, `1981` instead of `17 April 1986`, and `Kepler Field 739` instead of `Kepler Field 731`; the last is an explicit near-miss distractor in the held-out prompt bank. Per-head evidence-block hit rates were 0.4821 Atlas, 0.5357 Gallery, and 0.7500 Observatory; any-group-per-layer rates were 0.7857, 0.7500, and 0.8571. Median kernel/live-prefill/live-total speedups were 1.2914x/1.1348x/1.1099x; sum-weighted kernel/live-prefill speedups were 1.2776x/1.1636x. The 16K Gallery case was slower sparse, at 0.8872x kernel and 0.8434x live prefill; the two 32K cases were faster. Peak allocated memory was 6.459GB sparse versus 6.219GB dense.
**Conclusion:** Doubling the traversal and selected-block budget does not repair natural retrieval, so K10 aggressiveness is not the root cause and K18 is not a viable fallback. The wrong but semantically close answers and incomplete evidence routing make selector/objective failure the leading diagnosis. Replay the exact same exported selected blocks through PyTorch sparse and Triton before editing the loss; parity would close the remaining kernel-attribution gap.

## 2026-07-26 — Natural160/replay40 live Triton retrieval failure
**Amended 2026-07-27 — the headline is partly a bug, not a gate failure.** This 0/3 was measured
with the attention sink competing for learned slots. With `--sink-blocks 1` the same `learned`
path scores 8/25 on the full 25-target held-out set (3/25 without), including its first 32K
wins. The conclusion "selector failure is the leading diagnosis" is now contradicted:
`oracle-needle` (perfect selector) and `teacher-top8` (distillation ceiling) both fail, and
conditional exactness given a densified evidence row is 0.62. Retrieval is not gate-limited.
**Question:** Does the mixed natural/replay gate preserve dense retrieval when its live tree routes drive the Triton sparse-prefill kernel on representative dense-verified natural prompts?
**Config:** User-run Colab development smoke using `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; Qwen2.5-Coder-1.5B-Instruct; three dense-verified held-out prompts chosen to cover 16K shallow, 32K middle, and 32K deep evidence; beam=8; K=10; local window=1; FP16 selector with four-layer chunks; `single_head` Triton kernel; FP16 model; 16 generated tokens; three repeats. Only the aggregated CSV was copied into the repository; the JSON with case answers, route-hit metrics, runtime metadata, and exact GPU identity is missing, so timings are development signal only.
**Number:** Dense exact retrieval was 3/3 and sparse exact retrieval was 0/3; no sparse answer matched its paired dense answer. On the 16,384-token case, dense/sparse-kernel/sparse-live prefill was 1.6228/1.2540/1.3140 seconds, giving 1.2941x kernel and 1.2350x live-prefill speedup; dense/sparse exact was 1/1 versus 0/1 and sparse fuzzy score was 0.0. Across the two approximately 32K cases, mean dense/sparse-kernel/sparse-live prefill was 3.9983/2.4783/2.6105 seconds, giving 1.6134x kernel and 1.5316x live-prefill speedup; dense/sparse exact was 2/2 versus 0/2 and mean sparse fuzzy score was 0.5.
**Conclusion:** The current gate fails practical natural retrieval at the lengths on which it was trained; the measured speedup is unusable at 0/3 sparse accuracy and blocks scaling. Prior Triton/reference parity plus weak structural evidence routing makes selector failure the leading diagnosis, but the missing per-case JSON prevents a definitive attribution. Reuse these exact three prompts for a beam16/K18 control and, if needed, a PyTorch-sparse-versus-Triton replay with identical selected blocks before changing the loss.

## 2026-07-26 — Natural-only versus natural160/replay40 held-out comparison
**Question:** Does adding the fixed 20% legacy replay curriculum damage the natural-only gate's routing on the dense-verified natural held-out distribution, or improve it enough to change the Stage-2 decision?
**Config:** Paired structural comparison of `natural_gate_lamt075_lamn025_k10_lr5e4_e300.pt` and `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; Qwen2.5-Coder-1.5B teacher features; 28 layers; feature/projection width 128/128; lambda_topk=0.75; lambda_needle=0.25; K10; 300 epochs; native 16K/32K; block=64; flat budgets and radix-2 traversal beams={1,2,4,8,16}. The natural-only run used the now-deduplicated 24-target glob; the mixed run included one duplicate Atlas row, but its reported means remain unchanged to three decimals after removing that duplicated value, and worst-case comparisons are unaffected.
**Number:** Mixed versus natural-only flat worst recall@8 was 0.745 versus 0.743 (+0.002), worst coverage/oracle ratio@8 0.969 versus 0.967 (+0.002), worst per-head needle ratio@8 0.620 versus 0.620, and worst any-group-per-layer needle ratio@8 0.846 versus 0.833 (+0.013). At live beam 8, mixed versus natural-only mean/worst recall@8 was 0.739/0.728 versus 0.736/0.726 (+0.003/+0.002), mean coverage/oracle was 0.623/0.685=0.909 versus 0.617/0.685=0.901, and mean per-head needle hit was 0.49 versus 0.45. At beam 16, mean/worst recall@8 was 0.893/0.881 versus 0.894/0.883 (-0.001/-0.002), mean coverage/oracle was 0.710/0.753=0.943 versus 0.709/0.753=0.942, and mean per-head needle hit was 0.66 versus 0.62. The mixed gate also improved narrow-beam top-1 recovery: mean traversal recall@1 was 0.574/0.771/0.846/0.904 at beams 1/2/4/8 versus 0.552/0.741/0.810/0.880 for natural-only.
**Conclusion:** The 20% replay curriculum preserves natural held-out routing and modestly improves teacher-mass, needle, and narrow-beam top-1 behavior, so there is no natural-distribution reason to prefer the natural-only checkpoint. It does not solve the shared exact top-8 recall ceiling: both fail the 0.95 Stage-2 proxy by roughly the same margin. Before selecting the mixed gate, rerun the six legacy held-out targets to test whether replay actually repairs the natural-only gate's previously measured legacy regression; then treat the remaining natural top-8 gap as a data/objective or metric-alignment problem rather than a traversal-only problem.

## 2026-07-26 — Natural160/replay40 gate held-out flat and tree evaluation
**Question:** Does the corrected 160-natural/40-replay checkpoint recover the frozen Stage-2 top-block recall target on the dense-verified natural held-out set, and how much additional loss comes from inference-time radix-2 traversal?
**Config:** Qwen2.5-Coder-1.5B teacher features; checkpoint `natural160_replay40_lamt075_lamn025_k10_lr5e4_e300.pt`; 28 layers; feature width/projection width 128/128; lambda_topk=0.75; lambda_needle=0.25; K10 training; 300 epochs; held-out dense-verified natural targets at native 16K/32K with block=64. Flat evaluation used k={1,2,4,8,16}; traversal used radix=2 with matched beam/budget={1,2,4,8,16}. The glob contained 25 rows but only 24 unique targets because one Atlas file was duplicated; worst-case metrics are unaffected, while traversal means give that case double weight.
**Number:** Flat scoring worst recall@8=0.745, worst coverage/oracle ratio@8=0.969, worst per-head needle ratio@8=0.620, and worst any-group-per-layer needle ratio@8=0.846; the evaluator failed its 0.95 recall@8 bar. Live traversal at beam 8 selected 7.9 blocks on average and produced mean/worst recall@8=0.739/0.728, mean coverage/oracle=0.623/0.685=0.909, and mean per-head needle hit=0.49. At beam 16 it selected 15.6 blocks on average and produced mean/worst recall@8=0.893/0.881, mean coverage/oracle=0.710/0.753=0.943, and mean per-head needle hit=0.66. Thus matched K=8 traversal lowers worst recall@8 by 0.017 versus flat top-8 scoring, while doubling the traversal budget raises worst recall@8 by 0.136 but still misses 0.95.
**Conclusion:** This checkpoint fails the frozen Stage-2 top-block recall proxy on the natural held-out set despite high flat teacher-mass coverage; KS1 is not cleared. The live tree is not the main K=8 failure—the learned ranking already tops out at worst recall@8=0.745—but greedy traversal compounds it slightly. Run the prior natural-only gate on the identical glob and both evaluators before attributing the mixed checkpoint's outcome to replay retention or deciding the next training change.

## 2026-07-25 — Cross-runtime CUDA RNG resume fix
**Question:** Can a mixed-gate resume checkpoint restore its RNG state after Colab starts a fresh CUDA runtime?
**Config:** User-reported resume of the 160-natural/40-replay selector run from its Drive checkpoint. `torch.load(..., map_location="cuda")` moved the saved CUDA RNG byte tensors to the GPU before `torch.cuda.set_rng_state_all` was called.
**Number:** Resume stopped before the next epoch with `TypeError: RNG state must be a torch.ByteTensor`. After normalizing each saved CUDA RNG state through `detach().cpu().to(torch.uint8)`, the focused resume/loss suite passed 6/6 and the full local suite passed 98 tests with 11 skipped.
**Conclusion:** The checkpoint is valid; only the loader was wrong. Patch `selector/train.py` in the active runtime and rerun the same `--resume` command. Fresh source archives now include the fix.

## 2026-07-25 — Compact 40-target legacy replay pool
**Question:** Can mixed natural training retain a strict 160:40 natural-to-old epoch ratio without loading all 120 legacy targets?
**Config:** Selected one full teacher target from each of 40 distinct old training cases; deterministic seed 20260725; balanced across native context scale and evidence position. Packaged without modifying the original 120-target directory.
**Number:** Replay pool contains exactly 40 files from 40 semantic cases: 20 approximately 16K and 20 approximately 32K; evidence-depth bins are 13 shallow (`d<0.33`), 14 middle, and 13 deep (`d>=0.67`). Archive size is 4,190,289,374 bytes; SHA-256 `CE3C6FB913EF054397A2D8AB96AD8E6FB902C3623D5B0DE026CE67E5BB3DE3A9`.
**Conclusion:** Train with all 160 natural targets plus this fixed 40-target replay directory and `--natural-fraction 0.8`. The grouped sampler then uses every file exactly once per epoch, avoiding both excess eager-load memory and changing replay composition across epochs.

## 2026-07-25 — Diversity-first 160-target source bank
**Question:** Can the natural training source be expanded enough that an exact 160-target teacher set does not rely on repeated seed variants of twelve semantic facts?
**Config:** Expanded `scripts/prompt_banks/natural_train.json` from 12 to 60 held-out-disjoint semantic cases. The 48 additions span 48 labeled document genres and vary answers across names, dates, times, paths, codes, versions, measurements, ranges, percentages, counts, classifications, procedures, and locations; every case includes three close distractors. Added global `--max-per-case` enforcement to the balanced screened-manifest selector using an exact capacity allocation across length/depth strata. No dense screening or teacher extraction was run locally.
**Number:** Bank validation: 60/60 unique case IDs, zero overlap with the six-case natural held-out bank, 60/60 needles contained in their evidence, and at least three distractors per case. Full local suite: 97 passed, 11 skipped. Rebuilt `spruce_colab_train_source.zip` contains 82 source/test files, including the 60-case bank and capped selector.
**Conclusion:** Screen one variant of all 60 cases at 16K/32K and three depths (360 candidates), then select exactly 160 with `--max-per-case 3`. This makes semantic diversity a hard dataset constraint rather than an incidental effect of selection order.

## 2026-07-25 — 160-target extraction exposes semantic-duplication problem
**Question:** Does scaling the existing 12-case natural bank to 160 targets by increasing prose seeds create the intended practical-data diversity?
**Config:** User-reported in-progress Colab extraction from the four-variant screened/selected manifest; native 16K/32K; multiple depths; Qwen2.5-Coder-1.5B-Instruct; full FP16 teacher targets. Each seed changes deterministic surrounding prose, but case evidence, question, answer, and near-miss distractors remain fixed.
**Number:** The extraction log shows repeated source cases such as `archive_box_finch`, `beacon_frequency_88_4`, and `clinic_cobalt_room_214` across seeds 20260725–20260727, lengths, and depths. Reported peak VRAM is 4.19GB at 16K and 5.28GB at 32K.
**Conclusion:** These are distinct documents but semantic duplicates, so completing 160 from only 12 cases is poor use of extraction budget and risks learning case/question templates. Stop treating seed variants as new semantic examples. Expand the natural training bank substantially, use one or at most two variants per case, then screen and extract a diversity-first set; retain already extracted files only as a small augmentation subset.

## 2026-07-25 — Exact 160-target screened natural-set builder
**Question:** Can an oversized dense-screening run be reduced to exactly 160 accepted natural prompts without concentrating selection in one length, depth, or semantic case?
**Config:** Added `scripts/select_screened_prompts.py`. It filters completed dense-accepted records, allocates an exact count evenly across requested-length/evidence-depth strata subject to available capacity, round-robins source cases within each stratum, and writes an accepted-only verified manifest consumable by full teacher extraction. Selection is deterministic by seed and rejects duplicate IDs or insufficient accepted supply. No screening or extraction was run locally.
**Number:** Focused screening/selection tests: 10 passed. Full local suite: 95 passed, 11 skipped; compile, CLI-help, and whitespace checks passed.
**Conclusion:** Screen an oversized four-variant 16K/32K natural candidate pool, select exactly 160 accepted prompts, then extract only that manifest into a clean target directory. The resulting directory has exactly 160 jobs if extraction completes without unrelated pre-existing files.

## 2026-07-25 — Natural-majority balanced replay sampler
**Question:** Can mixed selector training prioritize practical natural documents without discarding the old repeated-needle distribution entirely?
**Config:** Added grouped trainer inputs `--natural-targets` and `--replay-targets` plus `--natural-fraction`. Each epoch uses every natural target exactly once, samples only enough old targets to reach the requested fraction, and deterministically shuffles the combined order. Sampling cycles without replacement before repeating when necessary. Recursive globs now support arbitrary directory layouts produced by teacher-target ZIPs. Existing ungrouped `--targets` remains backward compatible. No gate training was run locally.
**Number:** At the recommended 0.8 fraction, eight natural documents produce two replay draws and ten updates per epoch. Focused grouped-sampling/evaluator/target suite: 12 passed. Full local suite: 92 passed, 11 skipped; compile, CLI-help, and whitespace checks passed.
**Conclusion:** The next gate can train on an 80/20 natural/replay curriculum independent of raw file counts, initialized from the original gate. This tests whether natural held-out gains can be retained while using old data only as regression regularization.

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
