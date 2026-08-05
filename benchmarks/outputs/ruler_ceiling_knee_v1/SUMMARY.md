# SPRUCE compiler ceiling: beam x budget knee

- generated: 2026-08-02T22:07:40+00:00
- samples: 1560 (status completed)
- fixed: tail=3, D=4096, B=64, radius=1
- full-prompt ceiling: 0.9968

Compiler-only. No model weights, no GPU. Ceiling is what a perfect
model could score on the compiled packet, so it is the model-
independent measure of whether the compiler is the bottleneck.

## Pre-declared decision rule

- the compiler is not the bottleneck when every in-scope task-length cell ceiling is at least 0.9500
- target 0.9 + margin 0.05 = **0.9500000000000001**
- in scope: niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_multiquery, niah_multivalue, niah_single_1, niah_single_2, niah_single_3, qa_1, qa_2

## Per arm

| arm | beam | M | ceiling | in-scope | min cell | worst cell | packet frac | passes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| beam16_M9 | 16 | 9 | 0.8541 | 0.8385 | 0.2500 | qa_1@131072 | 0.1592 | no |
| beam16_M16 | 16 | 16 | 0.8687 | 0.8546 | 0.3000 | niah_multikey_3@131072 | 0.2348 | no |
| beam32_M9 | 32 | 9 | 0.8831 | 0.8762 | 0.2000 | qa_1@131072 | 0.1610 | no |
| beam32_M16 | 32 | 16 | 0.9057 | 0.9027 | 0.3500 | qa_1@131072 | 0.2504 | no |
| beam32_M32 | 32 | 32 | 0.9247 | 0.9250 | 0.4000 | qa_1@131072 | 0.3658 | no |
| beam64_M9 | 64 | 9 | 0.8839 | 0.8773 | 0.2000 | qa_1@131072 | 0.1600 | no |
| beam64_M16 | 64 | 16 | 0.9078 | 0.9054 | 0.3000 | qa_1@131072 | 0.2473 | no |
| beam64_M32 | 64 | 32 | 0.9334 | 0.9363 | 0.4000 | qa_2@131072 | 0.3945 | no |

## Why references are still missed

`selector_outside_beam` is fixable only by a wider beam;
`selector_in_beam` only by a larger M; `data_absent` by neither —
it is a property of the benchmark, not of the compiler, and should
be excluded before computing how much headroom is really left.

| arm | ok | data absent | in beam | outside beam | stitch |
| --- | ---: | ---: | ---: | ---: | ---: |
| beam16_M9 | 2595 | 1228 | 36 | 490 | 1 |
| beam16_M16 | 2662 | 1228 | 0 | 459 | 1 |
| beam32_M9 | 2657 | 1228 | 81 | 383 | 1 |
| beam32_M16 | 2735 | 1228 | 25 | 361 | 1 |
| beam32_M32 | 2816 | 1228 | 0 | 306 | 0 |
| beam64_M9 | 2660 | 1228 | 102 | 359 | 1 |
| beam64_M16 | 2744 | 1228 | 37 | 340 | 1 |
| beam64_M32 | 2844 | 1228 | 0 | 278 | 0 |

## Recommendation

- **No arm clears the rule.** beam32_M32 reaches the highest in-scope cell floor, so the compiler is still the bottleneck; widen the sweep before spending GPU time.
- ceiling 0.9247, in-scope cell floor 0.4000, median packet 0.3658 of the source prompt
- worst failing cells: qa_1@131072, qa_2@131072, qa_2@65536, niah_multikey_3@131072, qa_1@65536, qa_2@32768, qa_2@16384, qa_1@32768
