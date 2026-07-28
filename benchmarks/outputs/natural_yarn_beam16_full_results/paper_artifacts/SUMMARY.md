# SPRUCE unscreened natural YaRN sweep

## Overall paired result

- Paired prompts: 288
- Dense exact: 192/288 (66.667%)
- Compiler exact: 237/288 (82.292%)
- Accuracy delta: +15.625%
- Compiler-only / dense-only: 63 / 18
- Exact McNemar p: 5.2044e-07
- Semantic-case cluster bootstrap delta 95% interval: [-5.556%, +34.722%]
- Fully charged sum-weighted speedup: 9.581x
- Expanded evidence recall: 88.889%

## By requested context length

| Ki tokens | N | Dense | Compiler | Delta | Compiler-only | Dense-only | Speedup | Expanded recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 36 | 77.8% | 83.3% | +5.6% | 4 | 2 | 2.49x | 94.4% |
| 32 | 36 | 75.0% | 83.3% | +8.3% | 5 | 2 | 4.11x | 91.7% |
| 48 | 36 | 66.7% | 80.6% | +13.9% | 7 | 2 | 5.78x | 88.9% |
| 64 | 36 | 63.9% | 80.6% | +16.7% | 8 | 2 | 7.42x | 86.1% |
| 80 | 36 | 55.6% | 83.3% | +27.8% | 12 | 2 | 9.18x | 88.9% |
| 96 | 36 | 61.1% | 80.6% | +19.4% | 10 | 3 | 10.62x | 86.1% |
| 112 | 36 | 66.7% | 86.1% | +19.4% | 9 | 2 | 12.43x | 88.9% |
| 128 | 36 | 66.7% | 80.6% | +13.9% | 8 | 3 | 14.08x | 86.1% |

## Interpretation guardrails

- Prompts were generated from the sealed paper bank without dense screening.
- Prompt synthesis and model loading are harness setup, not request latency.
- Both modes use the same static YaRN configuration and generated prompt.
- Raw paired McNemar treats prompt rows independently; the case-clustered bootstrap is the more conservative semantic-diversity check.
- A positive compiler-minus-dense result supports this controlled natural-retrieval distribution only; it is not a general quality claim.
