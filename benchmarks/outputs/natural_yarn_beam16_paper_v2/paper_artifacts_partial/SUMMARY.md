# SPRUCE unscreened natural YaRN sweep

## Overall paired result

- Paired prompts: 288
- Dense exact: 192/288 (66.667%)
- Compiler exact: 255/288 (88.542%)
- Accuracy delta: +21.875%
- Compiler-only / dense-only: 63 / 0
- Exact McNemar p: 2.1684e-19
- Semantic-case cluster bootstrap delta 95% interval: [+10.061%, +34.384%]
- Fully charged sum-weighted speedup: 31.874x
- Expanded evidence recall: 100.000%

## By requested context length

| Ki tokens | N | Dense | Compiler | Delta | Compiler-only | Dense-only | Speedup | Expanded recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 36 | 77.8% | 88.9% | +11.1% | 4 | 0 | 5.82x | 100.0% |
| 32 | 36 | 75.0% | 88.9% | +13.9% | 5 | 0 | 11.45x | 100.0% |
| 48 | 36 | 66.7% | 86.1% | +19.4% | 7 | 0 | 17.46x | 100.0% |
| 64 | 36 | 63.9% | 91.7% | +27.8% | 10 | 0 | 23.22x | 100.0% |
| 80 | 36 | 55.6% | 86.1% | +30.6% | 11 | 0 | 29.40x | 100.0% |
| 96 | 36 | 61.1% | 88.9% | +27.8% | 10 | 0 | 33.95x | 100.0% |
| 112 | 36 | 66.7% | 91.7% | +25.0% | 9 | 0 | 43.08x | 100.0% |
| 128 | 36 | 66.7% | 86.1% | +19.4% | 7 | 0 | 55.49x | 100.0% |

## Interpretation guardrails

- Prompts were generated from the sealed paper bank without dense screening.
- Prompt synthesis and model loading are harness setup, not request latency.
- Both modes use the same static YaRN configuration and generated prompt.
- Raw paired McNemar treats prompt rows independently; the case-clustered bootstrap is the more conservative semantic-diversity check.
- A positive compiler-minus-dense result supports this controlled natural-retrieval distribution only; it is not a general quality claim.
