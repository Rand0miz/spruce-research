# Cached RULER compiler factorial ablation

- generated: 2026-08-02T18:45:35.981985+00:00
- complete: True
- samples: 1546
- baseline: `t3_d1024_m4`
- model weights: none (tokenizer and compiler only)

## Overall

| arm | ceiling | perfect samples | selector misses | packet % | vs baseline | wins/losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `t2_d1024_m4` | 0.8021 | 78.7% | 878 | 100.08% | +0.1055 | 292/26 |
| `t2_d1024_m9` | 0.8823 | 85.9% | 389 | 100.08% | +0.1857 | 443/15 |
| `t2_d4096_m4` | 0.8329 | 81.7% | 806 | 100.08% | +0.1363 | 338/24 |
| `t2_d4096_m9` | 0.8982 | 86.8% | 406 | 100.08% | +0.2016 | 467/10 |
| `t3_d1024_m4` | 0.6966 | 61.4% | 1212 | 8.03% | +0.0000 | 0/0 |
| `t3_d1024_m9` | 0.8112 | 75.2% | 591 | 16.03% | +0.1146 | 283/0 |
| `t3_d4096_m4` | 0.7734 | 70.6% | 1001 | 7.09% | +0.0769 | 208/41 |
| `t3_d4096_m9` | 0.8535 | 81.2% | 525 | 15.80% | +0.1569 | 402/18 |

## Joined end-to-end attribution

| arm | joined | compiler-side misses | model-side misses | compiler share of wrong |
| --- | ---: | ---: | ---: | ---: |
| `t2_d1024_m4` | 1174 | 191 | 433 | 30.6% |
| `t2_d1024_m9` | 1174 | 117 | 507 | 18.8% |
| `t2_d4096_m4` | 1174 | 160 | 464 | 25.6% |
| `t2_d4096_m9` | 1174 | 106 | 518 | 17.0% |
| `t3_d1024_m4` | 1174 | 376 | 248 | 60.3% |
| `t3_d1024_m9` | 1174 | 198 | 426 | 31.7% |
| `t3_d4096_m4` | 1174 | 267 | 357 | 42.8% |
| `t3_d4096_m9` | 1174 | 144 | 480 | 23.1% |

## Exact caching

- tokenizations/layouts: 1546 actual versus 12368 naive
- index builds: 6184 actual versus 12368 naive
- beam traversals: 6184 actual versus 12368 naive
- The low-M route is the asserted prefix of the cached high-M ranking for every (tail,D) configuration.
- baseline comparisons against ruler_compiler_only_1024: 0 checked, 0 mismatches

## Interpretation guardrails

- Compare D only while tail and M are fixed.
- Compare M only while tail and D are fixed.
- `cwe` and `fwe` common-word ceilings are sanity checks, not retrieval evidence.
- This is compiler-only; it cannot clear end-to-end RULER or Stage-3 KS1.
