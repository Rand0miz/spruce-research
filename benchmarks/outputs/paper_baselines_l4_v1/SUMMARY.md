# SPRUCE paper baselines on the fixed L4

- generated: 2026-08-02T20:40:58+00:00
- GPU: None
- model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- feature_dim: 1024, M=4, beam=16, B=64
- rows: 2304 (status completed)

## Per arm

| phase / yarn / arm | n | exact | median req s | median tok | peak alloc GB | expanded recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baselines|yarn4|bm25 | 288 | 254/288 (0.8819) | 0.5631 | 1904 | 3.278 | 1.0000 |
| baselines|yarn4|flat | 288 | 254/288 (0.8819) | 0.5641 | 1882 | 3.276 | 1.0000 |
| baselines|yarn4|lead | 288 | 0/288 (0.0000) | 0.4280 | 594 | 3.154 | 0.0000 |
| baselines|yarn4|random | 288 | 12/288 (0.0417) | 0.5484 | 1900 | 3.278 | 0.0347 |
| baselines|yarn4|stride | 288 | 3/288 (0.0104) | 0.5650 | 1898 | 3.278 | 0.0000 |
| baselines|yarn4|tail | 288 | 1/288 (0.0035) | 0.4524 | 554 | 3.150 | 0.0000 |
| baselines|yarn4|tree | 288 | 255/288 (0.8854) | 0.5667 | 1894 | 3.277 | 1.0000 |
| yarn|yarn1|dense | 72 | 54/72 (0.7500) | 3.2577 | 24538 | 5.438 | n/a |
| yarn|yarn1|tree | 72 | 63/72 (0.8750) | 0.3513 | 1820 | 3.271 | 1.0000 |
| yarn|yarn4|dense | 72 | 55/72 (0.7639) | 3.2594 | 24538 | 5.437 | n/a |
| yarn|yarn4|tree | 72 | 64/72 (0.8889) | 0.3427 | 1820 | 3.271 | 1.0000 |

## Paired against dense, same phase and YaRN factor

| arm | n | both | arm only | dense only | neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| yarn|yarn4|tree | 72 | 55 | 9 | 0 | 8 |
| yarn|yarn1|tree | 72 | 52 | 11 | 2 | 7 |

## Reading this

- The yarn phase is a confound check. If dense at factor 1 scores
  materially above dense at factor 4 on the same prompts, part of
  the published dense-versus-compiled gap is rotary scaling.
- The baselines phase is a matched-budget check. Every arm spends
  the same block budget through the same compiler, so a naive arm
  that matches the tree means the result is packet size, not
  selection.
- Memory columns are the L4 replacement for the superseded A100
  figure measured at feature_dim 512.
