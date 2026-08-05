# Paired RULER end-to-end: D=1024 vs D=4096

- generated: 2026-08-02T20:27:21.497629+00:00
- GPU: NVIDIA L4
- complete: False
- samples: 1471
- fixed configuration: tail=3, M=9, beam=16, B=64
- routes: cached factorial routes where available

## Overall

- D=1024: 0.5567
- D=4096: 0.6009
- paired delta: +0.0442 (paired bootstrap 95% CI +0.0295 to +0.0591)
- wins/ties/losses for D=4096: 179 / 1220 / 72

## By length

| length | D1024 | D4096 | delta | wins/ties/losses | n |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 0.682 | 0.680 | -0.001 | 6/233/4 | 243 |
| 8192 | 0.627 | 0.629 | +0.002 | 15/214/14 | 243 |
| 16384 | 0.577 | 0.619 | +0.041 | 27/206/12 | 245 |
| 32768 | 0.553 | 0.603 | +0.050 | 33/205/12 | 250 |
| 65536 | 0.498 | 0.596 | +0.098 | 49/180/11 | 240 |
| 131072 | 0.406 | 0.482 | +0.075 | 49/182/19 | 250 |

## By task

| task | D1024 | D4096 | delta | n |
| --- | ---: | ---: | ---: | ---: |
| cwe | 0.099 | 0.099 | +0.000 | 119 |
| fwe | 0.442 | 0.417 | -0.025 | 120 |
| niah_multikey_1 | 0.867 | 0.925 | +0.058 | 120 |
| niah_multikey_2 | 0.708 | 0.833 | +0.125 | 120 |
| niah_multikey_3 | 0.033 | 0.054 | +0.022 | 92 |
| niah_multiquery | 0.823 | 0.904 | +0.081 | 120 |
| niah_multivalue | 0.828 | 0.908 | +0.080 | 106 |
| niah_single_1 | 1.000 | 1.000 | +0.000 | 118 |
| niah_single_2 | 0.816 | 0.921 | +0.105 | 76 |
| niah_single_3 | 0.942 | 0.983 | +0.042 | 120 |
| qa_1 | 0.242 | 0.283 | +0.042 | 120 |
| qa_2 | 0.233 | 0.292 | +0.058 | 120 |
| vt | 0.213 | 0.220 | +0.007 | 120 |

> PARTIAL: 65/78 pairs complete. Do not quote as the final result.

## Interpretation guardrails

- Both arms use tail=3 and M=9; only feature width changes.
- Cached routes make this an accuracy/throughput run, not cold-request latency.
- Non-identical arms are generated together; paired batch time is not per-arm latency.
- RULER data generation is NVIDIA's; the serving loop is SPRUCE's.
- Record complete results in LOG.md before quoting them elsewhere.
