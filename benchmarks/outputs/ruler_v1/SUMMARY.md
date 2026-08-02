# SPRUCE on RULER

- generated: 2026-08-02T08:22:57.929070+00:00
- model: Qwen/Qwen2.5-Coder-1.5B-Instruct | GPU: NVIDIA L4 | feature_dim: 1024
- data: NVIDIA RULER synthetic generator, 50 samples per task/length
- query heuristic: last 3 non-empty lines of the RULER prompt

## Overall

- dense 0.5431, SPRUCE 0.5934, delta +0.0503 over 2875 samples
- per-sample: SPRUCE better on 633, dense better on 427

## By task

| task | dense | SPRUCE | delta | n |
| --- | --- | --- | --- | --- |
| cwe | 0.179 | 0.184 | +0.005 | 200 |
| fwe | 0.123 | 0.535 | +0.412 | 200 |
| niah_multikey_1 | 0.959 | 0.829 | -0.130 | 246 |
| niah_multikey_2 | 0.808 | 0.784 | -0.024 | 250 |
| niah_multikey_3 | 0.005 | 0.038 | +0.033 | 211 |
| niah_multiquery | 0.479 | 0.821 | +0.342 | 207 |
| niah_multivalue | 0.391 | 0.739 | +0.348 | 232 |
| niah_single_1 | 0.865 | 1.000 | +0.135 | 245 |
| niah_single_2 | 0.996 | 0.802 | -0.194 | 237 |
| niah_single_3 | 1.000 | 0.943 | -0.057 | 247 |
| qa_1 | 0.265 | 0.225 | -0.040 | 200 |
| qa_2 | 0.325 | 0.295 | -0.030 | 200 |
| vt | 0.295 | 0.204 | -0.091 | 200 |

## By length

| length | dense | SPRUCE | delta | n |
| --- | --- | --- | --- | --- |
| 4096 | 0.534 | 0.670 | +0.136 | 624 |
| 8192 | 0.566 | 0.576 | +0.010 | 628 |
| 16384 | 0.502 | 0.576 | +0.074 | 635 |
| 32768 | 0.490 | 0.545 | +0.056 | 640 |
| 65536 | 0.691 | 0.608 | -0.083 | 348 |

> PARTIAL RUN: 59 of 78 task-length pairs are complete. Do not quote these as the RULER result.

## Caveats to carry into any writeup

- vt, cwe and fwe are aggregation tasks whose instruction names no
  distinctive term for the sketch to hash. Weak SPRUCE results there are
  a scope finding, not a bug to tune away.
- The query is extracted by a trailing-lines heuristic, not supplied by
  RULER. Any task whose instruction sits elsewhere is mis-served.
- Data generation is RULER own; the serving loop is ours, so these are not
  directly comparable to published RULER leaderboard numbers.
