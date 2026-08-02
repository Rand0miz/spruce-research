# SPRUCE cached-index tests

- generated: 2026-08-02T06:07:32.782066+00:00
- GPU: NVIDIA L4
- prompts: 288

## Headline

- dense median request: 14.654 s
- cold median request: 0.543 s (32.37x vs dense)
- cached median query: 0.264 s (67.51x vs dense, 2.09x vs cold)
- cached latency slope: +0.0267 ms per Ki token (R2=0.019)
- traversal linear in tree depth: +0.2724 ms per level (R2=0.986)
- index build linear in n: +4.1028 us per token (R2=0.999)

Cold and cached accuracy are identical by construction; the run asserts
identical selected blocks and identical answers on every case.

## Per length

| Ki | dense s | cold s | cached s | build s | cached x | k* |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1.841 | 0.316 | 0.258 | 0.059 | 7.31x | 1.0 |
| 32 | 4.383 | 0.383 | 0.264 | 0.121 | 16.83x | 1.0 |
| 48 | 7.848 | 0.444 | 0.266 | 0.188 | 29.88x | 1.1 |
| 64 | 12.059 | 0.512 | 0.263 | 0.253 | 45.19x | 1.0 |
| 80 | 17.199 | 0.561 | 0.244 | 0.315 | 65.77x | 1.0 |
| 96 | 22.954 | 0.650 | 0.267 | 0.393 | 84.21x | 1.0 |
| 112 | 31.760 | 0.711 | 0.265 | 0.456 | 118.12x | 1.0 |
| 128 | 43.892 | 0.775 | 0.265 | 0.529 | 170.29x | 1.0 |
