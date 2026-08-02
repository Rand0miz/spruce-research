# SPRUCE cached-index tests

- generated: 2026-08-01T21:52:43.902933+00:00
- GPU: NVIDIA L4
- prompts: 288

## Headline

- dense median request: 14.586 s
- cold median request: 0.570 s (30.22x vs dense)
- cached median query: 0.296 s (57.71x vs dense, 1.91x vs cold)
- cached latency slope: +0.8519 ms per Ki token (R2=0.928)
- traversal linear in tree depth: +0.2129 ms per level (R2=0.969)
- index build linear in n: +4.0697 us per token (R2=0.999)

Cold and cached accuracy are identical by construction; the run asserts
identical selected blocks and identical answers on every case.

## Per length

| Ki | dense s | cold s | cached s | build s | cached x | k* |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1.795 | 0.297 | 0.243 | 0.054 | 6.93x | 1.0 |
| 32 | 4.355 | 0.396 | 0.282 | 0.114 | 15.38x | 1.0 |
| 48 | 7.814 | 0.444 | 0.269 | 0.180 | 27.83x | 1.0 |
| 64 | 12.035 | 0.530 | 0.293 | 0.241 | 39.39x | 1.0 |
| 80 | 17.125 | 0.606 | 0.308 | 0.304 | 54.97x | 1.0 |
| 96 | 22.936 | 0.715 | 0.330 | 0.383 | 69.70x | 1.0 |
| 112 | 31.529 | 0.786 | 0.337 | 0.449 | 92.82x | 1.0 |
| 128 | 43.762 | 0.850 | 0.340 | 0.520 | 128.97x | 1.0 |
