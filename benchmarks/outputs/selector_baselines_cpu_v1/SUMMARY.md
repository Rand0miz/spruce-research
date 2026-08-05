# SPRUCE selector baselines (CPU, selection level)

- generated: 2026-08-02T20:07:24+00:00
- prompts: 288
- tokenizer: Qwen/Qwen2.5-Coder-1.5B-Instruct
- feature_dim: 1024, M=4, beam=16, B=64

No model weights are loaded. These are evidence-recall and
packet-size numbers, not generated accuracy.

## Per arm

| arm | direct recall | expanded recall | median packet tok | median compression | median select s | blocks scored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tree | 0.9028 | 1.0000 | 1892 | 0.0244 | 0.001325 | 228 |
| flat | 0.9028 | 1.0000 | 1882 | 0.0246 | 0.001321 | 1152 |
| bm25 | 0.9340 | 1.0000 | 1904 | 0.0248 | 0.008876 | 1152 |
| lead | 0.0000 | 0.0000 | 594 | 0.0082 | 0.004476 | 0 |
| tail | 0.0000 | 0.0000 | 554 | 0.0075 | 0.004387 | 0 |
| stride | 0.0000 | 0.0000 | 1898 | 0.0261 | 0.004365 | 0 |
| random | 0.0000 | 0.0347 | 1900 | 0.0245 | 0.004596 | 0 |

## Tree versus flat scan

- identical block set: 0.5486 of 288 prompts
- identical ranked order: 0.5486
- top-1 match: 1.0000
- mean Jaccard: 0.7767

If these rates are 1.0 the tree is a cost result, not an
accuracy result, and the paper must frame it that way.

## Per length, expanded evidence recall

| length | tree | flat | bm25 | lead | tail | stride | random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1111 |
| 32768 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 |
| 49152 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 65536 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 81920 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 98304 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 114688 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 131072 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
