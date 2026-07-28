# Natural YaRN beam-16 result card

## Status

First run of the sealed, unscreened natural prompt bank. The selector and
analysis were frozen before inference.

## Primary result

- Model: Qwen2.5-Coder-1.5B-Instruct
- GPU: one NVIDIA A100-SXM4-40GB
- Precision: FP16
- Contexts: 16K–128K in 16K increments
- Semantic cases: 12
- Evidence depths: 0.1, 0.5, 0.9
- Paired rows: 288
- Dense exact: 192/288 (66.67%)
- Compiler exact: 237/288 (82.29%)
- Accuracy delta: +15.625 percentage points
- Paired outcomes: 63 compiler-only / 18 dense-only
- Exact McNemar p: 5.2044e-7
- Sum-weighted fully charged speedup: 9.5809x
- 128K speedup: 14.0828x
- Median compiled model input: 1,849.5 tokens

## Guardrail

The 10,000-draw semantic-case cluster bootstrap interval for the accuracy
delta is [-5.556, +34.722] percentage points. It crosses zero because the
suite has only 12 independent semantic cases and one large negative case.

The result supports the controlled retrieval distribution. It does not
establish general superiority over dense reading.

## Memory interpretation

Peak allocated GPU memory is path-attributable: compiler allocation remains
near 3.27 GB while dense rises to 15.58 GB at 128K.

Peak reserved memory is not a valid path comparison in this run. Dense and
compiled requests alternated in one process, so PyTorch retained the dense
allocator high-water mark for later compiler requests.

## Files

- `combined_report.json`: complete paired report and raw repetition samples
- `locked_config.json`: frozen method configuration
- `decision.json`: claim gates and result paths
- `tables/`: raw and aggregated CSVs
- `figures/`: PNG and vector-PDF paper figures
- `SUMMARY.md`: compact generated summary
