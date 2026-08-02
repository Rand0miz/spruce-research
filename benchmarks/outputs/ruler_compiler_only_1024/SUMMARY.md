# SPRUCE on RULER: compiler only

- generated: 2026-08-02T17:18:45.956301+00:00
- model (tokenizer only, no weights loaded): Qwen/Qwen2.5-Coder-1.5B-Instruct
- runtime: GPU present but unused: NVIDIA A100-SXM4-80GB
- source archive SHA-256: 03E971D58A110EB783BD62E245BC1187BF3F392DC537E537EFF1CD4A1E202C9E
- config: block 64, top-M 4, radius 1, beam 16, D=1024, boundary paragraph
- data: NVIDIA RULER synthetic generator, 20 samples per task/length
- query heuristic: last 3 non-empty lines of the RULER prompt
- wall clock: 7.5 minutes for 1546 samples

## What the number means

The ceiling scores the compiled packet with RULER's own metric, as if the
packet were the answer. A model that may only read the packet cannot
produce a string the packet does not contain, so the ceiling bounds the
end-to-end SPRUCE score from above. No model was run.

## Overall

- compiler ceiling 0.6966 over 1546 samples (4330 references)
- full-prompt ceiling 0.9968 (below 1.0 means the reference is not in the document; a data or query-extraction issue, not a compiler defect)
- 61.4% of samples have a perfect ceiling
- reference fate: ok 1888, data 1228, selector 1212, stitch 2
- stage survival: in document 0.716 -> selected 0.371 -> +radius 0.413 -> in packet 0.718
- median compiler cost 0.111 s/sample, median packet 8.03% of the prompt

## By task

| task | ceiling | prompt ceiling | selected | in packet | selector miss | stitch miss | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cwe | 1.000 | 1.000 | 0.007 | 1.000 | 0 | 0 | 120 |
| fwe | 1.000 | 1.000 | 0.931 | 1.000 | 0 | 0 | 120 |
| niah_multikey_1 | 0.792 | 1.000 | 0.675 | 0.792 | 25 | 0 | 120 |
| niah_multikey_2 | 0.725 | 1.000 | 0.650 | 0.725 | 33 | 0 | 120 |
| niah_multikey_3 | 0.539 | 1.000 | 0.278 | 0.539 | 52 | 1 | 115 |
| niah_multiquery | 0.754 | 1.000 | 0.633 | 0.754 | 118 | 0 | 120 |
| niah_multivalue | 0.771 | 1.000 | 0.653 | 0.771 | 108 | 0 | 118 |
| niah_single_1 | 1.000 | 1.000 | 0.958 | 1.000 | 0 | 0 | 119 |
| niah_single_2 | 0.719 | 1.000 | 0.640 | 0.719 | 32 | 0 | 114 |
| niah_single_3 | 0.925 | 1.000 | 0.875 | 0.925 | 9 | 0 | 120 |
| qa_1 | 0.308 | 1.000 | 0.095 | 0.259 | 289 | 0 | 120 |
| qa_2 | 0.300 | 0.958 | 0.158 | 0.300 | 79 | 0 | 120 |
| vt | 0.220 | 1.000 | 0.188 | 0.220 | 467 | 1 | 120 |

## By length

| length | ceiling | in packet | packet % of prompt | s/sample | n |
| --- | --- | --- | --- | --- | --- |
| 4096 | 0.857 | 0.826 | 56.39% | 0.025 | 255 |
| 8192 | 0.784 | 0.757 | 26.19% | 0.045 | 258 |
| 16384 | 0.738 | 0.732 | 11.67% | 0.085 | 257 |
| 32768 | 0.676 | 0.721 | 3.24% | 0.166 | 259 |
| 65536 | 0.588 | 0.653 | 1.58% | 0.337 | 258 |
| 131072 | 0.539 | 0.624 | 0.79% | 0.714 | 259 |

## Query heuristic sweep

Ceiling by QUERY_TAIL_LINES, averaged over lengths. A task that rises as the tail shortens was losing to the query heuristic, not to the selector or the stitch.

| task | tail=1 | tail=2 | tail=3 |
| --- | --- | --- | --- |
| cwe | 0.337 | 1.000 | 1.000 |
| fwe | 1.000 | 1.000 | 1.000 |
| niah_multikey_1 | 0.983 | 1.000 | 0.817 |
| niah_multikey_2 | 0.867 | 0.800 | 0.733 |
| niah_multikey_3 | 0.733 | 0.683 | 0.550 |
| niah_multiquery | 0.771 | 1.000 | 0.729 |
| niah_multivalue | 0.933 | 1.000 | 0.746 |
| niah_single_1 | 1.000 | 1.000 | 1.000 |
| niah_single_2 | 0.950 | 1.000 | 0.733 |
| niah_single_3 | 1.000 | 1.000 | 0.867 |
| qa_1 | 0.267 | 0.183 | 0.183 |
| qa_2 | 0.500 | 0.383 | 0.300 |
| vt | 0.220 | 0.220 | 0.227 |

## Attribution against the end-to-end run

- joined 1174 samples with ruler_v1
- correct with evidence present: 538
- **model side** (wrong, evidence was in the packet): 248
- **compiler side** (wrong, evidence was not in the packet): 376
- correct without the evidence: 12 (prior or format match, not retrieval)
- of 624 wrong answers, 39.7% are model side and 60.3% compiler side

## Caveats to carry into any writeup

- cwe and fwe reference common words that appear all over the document,
  so their ceiling is high for reasons unrelated to retrieval. They are
  a sanity check, not evidence.
- The query is the last 3 non-empty lines of the prompt, the same
  heuristic the end-to-end notebook uses. It pulls haystack text in
  alongside the real instruction, which is noise in the hashed query.
- A perfect ceiling does not mean the packet is easy to read. Evidence
  can be present and still be missed by the model; that is charged to
  the model here, which is the correct charge.
- Timings are CPU, no model loaded, on a Colab runtime. Development
  signal for the shape of the curve, not paper latency numbers.
