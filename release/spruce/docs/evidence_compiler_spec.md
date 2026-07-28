# Evidence-compiler interface

The evidence compiler is a separate inference path from sparse attention.
`selected_blocks` remains unchanged. Candidate block IDs are used only as
source locations; the compiler returns to the original prompt text, gathers
the selected regions, repairs their boundaries, and gives a frozen Qwen model
a fresh ordinary dense prompt.

## Contract

Inputs:

- Exact reconstructed prompt text and tokenizer.
- Exact user prompt and final question.
- Source block IDs referring to the original tokenized prompt.
- The extraction block size.
- A non-negative source-block expansion radius.
- Boundary mode `block` or `paragraph`.

Output:

- Selected and expanded source block IDs, sorted and duplicate-free.
- Disjoint source spans in original document order.
- Exact source text for each span.
- Original token and character offsets for provenance.
- A conventional chat-formatted dense prompt containing the evidence spans
  followed by the original final question.
- Original, evidence, and compiled token counts.

The compiler must never:

- Read sparse hidden states or sparse K/V tensors.
- Insert text from omitted source regions except deterministic boundary
  expansion around a selected block.
- Include the saved system/chat wrapper or final question as evidence.
- Reorder source spans.
- Paraphrase, summarize, or silently truncate evidence.

## Diagnostic modes

- `oracle`: compile the known evidence block. This tests whether the assembled
  packet is readable when selection is perfect.
- `selector`: compile the gate's top-M final-reader blocks after filtering
  candidates to blocks that overlap the source document.

Both modes use the same compiler and fresh dense Qwen read. The oracle mode is
an attribution control, not a deployable result. The selector diagnostic uses
saved teacher-extraction features, so feature extraction cost is not a live
deployment measurement and must be reported separately.
