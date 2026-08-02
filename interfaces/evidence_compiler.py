"""Compile scattered selected blocks into a short, ordinary dense-read prompt.

The compiler deliberately returns to the original prompt text. It does not
reuse sparse hidden states, K/V tensors, or summaries of omitted regions.
Selected blocks identify source locations; their text is clipped to the
document, expanded to coherent boundaries, merged in source order, and then
wrapped as a conventional chat prompt for a fresh dense Qwen pass.
"""
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


EVIDENCE_COMPILER_SYSTEM_PROMPT = (
    "Answer the user's final question using only the supplied evidence excerpts. "
    "The excerpts preserve source order and their labels are provenance metadata. "
    "Follow the requested answer format exactly. Do not continue, summarize, or "
    "rewrite the excerpts."
)


@dataclass(frozen=True)
class PromptLayout:
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    document_char_start: int
    document_char_end: int
    document_token_start: int
    document_token_end: int


@dataclass(frozen=True)
class EvidenceSpan:
    source_token_start: int
    source_token_end: int
    source_char_start: int
    source_char_end: int
    source_blocks: tuple[int, ...]
    text: str

    def to_dict(self) -> dict:
        return {
            "source_token_start": self.source_token_start,
            "source_token_end": self.source_token_end,
            "source_char_start": self.source_char_start,
            "source_char_end": self.source_char_end,
            "source_blocks": list(self.source_blocks),
            "text_chars": len(self.text),
        }


@dataclass(frozen=True)
class EvidencePacket:
    prompt: str
    content: str
    spans: tuple[EvidenceSpan, ...]
    selected_blocks: tuple[int, ...]
    expanded_blocks: tuple[int, ...]
    original_prompt_tokens: int
    evidence_source_tokens: int
    compiled_prompt_tokens: int

    @property
    def compression_fraction(self) -> float:
        return self.compiled_prompt_tokens / max(1, self.original_prompt_tokens)

    def metadata(self) -> dict:
        return {
            "selected_blocks": list(self.selected_blocks),
            "expanded_blocks": list(self.expanded_blocks),
            "span_count": len(self.spans),
            "spans": [span.to_dict() for span in self.spans],
            "original_prompt_tokens": self.original_prompt_tokens,
            "evidence_source_tokens": self.evidence_source_tokens,
            "compiled_prompt_tokens": self.compiled_prompt_tokens,
            "compression_fraction": self.compression_fraction,
        }


def normalize_block_ids(
        block_ids: Iterable[int], total_blocks: int) -> tuple[int, ...]:
    """Return stable source-order block IDs, rejecting invalid locations."""
    if int(total_blocks) < 1:
        raise ValueError(f"total_blocks must be >= 1, got {total_blocks}")
    normalized = []
    for raw in block_ids:
        block = int(raw)
        if not 0 <= block < int(total_blocks):
            raise ValueError(
                f"source block {block} outside [0,{int(total_blocks)})")
        normalized.append(block)
    unique = tuple(sorted(set(normalized)))
    if not unique:
        raise ValueError("at least one source block is required")
    return unique


def expand_block_ids(
        block_ids: Iterable[int], total_blocks: int,
        radius: int = 0) -> tuple[int, ...]:
    """Expand every selected block by a clamped source-order neighborhood."""
    if int(radius) < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    selected = normalize_block_ids(block_ids, total_blocks)
    expanded = set()
    for block in selected:
        expanded.update(range(
            max(0, block - int(radius)),
            min(int(total_blocks), block + int(radius) + 1),
        ))
    return tuple(sorted(expanded))


def document_block_ids(
        layout: PromptLayout, block_size: int) -> tuple[int, ...]:
    """Blocks with at least one token overlapping the source document."""
    if int(block_size) < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if layout.document_token_end <= layout.document_token_start:
        return ()
    first = layout.document_token_start // int(block_size)
    last = (layout.document_token_end - 1) // int(block_size)
    return tuple(range(first, last + 1))


def _as_flat_ints(values) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one tokenized prompt")
        values = values[0]
    return tuple(int(value) for value in values)


def _as_offsets(values) -> tuple[tuple[int, int], ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(
            values[0][0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one offset-mapped prompt")
        values = values[0]
    return tuple((int(start), int(end)) for start, end in values)


def token_range_for_char_range(
        offsets: Sequence[tuple[int, int]], char_start: int,
        char_end: int, *, search_start: int = 0,
        search_end: int | None = None) -> tuple[int, int]:
    """Return the minimal token range overlapping a half-open char range.

    ``search_start``/``search_end`` bound the scan to a token window. The
    default scans everything, which is O(n) in prompt tokens; callers that
    already know roughly where the span lives pass a window so a per-query
    stitch does not walk the whole document. Offsets are not reliably
    monotonic -- chat-template specials carry (0, 0) -- so this stays a scan
    over a bounded slice rather than a bisect over the whole list.
    """
    if not 0 <= int(char_start) < int(char_end):
        raise ValueError(
            f"invalid character range [{char_start},{char_end})")
    lo = max(0, int(search_start))
    hi = len(offsets) if search_end is None else min(
        len(offsets), int(search_end))
    overlapping = [
        lo + index
        for index, (start, end) in enumerate(offsets[lo:hi])
        if end > start and end > int(char_start) and start < int(char_end)
    ]
    if not overlapping:
        if lo > 0 or hi < len(offsets):
            # The window guess was wrong. Fall back to the full scan rather
            # than report a locatable span as unlocatable.
            return token_range_for_char_range(offsets, char_start, char_end)
        raise ValueError(
            f"no tokenizer offsets overlap [{char_start},{char_end})")
    return overlapping[0], overlapping[-1] + 1


def locate_prompt_layout(
        tokenizer, full_prompt: str, user_prompt_text: str,
        question: str) -> PromptLayout:
    """Locate the document inside a saved plain or chat-formatted prompt."""
    if not isinstance(full_prompt, str) or not full_prompt:
        raise ValueError("full_prompt must be a non-empty string")
    if not isinstance(user_prompt_text, str) or not user_prompt_text:
        raise ValueError("user_prompt_text must be a non-empty string")
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a non-empty string")
    if not user_prompt_text.endswith(question):
        raise ValueError("user_prompt_text must end with the exact question")
    user_start = full_prompt.rfind(user_prompt_text)
    if user_start < 0:
        raise ValueError("user_prompt_text was not found in full_prompt")
    document_end = user_start + len(user_prompt_text) - len(question)
    if document_end <= user_start:
        raise ValueError("the source document is empty")

    encoded = tokenizer(full_prompt, return_offsets_mapping=True)
    input_ids = _as_flat_ints(encoded["input_ids"])
    offsets = _as_offsets(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError(
            "tokenizer returned different input-id and offset counts")
    token_start, token_end = token_range_for_char_range(
        offsets, user_start, document_end)
    return PromptLayout(
        input_ids=input_ids,
        offsets=offsets,
        document_char_start=user_start,
        document_char_end=document_end,
        document_token_start=token_start,
        document_token_end=token_end,
    )


def _contiguous_runs(values: Sequence[int]) -> list[tuple[int, ...]]:
    runs = []
    current = []
    for value in values:
        if current and value != current[-1] + 1:
            runs.append(tuple(current))
            current = []
        current.append(value)
    if current:
        runs.append(tuple(current))
    return runs


def _char_range_for_tokens(
        offsets: Sequence[tuple[int, int]], token_start: int, token_end: int,
        allowed_start: int, allowed_end: int) -> tuple[int, int] | None:
    token_start = max(0, int(token_start))
    token_end = min(len(offsets), int(token_end))
    overlapping = [
        (max(start, allowed_start), min(end, allowed_end))
        for start, end in offsets[token_start:token_end]
        if end > start and end > allowed_start and start < allowed_end
    ]
    if not overlapping:
        return None
    return min(start for start, _ in overlapping), max(
        end for _, end in overlapping)


# A paragraph break is expected within a few blocks of any selected span. The
# search is bounded so one stitch cannot scan the whole document, and so the
# no-delimiter case cannot silently swallow it either.
PARAGRAPH_SEARCH_CHARS = 8192


def _expand_to_paragraph(
        text: str, start: int, end: int,
        lower: int, upper: int, *,
        window: int = PARAGRAPH_SEARCH_CHARS) -> tuple[int, int]:
    """Grow a span outward to paragraph edges, searching a bounded window.

    Without the bound this is O(n) per span, and worse: the old fallback on a
    missing delimiter was ``lower``/``upper``, the document edges, so a span
    in a document with no blank lines pulled in the whole document.

    The document edge is still the right answer when the search *reached* it.
    A span inside the opening paragraph has no delimiter to its left because
    the document starts there, and snapping to ``lower`` costs at most one
    window. Only a window that stopped short of the edge falls back to the
    span itself, which is exactly the pathological case the bound exists for.
    """
    search_floor = max(lower, start - int(window))
    left_delimiter = text.rfind("\n\n", search_floor, start)
    if left_delimiter >= 0:
        expanded_start = left_delimiter + 2
    else:
        expanded_start = lower if search_floor <= lower else start
    search_ceiling = min(upper, end + int(window))
    right_delimiter = text.find("\n\n", end, search_ceiling)
    if right_delimiter >= 0:
        expanded_end = right_delimiter
    else:
        expanded_end = upper if search_ceiling >= upper else end
    return expanded_start, expanded_end


def compile_evidence_spans(
        full_prompt: str, layout: PromptLayout,
        selected_blocks: Iterable[int], block_size: int, *,
        block_radius: int = 0,
        boundary: str = "paragraph") -> tuple[
            tuple[EvidenceSpan, ...], tuple[int, ...], tuple[int, ...]]:
    """Extract selected original text and stitch it into disjoint source spans."""
    if int(block_size) < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if boundary not in {"block", "paragraph"}:
        raise ValueError(
            f"boundary must be 'block' or 'paragraph', got {boundary!r}")
    total_blocks = math.ceil(len(layout.input_ids) / int(block_size))
    selected = normalize_block_ids(selected_blocks, total_blocks)
    expanded = expand_block_ids(
        selected, total_blocks=total_blocks, radius=block_radius)

    provisional = []
    for run in _contiguous_runs(expanded):
        token_start = run[0] * int(block_size)
        token_end = min(
            len(layout.input_ids), (run[-1] + 1) * int(block_size))
        char_range = _char_range_for_tokens(
            layout.offsets, token_start, token_end,
            layout.document_char_start, layout.document_char_end)
        if char_range is None:
            continue
        char_start, char_end = char_range
        # A block can begin inside the blank-line separator preceding the
        # actual selected paragraph. Trim only boundary whitespace before
        # finding paragraph edges so that separator tokens do not pull the
        # preceding paragraph into the packet.
        while char_start < char_end and full_prompt[char_start].isspace():
            char_start += 1
        while char_end > char_start and full_prompt[char_end - 1].isspace():
            char_end -= 1
        if char_end <= char_start:
            continue
        trimmed_start, trimmed_end = char_start, char_end
        if boundary == "paragraph":
            char_start, char_end = _expand_to_paragraph(
                full_prompt, char_start, char_end,
                layout.document_char_start, layout.document_char_end)
        # A token spans at least one character, so character growth bounds
        # token growth. That makes this window a safe over-estimate and keeps
        # the later char-to-token lookup off the full offset list.
        provisional.append({
            "char_start": char_start,
            "char_end": char_end,
            "blocks": set(run),
            "token_lo": token_start - (trimmed_start - char_start),
            "token_hi": token_end + (char_end - trimmed_end),
        })

    if not provisional:
        raise ValueError(
            "selected blocks do not overlap the source document")

    merged = []
    for item in provisional:
        if merged:
            previous = merged[-1]
            if item["char_start"] <= previous["char_end"]:
                previous["char_end"] = max(
                    previous["char_end"], item["char_end"])
                previous["blocks"].update(item["blocks"])
                previous["token_lo"] = min(
                    previous["token_lo"], item["token_lo"])
                previous["token_hi"] = max(
                    previous["token_hi"], item["token_hi"])
                continue
        merged.append(item)

    spans = []
    for item in merged:
        # Bounded by the block window the span came from, widened by however
        # far paragraph repair moved its edges. Keeps the stitch off the full
        # offset list; token_range_for_char_range re-scans everything if the
        # window ever proves too tight.
        token_start, token_end = token_range_for_char_range(
            layout.offsets, item["char_start"], item["char_end"],
            search_start=max(0, item["token_lo"] - 1),
            search_end=min(len(layout.offsets), item["token_hi"] + 1))
        text = full_prompt[item["char_start"]:item["char_end"]]
        if not text.strip():
            continue
        spans.append(EvidenceSpan(
            source_token_start=token_start,
            source_token_end=token_end,
            source_char_start=item["char_start"],
            source_char_end=item["char_end"],
            source_blocks=tuple(sorted(item["blocks"])),
            text=text,
        ))
    if not spans:
        raise ValueError("compiled evidence contains no non-whitespace text")
    return tuple(spans), selected, expanded


def render_evidence_content(
        question: str, spans: Sequence[EvidenceSpan]) -> str:
    """Render exact excerpts with compact, deterministic provenance labels."""
    if not spans:
        raise ValueError("at least one evidence span is required")
    sections = [
        "The original document was reduced to the following exact excerpts. "
        "Read them together before answering the final question."
    ]
    for index, span in enumerate(spans, start=1):
        blocks = span.source_blocks
        block_label = (
            str(blocks[0]) if len(blocks) == 1
            else f"{blocks[0]}-{blocks[-1]}"
        )
        sections.append(
            f"[Evidence {index} | source blocks {block_label} | "
            f"source tokens {span.source_token_start}-"
            f"{span.source_token_end - 1}]\n{span.text.strip()}"
        )
    sections.append(f"[Final question]\n{question.strip()}")
    return "\n\n".join(sections)


def format_evidence_chat_prompt(tokenizer, content: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer does not provide apply_chat_template")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": EVIDENCE_COMPILER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def compile_evidence_packet(
        tokenizer, full_prompt: str, user_prompt_text: str, question: str,
        selected_blocks: Iterable[int], block_size: int, *,
        block_radius: int = 0, boundary: str = "paragraph") -> EvidencePacket:
    """End-to-end exact-text compilation for a fresh dense model read."""
    layout = locate_prompt_layout(
        tokenizer, full_prompt, user_prompt_text, question)
    return compile_evidence_packet_from_layout(
        tokenizer, full_prompt, layout, question, selected_blocks,
        block_size, block_radius=block_radius, boundary=boundary)


def compile_evidence_packet_from_layout(
        tokenizer, full_prompt: str, layout: PromptLayout, question: str,
        selected_blocks: Iterable[int], block_size: int, *,
        block_radius: int = 0, boundary: str = "paragraph") -> EvidencePacket:
    """Compile a packet while reusing an already-tokenized prompt layout."""
    spans, selected, expanded = compile_evidence_spans(
        full_prompt, layout, selected_blocks, block_size,
        block_radius=block_radius, boundary=boundary)
    content = render_evidence_content(question, spans)
    prompt = format_evidence_chat_prompt(tokenizer, content)
    compiled = tokenizer(prompt)["input_ids"]
    if hasattr(compiled, "tolist"):
        compiled = compiled.tolist()
    if compiled and isinstance(compiled[0], (list, tuple)):
        compiled = compiled[0]
    return EvidencePacket(
        prompt=prompt,
        content=content,
        spans=spans,
        selected_blocks=selected,
        expanded_blocks=expanded,
        original_prompt_tokens=len(layout.input_ids),
        evidence_source_tokens=sum(
            span.source_token_end - span.source_token_start
            for span in spans),
        compiled_prompt_tokens=len(compiled),
    )
