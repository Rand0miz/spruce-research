"""Alternative block selectors used as paper baselines.

Every selector here answers the same question the recursive tree answers:
given a prompt layout and a question, which document blocks should the
evidence compiler stitch?  They all return absolute source-block IDs in the
same form as ``selector.pre_qwen.PreQwenSelection.blocks``, so the compiler,
the paragraph repair, and the packet format are identical across arms and the
only thing that varies is the selection rule.

Three families:

``flat``
    Scores every leaf block with the exact query weights the tree uses, then
    takes the top ``M``.  This is the tree's own scoring function without the
    top-down descent, so a tie against the tree means the tree's contribution
    is cost, not accuracy.

``bm25``
    Ordinary Okapi BM25 over the same block grid, treating tokenizer IDs as
    terms.  Query-independent statistics (document frequency, block lengths)
    come from the same document range the lexical index uses.

positional
    ``lead`` / ``tail`` / ``stride`` / ``random`` ignore the question
    entirely.  They exist to answer "would any compact packet do?" at a
    matched block budget.
"""
from dataclasses import dataclass
import math

import torch

from interfaces.evidence_compiler import PromptLayout, document_block_ids
# Reuse the tree's own scoring function so the flat arm cannot silently drift
# away from the tree it is being compared against.
from selector.pre_qwen import PreQwenIndex, _node_scores


POSITIONAL_METHODS = ("lead", "tail", "stride", "random")
QUERY_METHODS = ("tree", "flat", "bm25")
ALL_METHODS = QUERY_METHODS + POSITIONAL_METHODS


@dataclass(frozen=True)
class BaselineSelection:
    """Ranked source blocks from a non-tree selector."""

    blocks: tuple[int, ...]
    scores: tuple[float, ...]
    method: str
    scored_blocks: int

    @property
    def visited_nodes(self) -> int:
        """Nodes touched, for cost comparison against the tree traversal."""
        return int(self.scored_blocks)


def _top_m_positions(scores: torch.Tensor, top_m: int) -> torch.Tensor:
    """Stable descending top-M over leaf positions, tie-broken by block order."""
    width = min(int(top_m), int(scores.numel()))
    return torch.argsort(scores, descending=True, stable=True)[:width]


def _as_blocks(
        document_blocks: tuple[int, ...],
        positions: torch.Tensor,
        scores: torch.Tensor,
        method: str,
        scored_blocks: int) -> BaselineSelection:
    chosen = positions.tolist()
    return BaselineSelection(
        blocks=tuple(int(document_blocks[int(p)]) for p in chosen),
        scores=tuple(float(scores[int(p)]) for p in chosen),
        method=method,
        scored_blocks=int(scored_blocks),
    )


def flat_select(
        index: PreQwenIndex, query_weights: torch.Tensor, *,
        top_m: int = 4) -> BaselineSelection:
    """Score every leaf block with the tree's scoring rule and take top-M."""
    if int(top_m) < 1:
        raise ValueError("top_m must be >= 1")
    if query_weights.shape != (index.feature_dim,):
        raise ValueError(
            f"query_weights must be [{index.feature_dim}], got "
            f"{tuple(query_weights.shape)}")
    if float(query_weights.sum()) <= 0:
        raise ValueError("query_weights must contain positive mass")

    leaves = index.levels[0].features
    scores = _node_scores(leaves, query_weights)
    positions = _top_m_positions(scores, top_m)
    return _as_blocks(
        index.document_blocks, positions, scores, "flat",
        int(leaves.shape[0]))


def _document_rows(
        layout: PromptLayout, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Return document token IDs, their block rows, and the block IDs."""
    blocks = document_block_ids(layout, int(block_size))
    if not blocks:
        raise ValueError("the prompt layout contains no document blocks")
    start = int(layout.document_token_start)
    end = int(layout.document_token_end)
    token_ids = torch.tensor(layout.input_ids[start:end], dtype=torch.int64)
    positions = torch.arange(start, end, dtype=torch.int64)
    rows = torch.div(
        positions, int(block_size), rounding_mode="floor") - int(blocks[0])
    return token_ids, rows, tuple(int(block) for block in blocks)


def _flat_question_ids(tokenizer, question: str) -> torch.Tensor:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    encoded = tokenizer(question, add_special_tokens=False)
    values = encoded["input_ids"]
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one tokenized question")
        values = values[0]
    ids = torch.tensor([int(value) for value in values], dtype=torch.int64)
    if ids.numel() == 0:
        raise ValueError("question tokenized to zero tokens")
    return ids


def bm25_block_scores(
        layout: PromptLayout, block_size: int, question_ids: torch.Tensor, *,
        k1: float = 1.5,
        b: float = 0.75) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Okapi BM25 score per document block, terms being tokenizer IDs."""
    if float(k1) <= 0:
        raise ValueError("k1 must be > 0")
    if not 0.0 <= float(b) <= 1.0:
        raise ValueError("b must be in [0, 1]")
    token_ids, rows, blocks = _document_rows(layout, block_size)
    block_count = len(blocks)
    lengths = torch.bincount(rows, minlength=block_count).to(torch.float32)
    average_length = float(lengths.mean().clamp_min(1.0))
    denominator_length = float(k1) * (
        1.0 - float(b) + float(b) * lengths / average_length)

    scores = torch.zeros(block_count, dtype=torch.float32)
    for term in torch.unique(question_ids).tolist():
        matches = token_ids == int(term)
        if not bool(matches.any()):
            continue
        frequency = torch.bincount(
            rows[matches], minlength=block_count).to(torch.float32)
        document_frequency = float((frequency > 0).sum())
        idf = math.log(
            1.0
            + (block_count - document_frequency + 0.5)
            / (document_frequency + 0.5))
        scores += idf * (
            frequency * (float(k1) + 1.0)
            / (frequency + denominator_length))
    return scores, blocks


def bm25_select(
        tokenizer, layout: PromptLayout, question: str, block_size: int, *,
        top_m: int = 4, k1: float = 1.5,
        b: float = 0.75) -> BaselineSelection:
    """Rank the same block grid with BM25 and take top-M."""
    if int(top_m) < 1:
        raise ValueError("top_m must be >= 1")
    question_ids = _flat_question_ids(tokenizer, question)
    scores, blocks = bm25_block_scores(
        layout, block_size, question_ids, k1=k1, b=b)
    positions = _top_m_positions(scores, top_m)
    return _as_blocks(blocks, positions, scores, "bm25", len(blocks))


def positional_select(
        layout: PromptLayout, block_size: int, *, method: str,
        top_m: int = 4, seed: int = 0) -> BaselineSelection:
    """Choose blocks without reading the question, at the same budget."""
    if int(top_m) < 1:
        raise ValueError("top_m must be >= 1")
    if method not in POSITIONAL_METHODS:
        raise ValueError(
            f"method must be one of {POSITIONAL_METHODS}, got {method!r}")
    _, _, blocks = _document_rows(layout, block_size)
    count = len(blocks)
    width = min(int(top_m), count)

    if method == "lead":
        positions = list(range(width))
    elif method == "tail":
        positions = list(range(count - width, count))
    elif method == "stride":
        # Evenly spaced over the document, first sample at the half step so
        # the arm is not accidentally a lead or tail variant.
        step = count / float(width)
        positions = sorted({
            min(count - 1, int(math.floor((index + 0.5) * step)))
            for index in range(width)
        })
    else:
        generator = torch.Generator().manual_seed(int(seed))
        positions = torch.randperm(
            count, generator=generator)[:width].sort().values.tolist()

    chosen = torch.tensor(positions, dtype=torch.int64)
    scores = torch.zeros(count, dtype=torch.float32)
    scores[chosen] = 1.0
    return _as_blocks(blocks, chosen, scores, method, 0)


def selection_agreement(
        left: tuple[int, ...], right: tuple[int, ...]) -> dict:
    """Compare two block selections as sets and as ranked lists."""
    left_set = set(int(block) for block in left)
    right_set = set(int(block) for block in right)
    union = left_set | right_set
    intersection = left_set & right_set
    return {
        "identical_set": left_set == right_set,
        "identical_order": tuple(int(b) for b in left) == tuple(
            int(b) for b in right),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "overlap_count": len(intersection),
        "left_only": sorted(left_set - right_set),
        "right_only": sorted(right_set - left_set),
        "top1_match": (
            bool(left) and bool(right) and int(left[0]) == int(right[0])),
    }
