"""Cheap query-aware retrieval over a query-independent lexical block tree.

This is a live pre-Qwen path: it uses only tokenizer IDs from the source
document and question.  It never reads Qwen hidden states, Q/K features, or a
selector checkpoint.  Leaf blocks are represented by fixed-width hashed
unigram and adjacent-token features.  Parent nodes are bitwise maxima (set
union), so a rare lexical cue is not diluted as the tree becomes coarser.

Building the representation is O(nD) for fixed sketch width D.  Top-down
selection visits O(beam * log n) nodes.  Tokenization and index construction
are still linear in the input length; this module does not make a sub-linear
total-cost claim.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import torch

from interfaces.evidence_compiler import PromptLayout, document_block_ids


@dataclass(frozen=True)
class LexicalTreeLevel:
    """One leaf-first level of the deterministic lexical tree."""

    level: int
    node_id_offset: int
    features: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor

    @property
    def node_ids(self) -> torch.Tensor:
        return (
            torch.arange(self.features.shape[0], dtype=torch.int64)
            + int(self.node_id_offset)
        )


@dataclass(frozen=True)
class PreQwenIndex:
    """Query-independent document representation built before Qwen."""

    levels: tuple[LexicalTreeLevel, ...]
    document_blocks: tuple[int, ...]
    document_frequencies: torch.Tensor
    block_size: int
    feature_dim: int
    unigram_dim: int
    indexed_tokens: int

    @property
    def leaf_count(self) -> int:
        return len(self.document_blocks)

    @property
    def node_count(self) -> int:
        return sum(int(level.features.shape[0]) for level in self.levels)


@dataclass(frozen=True)
class PreQwenSelection:
    """Ranked source blocks and traversal accounting."""

    blocks: tuple[int, ...]
    scores: tuple[float, ...]
    visited_nodes: int
    levels_descended: int
    final_candidates: int


def _validate_dimensions(feature_dim: int, unigram_fraction: float) -> int:
    if int(feature_dim) < 16:
        raise ValueError("feature_dim must be >= 16")
    if not 0.1 <= float(unigram_fraction) <= 0.9:
        raise ValueError("unigram_fraction must be in [0.1, 0.9]")
    unigram_dim = int(round(int(feature_dim) * float(unigram_fraction)))
    return min(int(feature_dim) - 1, max(1, unigram_dim))


def _unigram_buckets(token_ids: torch.Tensor, width: int) -> torch.Tensor:
    values = token_ids.to(torch.int64)
    hashed = (values * 2_654_435_761) ^ (values >> 16)
    return torch.remainder(hashed, int(width))


def _bigram_buckets(
        left: torch.Tensor, right: torch.Tensor,
        offset: int, width: int) -> torch.Tensor:
    left = left.to(torch.int64)
    right = right.to(torch.int64)
    hashed = (
        left * 1_000_003
        + right * 9_176
        + (left ^ (right << 7)) * 37
    )
    return int(offset) + torch.remainder(hashed, int(width))


def _set_features(
        matrix: torch.Tensor, rows: torch.Tensor,
        buckets: torch.Tensor) -> None:
    if rows.numel() == 0:
        return
    flat = matrix.view(-1)
    indices = rows.to(torch.int64) * matrix.shape[1] + buckets.to(torch.int64)
    flat[indices] = True


def _merge_level(features: torch.Tensor, radix: int) -> torch.Tensor:
    nodes, width = features.shape
    parents = math.ceil(nodes / int(radix))
    padded = parents * int(radix)
    if padded != nodes:
        features = torch.cat([
            features,
            torch.zeros(
                padded - nodes, width, dtype=torch.bool,
                device=features.device),
        ], dim=0)
    return features.view(parents, int(radix), width).any(dim=1)


def build_pre_qwen_index(
        layout: PromptLayout, block_size: int, *,
        feature_dim: int = 4096,
        unigram_fraction: float = 0.5,
        radix: int = 2) -> PreQwenIndex:
    """Build a fixed-width lexical hierarchy over document-overlapping blocks."""
    if int(block_size) < 1:
        raise ValueError("block_size must be >= 1")
    if int(radix) < 2:
        raise ValueError("radix must be >= 2")
    unigram_dim = _validate_dimensions(feature_dim, unigram_fraction)
    bigram_dim = int(feature_dim) - unigram_dim
    blocks = document_block_ids(layout, int(block_size))
    if not blocks:
        raise ValueError("the prompt layout contains no document blocks")

    first_block = blocks[0]
    leaf_count = len(blocks)
    leaf = torch.zeros(leaf_count, int(feature_dim), dtype=torch.bool)
    token_start = int(layout.document_token_start)
    token_end = int(layout.document_token_end)
    token_ids = torch.tensor(
        layout.input_ids[token_start:token_end], dtype=torch.int64)
    absolute_positions = torch.arange(
        token_start, token_end, dtype=torch.int64)
    rows = torch.div(
        absolute_positions, int(block_size),
        rounding_mode="floor") - int(first_block)

    unigram_buckets = _unigram_buckets(token_ids, unigram_dim)
    _set_features(leaf, rows, unigram_buckets)

    if token_ids.numel() > 1:
        adjacent = rows[:-1] == rows[1:]
        bigram_rows = rows[:-1][adjacent]
        bigram_buckets = _bigram_buckets(
            token_ids[:-1][adjacent], token_ids[1:][adjacent],
            unigram_dim, bigram_dim)
        _set_features(leaf, bigram_rows, bigram_buckets)

    document_frequencies = leaf.sum(dim=0, dtype=torch.int32)
    starts = torch.arange(leaf_count, dtype=torch.int64)
    ends = starts + 1
    levels = []
    current = leaf
    node_id_offset = 0
    level_number = 0
    while True:
        levels.append(LexicalTreeLevel(
            level=level_number,
            node_id_offset=node_id_offset,
            features=current,
            starts=starts,
            ends=ends,
        ))
        node_id_offset += int(current.shape[0])
        if current.shape[0] == 1:
            break
        nodes = int(current.shape[0])
        parent_count = math.ceil(nodes / int(radix))
        parent_ids = torch.arange(parent_count, dtype=torch.int64)
        previous_starts = starts
        previous_ends = ends
        starts = previous_starts[parent_ids * int(radix)]
        ends = previous_ends[
            ((parent_ids + 1) * int(radix) - 1).clamp_max(nodes - 1)]
        current = _merge_level(current, int(radix))
        level_number += 1

    return PreQwenIndex(
        levels=tuple(levels),
        document_blocks=tuple(int(block) for block in blocks),
        document_frequencies=document_frequencies,
        block_size=int(block_size),
        feature_dim=int(feature_dim),
        unigram_dim=unigram_dim,
        indexed_tokens=int(token_ids.numel()),
    )


def _flat_token_ids(values) -> torch.Tensor:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one tokenized question")
        values = values[0]
    return torch.tensor([int(value) for value in values], dtype=torch.int64)


def question_feature_weights(
        tokenizer, question: str, index: PreQwenIndex, *,
        idf_power: float = 2.0) -> torch.Tensor:
    """Hash the question and weight its features by document rarity."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if float(idf_power) <= 0:
        raise ValueError("idf_power must be > 0")
    encoded = tokenizer(question, add_special_tokens=False)
    token_ids = _flat_token_ids(encoded["input_ids"])
    if token_ids.numel() == 0:
        raise ValueError("question tokenized to zero tokens")

    buckets = [_unigram_buckets(token_ids, index.unigram_dim)]
    if token_ids.numel() > 1:
        buckets.append(_bigram_buckets(
            token_ids[:-1], token_ids[1:],
            index.unigram_dim,
            index.feature_dim - index.unigram_dim))
    all_buckets = torch.cat(buckets)
    counts = torch.bincount(
        all_buckets, minlength=index.feature_dim).to(torch.float32)
    document_frequency = index.document_frequencies.to(torch.float32)
    idf = torch.log(
        (float(index.leaf_count) + 1.0) / (document_frequency + 1.0)
    ) + 1.0
    weights = counts.log1p() * idf.pow(float(idf_power))
    return weights


def _node_scores(
        features: torch.Tensor, query_weights: torch.Tensor) -> torch.Tensor:
    total = query_weights.sum().clamp_min(torch.finfo(torch.float32).eps)
    return (
        features.to(torch.float32).matmul(query_weights.to(torch.float32))
        / total
    )


def select_pre_qwen_blocks(
        index: PreQwenIndex, query_weights: torch.Tensor, *,
        top_m: int = 4, beam: int = 16, radix: int = 2) -> PreQwenSelection:
    """Descend the hierarchy and return ranked absolute source-block IDs."""
    if int(top_m) < 1:
        raise ValueError("top_m must be >= 1")
    if int(beam) < int(top_m):
        raise ValueError("beam must be >= top_m")
    if int(radix) < 2:
        raise ValueError("radix must be >= 2")
    if query_weights.shape != (index.feature_dim,):
        raise ValueError(
            f"query_weights must be [{index.feature_dim}], got "
            f"{tuple(query_weights.shape)}")
    if not bool(torch.isfinite(query_weights).all()):
        raise ValueError("query_weights contains non-finite values")
    if float(query_weights.sum()) <= 0:
        raise ValueError("query_weights must contain positive mass")

    current = torch.tensor([0], dtype=torch.int64)
    visited = 1
    levels_descended = 0
    for level_index in range(len(index.levels) - 2, -1, -1):
        child_count = int(index.levels[level_index].features.shape[0])
        children = []
        for parent in current.tolist():
            start = int(parent) * int(radix)
            children.extend(
                range(start, min(start + int(radix), child_count)))
        candidates = torch.tensor(
            sorted(set(children)), dtype=torch.int64)
        scores = _node_scores(
            index.levels[level_index].features.index_select(0, candidates),
            query_weights)
        visited += int(candidates.numel())
        keep = min(int(beam), int(candidates.numel()))
        # Stable source-order tie breaking: sort IDs first, then stable score.
        order = torch.argsort(scores, descending=True, stable=True)[:keep]
        current = candidates.index_select(0, order)
        levels_descended += 1

    leaf_scores = _node_scores(
        index.levels[0].features.index_select(0, current),
        query_weights)
    width = min(int(top_m), int(current.numel()))
    order = torch.argsort(
        leaf_scores, descending=True, stable=True)[:width]
    chosen_positions = current.index_select(0, order)
    chosen_scores = leaf_scores.index_select(0, order)
    blocks = tuple(
        index.document_blocks[int(position)]
        for position in chosen_positions.tolist())
    return PreQwenSelection(
        blocks=blocks,
        scores=tuple(float(score) for score in chosen_scores.tolist()),
        visited_nodes=visited,
        levels_descended=levels_descended,
        final_candidates=int(current.numel()),
    )


def retrieve_pre_qwen_blocks(
        tokenizer, layout: PromptLayout, question: str, block_size: int, *,
        top_m: int = 4,
        beam: int = 16,
        feature_dim: int = 4096,
        unigram_fraction: float = 0.5,
        idf_power: float = 2.0,
        radix: int = 2) -> tuple[PreQwenIndex, PreQwenSelection]:
    """Build the live index, encode the question, and retrieve source blocks."""
    index = build_pre_qwen_index(
        layout, block_size,
        feature_dim=feature_dim,
        unigram_fraction=unigram_fraction,
        radix=radix)
    weights = question_feature_weights(
        tokenizer, question, index, idf_power=idf_power)
    selection = select_pre_qwen_blocks(
        index, weights, top_m=top_m, beam=beam, radix=radix)
    return index, selection
