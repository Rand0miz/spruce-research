"""Focused tests for the paper-baseline selectors and their two runners."""
import pytest
import torch

from interfaces.evidence_compiler import locate_prompt_layout
from selector.baselines import (
    ALL_METHODS,
    POSITIONAL_METHODS,
    bm25_block_scores,
    bm25_select,
    flat_select,
    positional_select,
    selection_agreement,
)
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)
from tests.test_pre_qwen_selector import WhitespaceTokenizer


EVIDENCE = (
    "The geology committee designated Lake Orison as the primary "
    "groundwater source."
)


def _fixture(filler_paragraphs=6):
    """Build a layout whose evidence sits in a known middle paragraph."""
    tokenizer = WhitespaceTokenizer()
    paragraphs = [
        f"Routine harbor schedule {index} lists cargo inspections, "
        f"ordinary repairs, and dock staffing for the season."
        for index in range(filler_paragraphs)
    ]
    paragraphs.insert(filler_paragraphs // 2, EVIDENCE)
    document = "\n\n".join(paragraphs)
    question = (
        "\n\nWhich lake did the geology committee designate as the primary "
        "groundwater source?"
    )
    user = document + question
    full = f"system wrapper user {user} assistant"
    layout = locate_prompt_layout(tokenizer, full, user, question)
    return tokenizer, full, question, layout


def _index_and_weights(tokenizer, layout, question, block_size=8):
    index = build_pre_qwen_index(
        layout, block_size, feature_dim=1024, unigram_fraction=0.5, radix=2)
    weights = question_feature_weights(
        tokenizer, question, index, idf_power=2.0)
    return index, weights


def _evidence_blocks(tokenizer, prompt, block_size):
    """Absolute block IDs covering the evidence sentence."""
    encoded = tokenizer(prompt, return_offsets_mapping=True)
    start = prompt.index(EVIDENCE)
    end = start + len(EVIDENCE)
    blocks = {
        position // block_size
        for position, (left, right) in enumerate(encoded["offset_mapping"])
        if left >= start and right <= end
    }
    assert blocks, "fixture did not locate the evidence tokens"
    return blocks


def test_flat_and_tree_agree_on_a_small_document():
    tokenizer, _, question, layout = _fixture()
    index, weights = _index_and_weights(tokenizer, layout, question)
    tree = select_pre_qwen_blocks(index, weights, top_m=2, beam=16, radix=2)
    flat = flat_select(index, weights, top_m=2)
    agreement = selection_agreement(tree.blocks, flat.blocks)
    # The beam is wider than the document here, so the descent cannot prune a
    # block the flat scan would have kept. Divergence would mean the scoring
    # functions have drifted apart, which is exactly what the runner measures.
    assert agreement["identical_set"], agreement
    assert flat.scored_blocks == len(index.document_blocks)


def test_flat_scores_every_leaf_not_a_beam():
    tokenizer, _, question, layout = _fixture(filler_paragraphs=12)
    index, weights = _index_and_weights(tokenizer, layout, question)
    flat = flat_select(index, weights, top_m=4)
    assert flat.method == "flat"
    assert flat.scored_blocks == len(index.document_blocks)
    assert len(flat.blocks) == 4
    assert len(set(flat.blocks)) == 4


def test_flat_rejects_a_mismatched_query_width():
    tokenizer, _, question, layout = _fixture()
    index, _ = _index_and_weights(tokenizer, layout, question)
    with pytest.raises(ValueError, match="query_weights must be"):
        flat_select(index, torch.ones(7), top_m=2)


def test_flat_rejects_empty_query_mass():
    tokenizer, _, question, layout = _fixture()
    index, _ = _index_and_weights(tokenizer, layout, question)
    with pytest.raises(ValueError, match="positive mass"):
        flat_select(index, torch.zeros(index.feature_dim), top_m=2)


def test_bm25_ranks_the_evidence_block_first():
    block_size = 8
    tokenizer, prompt, question, layout = _fixture()
    selection = bm25_select(tokenizer, layout, question, block_size, top_m=2)
    assert selection.method == "bm25"
    expected = _evidence_blocks(tokenizer, prompt, block_size)
    assert selection.blocks[0] in expected, (
        f"BM25 ranked {selection.blocks} first, evidence lives in {expected}")


def test_bm25_scores_are_non_negative_and_one_per_block():
    block_size = 8
    tokenizer, _, question, layout = _fixture()
    question_ids = torch.tensor(
        tokenizer(question, add_special_tokens=False)["input_ids"],
        dtype=torch.int64)
    scores, blocks = bm25_block_scores(layout, block_size, question_ids)
    assert scores.shape == (len(blocks),)
    assert bool((scores >= 0).all())
    assert float(scores.max()) > 0.0


def test_bm25_rejects_invalid_parameters():
    tokenizer, _, question, layout = _fixture()
    with pytest.raises(ValueError, match="k1 must be"):
        bm25_select(tokenizer, layout, question, 8, top_m=1, k1=0.0)
    with pytest.raises(ValueError, match="b must be"):
        bm25_select(tokenizer, layout, question, 8, top_m=1, b=1.5)


@pytest.mark.parametrize("method", POSITIONAL_METHODS)
def test_positional_arms_spend_the_same_budget(method):
    _, _, _, layout = _fixture(filler_paragraphs=12)
    selection = positional_select(
        layout, 8, method=method, top_m=4, seed=20260728)
    assert selection.method == method
    assert len(selection.blocks) == 4
    assert len(set(selection.blocks)) == 4
    assert list(selection.blocks) == sorted(selection.blocks)
    assert selection.scored_blocks == 0


def test_lead_and_tail_sit_at_opposite_ends():
    _, _, _, layout = _fixture(filler_paragraphs=12)
    lead = positional_select(layout, 8, method="lead", top_m=3)
    tail = positional_select(layout, 8, method="tail", top_m=3)
    assert max(lead.blocks) < min(tail.blocks)


def test_stride_is_not_a_lead_variant():
    _, _, _, layout = _fixture(filler_paragraphs=16)
    stride = positional_select(layout, 8, method="stride", top_m=4)
    lead = positional_select(layout, 8, method="lead", top_m=4)
    assert stride.blocks != lead.blocks
    assert max(stride.blocks) - min(stride.blocks) > 1


def test_random_is_seed_deterministic():
    _, _, _, layout = _fixture(filler_paragraphs=16)
    first = positional_select(layout, 8, method="random", top_m=4, seed=11)
    same = positional_select(layout, 8, method="random", top_m=4, seed=11)
    other = positional_select(layout, 8, method="random", top_m=4, seed=12)
    assert first.blocks == same.blocks
    assert first.blocks != other.blocks


def test_positional_rejects_an_unknown_method():
    _, _, _, layout = _fixture()
    with pytest.raises(ValueError, match="method must be one of"):
        positional_select(layout, 8, method="bm25", top_m=2)


def test_selection_agreement_reports_set_and_order():
    identical = selection_agreement((3, 7), (3, 7))
    assert identical["identical_set"] and identical["identical_order"]
    assert identical["jaccard"] == 1.0 and identical["top1_match"]

    reordered = selection_agreement((3, 7), (7, 3))
    assert reordered["identical_set"]
    assert not reordered["identical_order"]
    assert not reordered["top1_match"]

    disjoint = selection_agreement((1, 2), (3, 4))
    assert not disjoint["identical_set"]
    assert disjoint["jaccard"] == 0.0
    assert disjoint["left_only"] == [1, 2]
    assert disjoint["right_only"] == [3, 4]


def test_all_methods_cover_every_family():
    assert set(ALL_METHODS) == {
        "tree", "flat", "bm25", "lead", "tail", "stride", "random"}


def _cpu_row(candidate_id, length, methods, needle_hit=True):
    return {
        "candidate_id": candidate_id,
        "requested_length": length,
        "arms": {
            method: {
                "selected_contains_needle": needle_hit,
                "expanded_contains_needle": needle_hit,
                "packet_tokens": 1500,
                "compression_fraction": 0.08,
                "select_seconds": 0.002,
                "scored_blocks": 40,
            }
            for method in methods
        },
        "tree_vs_flat": selection_agreement((1, 2), (1, 2)),
    }


def test_cpu_aggregate_groups_by_method_and_length():
    from benchmarks.selector_baselines_cpu import aggregate_rows

    rows = [
        _cpu_row("a", 16384, ["tree", "flat"], needle_hit=True),
        _cpu_row("b", 32768, ["tree", "flat"], needle_hit=False),
    ]
    aggregate = aggregate_rows(rows, ["tree", "flat"])
    assert aggregate["prompts"] == 2
    assert aggregate["by_method"]["tree"]["expanded_needle_recall"] == 0.5
    assert aggregate["by_length"]["16384"]["tree"][
        "expanded_needle_recall"] == 1.0
    assert aggregate["by_length"]["32768"]["tree"][
        "expanded_needle_recall"] == 0.0
    assert aggregate["tree_vs_flat"]["identical_set_rate"] == 1.0
    assert aggregate["tree_vs_flat"]["disagreeing_prompts"] == []


def _l4_row(phase, yarn_factor, arm, candidate_id, exact, length=16384):
    return {
        "row_id": f"{phase}|y{yarn_factor:g}|{arm}|{candidate_id}",
        "phase": phase,
        "yarn_factor": yarn_factor,
        "arm": arm,
        "candidate_id": candidate_id,
        "requested_length": length,
        "exact": exact,
        "request_seconds": 1.0,
        "input_tokens": 1500,
        "peak_memory_allocated_gb": 3.2,
        "peak_memory_reserved_gb": 3.6,
    }


def test_l4_aggregate_pairs_each_arm_against_its_own_dense():
    from benchmarks.paper_baselines_l4 import aggregate_rows

    rows = [
        _l4_row("yarn", 4.0, "dense", "a", False),
        _l4_row("yarn", 4.0, "tree", "a", True),
        _l4_row("yarn", 1.0, "dense", "a", True),
        _l4_row("yarn", 1.0, "tree", "a", True),
    ]
    aggregate = aggregate_rows(rows)
    assert aggregate["by_arm"]["yarn|yarn4|dense"]["exact_rate"] == 0.0
    assert aggregate["by_arm"]["yarn|yarn1|dense"]["exact_rate"] == 1.0
    # The factor-4 pairing must not borrow the factor-1 dense result.
    assert aggregate["paired_against_dense"]["yarn|yarn4|tree"][
        "arm_only"] == 1
    assert aggregate["paired_against_dense"]["yarn|yarn1|tree"][
        "both_correct"] == 1


def test_l4_aggregate_reports_recall_only_when_present():
    from benchmarks.paper_baselines_l4 import aggregate_rows

    dense = _l4_row("baselines", 4.0, "dense", "a", True)
    tree = _l4_row("baselines", 4.0, "tree", "a", True)
    tree["expanded_contains_needle"] = True
    tree["selected_contains_needle"] = False
    aggregate = aggregate_rows([dense, tree])
    assert "expanded_needle_recall" not in aggregate["by_arm"][
        "baselines|yarn4|dense"]
    assert aggregate["by_arm"]["baselines|yarn4|tree"][
        "expanded_needle_recall"] == 1.0
    assert aggregate["by_arm"]["baselines|yarn4|tree"][
        "direct_needle_recall"] == 0.0


class _PlanArgs:
    """Minimal stand-in for the runner's parsed arguments."""

    def __init__(self, **overrides):
        self.phases = ["baselines", "yarn"]
        self.methods = ["tree", "flat"]
        self.include_dense = False
        self.lengths = [16384, 131072]
        self.yarn_lengths = [16384, 32768, 131072]
        self.yarn_factor = 4.0
        self.original_max_position_embeddings = 32768
        for name, value in overrides.items():
            setattr(self, name, value)


def test_l4_plan_loads_factor_four_before_factor_one():
    from benchmarks.paper_baselines_l4 import _plan

    groups = _plan(_PlanArgs())
    assert [group["phase"] for group in groups] == [
        "baselines", "yarn", "yarn"]
    assert [group["yarn_factor"] for group in groups] == [4.0, 4.0, 1.0]
    # 131072 exceeds the native context, so factor 1 cannot legally run there.
    assert groups[1]["lengths"] == [16384, 32768]
    assert groups[2]["lengths"] == [16384, 32768]


def test_l4_plan_rejects_yarn_lengths_beyond_the_native_context():
    from benchmarks.paper_baselines_l4 import _plan

    with pytest.raises(SystemExit, match="native context"):
        _plan(_PlanArgs(phases=["yarn"], yarn_lengths=[131072]))
