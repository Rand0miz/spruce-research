from argparse import Namespace

import pytest

from benchmarks.ruler_paired_feature_e2e import (
    aggregate_block,
    cached_selections,
    deduplicate_prompts,
    score_generated_pair,
    validate_route_config,
)


def test_deduplicate_prompts_reuses_identical_packet():
    unique, mapping = deduplicate_prompts(["same", "same"])
    assert unique == ["same"]
    assert mapping == [0, 0]

    unique, mapping = deduplicate_prompts(["first", "second"])
    assert unique == ["first", "second"]
    assert mapping == [0, 1]


def test_factorial_route_cache_is_config_and_reference_checked():
    args = Namespace(
        model="model", beam=16, block_size=64, block_radius=1,
        boundary="paragraph", unigram_fraction=0.5, idf_power=2.0,
        radix=2, query_tail_lines=3, candidate_blocks=9)
    report = {
        "config": {
            "model": "model", "beam": 16, "block_size": 64,
            "block_radius": 1, "boundary": "paragraph",
            "unigram_fraction": 0.5, "idf_power": 2.0, "radix": 2,
            "tails": [2, 3], "feature_dims": [1024, 4096],
            "budgets": [4, 9],
        },
        "cases": [{
            "references": ["needle"],
            "arms": {
                "t3_d1024_m9": {"selected_blocks": [1, 2, 3]},
                "t3_d4096_m9": {"selected_blocks": [4, 5, 6]},
            },
        }],
    }
    validate_route_config(report, args)
    assert cached_selections(report, 0, ["needle"]) == {
        1024: [1, 2, 3], 4096: [4, 5, 6]}
    with pytest.raises(ValueError, match="reference mismatch"):
        cached_selections(report, 0, ["different"])


def _case(score_1024, score_4096, *, identical=False):
    return {
        "task": "niah_single_1",
        "length": 4096,
        "references": ["needle"],
        "identical_packet": identical,
        "score_delta_d4096_minus_d1024": score_4096 - score_1024,
        "paired_batch": {"batch_seconds": 2.0},
        "arms": {
            "d1024": {
                "score": score_1024, "perfect": score_1024 == 1.0,
                "evidence_score": 0.5, "input_tokens": 100,
                "compression_fraction": 0.1,
            },
            "d4096": {
                "score": score_4096, "perfect": score_4096 == 1.0,
                "evidence_score": 1.0, "input_tokens": 120,
                "compression_fraction": 0.12,
            },
        },
    }


def test_score_pair_and_aggregate_keep_pairing_explicit():
    case = {
        "references": ["needle"],
        "arms": {"d1024": {}, "d4096": {}},
    }
    scored = score_generated_pair(
        case, {"d1024": "unused", "d4096": "unused"},
        ["no answer", "the needle"], [0, 1], "niah_single_1")
    assert scored["arms"]["d1024"]["score"] == 0.0
    assert scored["arms"]["d4096"]["score"] == 1.0
    assert scored["score_delta_d4096_minus_d1024"] == 1.0

    block = aggregate_block([
        _case(0.0, 1.0),
        _case(1.0, 1.0, identical=True),
        _case(1.0, 0.0),
    ])
    assert block["samples"] == 3
    assert block["score_d1024"] == pytest.approx(2 / 3)
    assert block["score_d4096"] == pytest.approx(2 / 3)
    assert block["delta_d4096_minus_d1024"] == pytest.approx(0.0)
    assert (block["d4096_wins"], block["ties"], block["d1024_wins"]) == (1, 1, 1)
    assert block["identical_packet_fraction"] == pytest.approx(1 / 3)
