import subprocess
import sys

import torch

from benchmarks.evaluate_evidence_compiler import (
    _mode_aggregate,
    flat_reader_candidates,
    tree_reader_candidates,
)
from selector.gate import FlatGate


def _features(blocks=8):
    torch.manual_seed(7)
    return {
        "q_feat": torch.randn(2, 1, blocks, 2, 4),
        "k_feat": torch.randn(2, 1, blocks, 2, 4),
        "meta": {
            "num_layers": 2,
            "head_dim": 4,
            "kb": blocks,
        },
    }


def test_flat_reader_candidates_never_returns_wrapper_blocks():
    features = _features()
    gate = FlatGate(2, 4, proj_dim=4).eval()
    blocks, scores, timing = flat_reader_candidates(
        gate, features, allowed_blocks=[2, 3, 4, 5], top_m=3)
    assert len(blocks) == len(scores) == 3
    assert set(blocks) <= {2, 3, 4, 5}
    assert timing["selection_seconds"] >= 0
    assert timing["selection_complexity"].startswith("flat_")


def test_tree_reader_candidates_are_bounded_document_leaves():
    features = _features(blocks=9)
    gate = FlatGate(2, 4, proj_dim=4).eval()
    blocks, scores, timing = tree_reader_candidates(
        gate, features, allowed_blocks=range(1, 8), top_m=3,
        beam=3, radix=2, layer_chunk=1)
    assert 1 <= len(blocks) <= 3
    assert len(blocks) == len(scores)
    assert set(blocks) <= set(range(1, 8))
    assert timing["tree_candidate_union"] >= len(blocks)
    assert timing["selection_complexity"].startswith("recursive_")


def test_mode_aggregate_reports_length_splits_and_packet_size():
    cases = [
        {
            "requested_length": 16384,
            "modes": {
                "tree": {
                    "exact": True,
                    "selected_contains_needle": True,
                    "expanded_contains_needle": True,
                    "prefill_seconds": 0.1,
                    "packet": {
                        "compiled_prompt_tokens": 600,
                        "compression_fraction": 0.04,
                    },
                },
            },
        },
        {
            "requested_length": 32768,
            "modes": {
                "tree": {
                    "exact": False,
                    "selected_contains_needle": False,
                    "expanded_contains_needle": True,
                    "prefill_seconds": 0.2,
                    "packet": {
                        "compiled_prompt_tokens": 800,
                        "compression_fraction": 0.03,
                    },
                },
            },
        },
    ]
    aggregate = _mode_aggregate(cases, "tree")
    assert aggregate["exact_count"] == 1
    assert aggregate["exact_rate"] == 0.5
    assert aggregate["by_requested_length"]["16384"]["exact_count"] == 1
    assert aggregate["by_requested_length"]["32768"]["exact_count"] == 0
    assert aggregate["median_compiled_prompt_tokens"] == 700
    assert aggregate["needle_block_recall"] == 0.5
    assert aggregate["expanded_needle_recall"] == 1.0


def test_evidence_compiler_cli_help_parses_without_cuda():
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/evaluate_evidence_compiler.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--candidate-blocks" in completed.stdout
    assert "--boundary" in completed.stdout
