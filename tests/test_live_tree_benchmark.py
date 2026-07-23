import math

import torch

from benchmarks.compare_dense_sparse_live_tree import (
    aggregate_results,
    live_route,
    resolve_k_sweep,
    summarize_timings,
)
from selector.gate import FlatGate


def _selector_target(path, *, layers=2, groups=2, blocks=5, proto=2, dim=4):
    torch.save({
        "pooledQ": torch.randn(1, layers, groups, blocks, proto, dim).half(),
        "pooledK": torch.randn(1, layers, groups, blocks, proto, dim).half(),
        "seq_len": blocks * 64,
        "block_size": 64,
        "needle_block": 1,
        "proto": proto,
    }, path)


def test_live_route_builds_valid_selected_blocks_and_reports_components(tmp_path):
    target = tmp_path / "target.pt"
    _selector_target(target)
    gate = FlatGate(2, 4, proj_dim=4).eval()
    selected, timing = live_route(
        gate, {"num_layers": 2, "head_dim": 4}, target,
        beam=2, radix=2, k_selected=4, local_window=1,
        selector_device="cpu", validate=True, selector_layer_chunk=2,
    )

    assert selected.shape == (1, 2, 2, 5, 4)
    assert selected.dtype == torch.int32
    component_sum = sum(timing[key] for key in (
        "tree_build_seconds", "tree_traversal_seconds",
        "route_pack_seconds", "route_transfer_seconds"))
    assert math.isclose(timing["selector_seconds"], component_sum)


def test_timing_summary_and_aggregate_use_medians_and_live_tree_cost():
    samples = [{"x": 3.0}, {"x": 1.0}, {"x": 2.0}]
    assert summarize_timings(samples, ("x",)) == {"x": 2.0}

    case = {
        "answers_match": True,
        "sparse": {"exact": True, "fuzzy": 1.0, "prefill_seconds": 2.0,
                   "live_prefill_seconds": 2.5},
        "dense": {"exact": True, "fuzzy": 1.0, "prefill_seconds": 3.0},
        "comparison": {"kernel_prefill_speedup": 1.5,
                       "live_prefill_speedup": 1.2, "live_total_speedup": 1.1},
    }
    aggregate = aggregate_results([case])
    assert aggregate["median_kernel_prefill_speedup"] == 1.5
    assert aggregate["median_live_prefill_speedup"] == 1.2
    assert aggregate["sum_live_prefill_speedup"] == 1.2


def test_k_sweep_reduces_effective_beam_to_fit_each_width():
    assert resolve_k_sweep(
        beam=16, local_window=1, k_values=[10, 12, 18]) == [
        (10, 8), (12, 10), (18, 16),
    ]
