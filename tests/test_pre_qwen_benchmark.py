import pytest

from benchmarks.benchmark_pre_qwen_e2e import aggregate_cases


def _case(length, dense_seconds, compiled_seconds, exact=True):
    return {
        "requested_length": length,
        "dense": {
            "exact": True,
            "request_seconds": dense_seconds,
            "prefill_seconds": dense_seconds * 0.8,
        },
        "compiled": {
            "exact": exact,
            "request_seconds": compiled_seconds,
            "prefill_seconds": compiled_seconds * 0.5,
            "input_tokens": 900,
            "compression_fraction": 0.05,
            "selected_contains_needle": exact,
            "expanded_contains_needle": True,
        },
    }


def test_aggregate_uses_sum_weighted_request_time_not_median_speedups():
    cases = [
        _case(16384, 10.0, 2.0),
        _case(32768, 30.0, 3.0, exact=False),
    ]
    result = aggregate_cases(cases)
    assert result["sum_weighted_speedup"] == pytest.approx(8.0)
    assert result["dense"]["sum_request_seconds"] == 40.0
    assert result["compiled"]["sum_request_seconds"] == 5.0
    assert result["compiled"]["exact_count"] == 1
    assert result["compiled"]["expanded_needle_recall"] == 1.0
    assert result["by_requested_length"]["32768"]["compiled"] == {
        "cases": 1,
        "exact_count": 0,
        "exact_rate": 0.0,
    }
