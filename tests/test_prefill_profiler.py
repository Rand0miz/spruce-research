from types import SimpleNamespace

from benchmarks.profile_sparse_prefill import summarize_events
from kernels.sparse_prefill import KERNEL_VARIANTS, kernel_runtime_metadata


def test_profiler_summary_sorts_by_self_device_time():
    events = [
        SimpleNamespace(
            key="small", count=2,
            self_device_time_total=3.0, device_time_total=4.0,
            self_cpu_time_total=5.0, cpu_time_total=6.0),
        SimpleNamespace(
            key="large", count=1,
            self_device_time_total=9.0, device_time_total=10.0,
            self_cpu_time_total=2.0, cpu_time_total=3.0),
    ]
    summary = summarize_events(events, row_limit=1)
    assert summary["total_self_device_time_us"] == 12.0
    assert summary["top_operators"][0]["name"] == "large"


def test_kernel_variants_are_isolated_and_reportable():
    assert {
        "single_head",
        "single_head_causal",
        "single_head_prescale",
        "single_head_qtile",
    }.issubset(KERNEL_VARIANTS)
    assert kernel_runtime_metadata("single_head")["variant"] == "single_head"
