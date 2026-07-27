import csv
import json

from benchmarks.run_route_control_suite import (
    case_rows,
    combo_stem,
    combo_summary,
    load_report,
    write_csv,
)


def _case(case_id, *, exact, contains_needle, seq_len=16375, dense=True):
    sparse = {
        "answer": "17 April 1986" if exact else "1987",
        "exact": 1.0 if exact else 0.0,
        "fuzzy": 1.0 if exact else 0.4,
        "needle_layer_group_hit_rate": 0.75,
        "needle_any_group_all_layers_rate": 0.86,
        "prefill_seconds": 1.10,
        "live_prefill_seconds": 1.24,
        "selector_seconds": 0.14,
        "peak_memory_allocated_gb": 6.4,
        "timing_samples": [{
            "candidate_contains_needle": contains_needle,
            "densified_rows": 12,
            "span_contains_needle": 1 if contains_needle else 0,
            "dense_layers": [24, 25, 26, 27],
            "dense_layer_count": 4,
            "dense_layer_fraction": 4 / 28,
            "charged_attention_fraction": 0.2,
            "charged_sparsity": 0.8,
        }],
    }
    case = {
        "case_id": case_id, "seq_len": seq_len, "needle_block": 127,
        "sparse": sparse, "dense": None, "comparison": {},
    }
    if dense:
        case["dense"] = {
            "answer": "17 April 1986", "exact": 1.0,
            "prefill_seconds": 1.62, "peak_memory_allocated_gb": 6.2,
        }
        case["comparison"] = {
            "kernel_prefill_speedup": 1.47,
            "live_prefill_speedup": 1.31,
            "live_total_speedup": 1.22,
        }
    return case


def _report(cases, aggregate=None):
    return {
        "cases": cases,
        "aggregate": aggregate if aggregate is not None else {
            "mean_sparse_fuzzy": 0.7,
            "dense_exact_rate": 1.0,
            "median_kernel_prefill_speedup": 1.47,
            "median_live_prefill_speedup": 1.31,
            "sum_kernel_prefill_speedup": 1.44,
            "sum_live_prefill_speedup": 1.29,
            "max_sparse_peak_memory_allocated_gb": 6.5,
            "max_dense_peak_memory_allocated_gb": 6.2,
        },
    }


def test_combo_stem_only_carries_m_for_candidate_modes():
    assert combo_stem("triton", "dense-candidates", 16) == (
        "triton__dense-candidates__M16")
    assert combo_stem("triton", "learned", 16) == "triton__learned"


def test_combo_stem_carries_width_and_k_when_set():
    assert combo_stem("triton", "dense-candidates", 4, 2, 10) == (
        "triton__dense-candidates__M4__W2__K10")
    # W0 is the scattered-top-M default and must not perturb the stem.
    assert combo_stem("triton", "dense-candidates", 4, 0, 10) == (
        "triton__dense-candidates__M4__K10")
    assert combo_stem("triton", "learned", 4, 2, 32) == "triton__learned__K32"
    assert combo_stem(
        "triton", "dense-candidates", 4, 0, 10, 0, [24, 25, 26, 27]
    ) == "triton__dense-candidates__M4__K10__D24-25-26-27"


def test_case_rows_carry_candidate_flag_and_dense_pairing():
    report = _report([_case("gallery", exact=True, contains_needle=True)])
    row = case_rows(report, "dense-candidates", "triton", 8)[0]
    assert row["exact"] == 1
    assert row["candidate_blocks"] == 8
    assert row["candidate_contains_needle"] is True
    assert row["dense_exact"] == 1
    assert row["kernel_prefill_speedup"] == 1.47


def test_case_rows_without_dense_omit_speedup_fields():
    case = _case("atlas", exact=False, contains_needle=True, dense=False)
    row = case_rows(_report([case]), "dense-candidates", "pytorch", 8)[0]
    assert "kernel_prefill_speedup" not in row
    assert row["exact"] == 0


def test_combo_summary_rates_over_cases():
    cases = [
        _case("gallery", exact=True, contains_needle=True),
        _case("atlas", exact=False, contains_needle=True),
        _case("observatory", exact=True, contains_needle=False),
    ]
    report = _report(cases)
    rows = case_rows(report, "dense-candidates", "triton", 8)
    summary = combo_summary(report, rows, "dense-candidates", "triton", 8)
    assert summary["cases"] == 3
    assert summary["exact_count"] == 2
    assert summary["exact_rate"] == 2 / 3
    # candidate_recall is gate retrieval, tracked separately from exactness.
    assert summary["candidate_recall"] == 2 / 3
    assert summary["median_kernel_prefill_speedup"] == 1.47


def test_span_cost_comes_from_the_child_not_from_m_times_width():
    # Overlapping +-W windows collapse, so the runner must report the measured
    # densified-row count rather than recomputing M*(2W+1) = 12 here.
    report = _report([_case("gallery", exact=True, contains_needle=True)])
    rows = case_rows(report, "dense-candidates", "triton", 4, 2, 10)
    assert rows[0]["densified_rows"] == 12
    assert rows[0]["candidate_neighborhood"] == 2
    assert rows[0]["k_selected"] == 10
    summary = combo_summary(report, rows, "dense-candidates", "triton", 4, 2, 10)
    assert summary["median_densified_rows"] == 12
    assert summary["span_recall"] == 1.0
    assert summary["dense_layers"] == "24 25 26 27"
    assert summary["dense_layer_count"] == 4
    assert summary["median_charged_attention_fraction"] == 0.2
    assert summary["median_charged_sparsity"] == 0.8


def test_combo_summary_blank_m_for_non_candidate_modes():
    report = _report([_case("gallery", exact=True, contains_needle=None)])
    rows = case_rows(report, "learned", "triton", 8)
    summary = combo_summary(report, rows, "learned", "triton", 8)
    assert summary["candidate_blocks"] == ""
    assert summary["candidate_recall"] is None


def test_load_report_unwraps_multi_k_sweeps(tmp_path):
    path = tmp_path / "report.json"
    inner = _report([_case("gallery", exact=True, contains_needle=True)])
    path.write_text(json.dumps({"sweeps": [inner]}), encoding="utf-8")
    assert load_report(str(path))["cases"][0]["case_id"] == "gallery"


def test_write_csv_fills_missing_fields(tmp_path):
    path = tmp_path / "cases.csv"
    write_csv(str(path), [{"a": 1}], ["a", "b"])
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": ""}]
