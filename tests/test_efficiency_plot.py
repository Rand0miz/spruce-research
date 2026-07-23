import pytest

from benchmarks.plot_efficiency_accuracy import (
    build_scaling_series,
    save_efficiency_accuracy_plot,
)


def _case(seq_len, dense, sparse, live, exact=True):
    return {
        "seq_len": seq_len,
        "answers_match": True,
        "dense": {
            "prefill_seconds": dense,
            "exact": exact,
            "fuzzy": 1.0,
        },
        "sparse": {
            "prefill_seconds": sparse,
            "live_prefill_seconds": live,
            "exact": exact,
            "fuzzy": 1.0,
            "tree_build_seconds": 0.1,
            "tree_traversal_seconds": 0.2,
            "route_pack_seconds": 0.01,
        },
    }


def test_scaling_series_groups_repeated_lengths():
    report = {"cases": [
        _case(1024, 2.0, 1.0, 1.5),
        _case(1024, 4.0, 2.0, 3.0),
        _case(2048, 6.0, 2.0, 3.0),
    ]}
    series = build_scaling_series(report)
    assert [row["seq_len"] for row in series] == [1024, 2048]
    assert series[0]["targets"] == 2
    assert series[0]["kernel_prefill_speedup"] == 2.0
    assert series[1]["live_prefill_speedup"] == 2.0


def test_plot_writes_png_and_csv(tmp_path):
    pytest.importorskip("matplotlib")
    report = {
        "model": "test-model",
        "selector": {"beam": 16, "k_selected": 18},
        "cases": [_case(1024, 2.0, 1.0, 1.5)],
    }
    png = tmp_path / "scaling.png"
    plot_path, csv_path = save_efficiency_accuracy_plot(report, png)
    assert plot_path == png
    assert png.stat().st_size > 0
    assert (tmp_path / "scaling.csv").stat().st_size > 0
