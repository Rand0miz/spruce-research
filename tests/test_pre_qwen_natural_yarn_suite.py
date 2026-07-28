import json
from pathlib import Path

import pytest

from benchmarks.report_pre_qwen_natural_yarn import (
    _exact_binomial_two_sided,
    build_summary,
    make_figures,
    write_markdown_summary,
    write_tables,
)
from benchmarks.run_pre_qwen_natural_yarn_suite import (
    DEFAULT_LENGTHS,
    _completed,
)
from scripts.extract_teacher_targets import load_prompt_bank


ROOT = Path(__file__).resolve().parents[1]
PAPER_BANK = (
    ROOT / "scripts" / "prompt_banks" / "natural_paper_untouched.json")


def _mode(exact, request, tokens, compiled=False):
    result = {
        "exact": bool(exact),
        "answer": "answer",
        "request_seconds": float(request),
        "prefill_seconds": float(request) * 0.7,
        "decode_seconds": float(request) * 0.2,
        "input_transfer_seconds": float(request) * 0.01,
        "input_tokens": int(tokens),
        "peak_memory_allocated_gb": 3.0 if compiled else 6.0,
        "peak_memory_reserved_gb": 4.0 if compiled else 7.0,
    }
    if compiled:
        result.update({
            "compression_fraction": tokens / 16_384,
            "selected_contains_needle": bool(exact),
            "expanded_contains_needle": True,
            "layout_tokenize_seconds": 0.02,
            "index_seconds": 0.01,
            "selection_seconds": 0.002,
            "compile_seconds": 0.01,
            "compact_tokenize_seconds": 0.003,
            "visited_nodes": 100,
        })
    return result


def _row(case_id, length, depth, dense_exact, compiled_exact):
    outcome = (
        "both_correct"
        if dense_exact and compiled_exact
        else "dense_only"
        if dense_exact
        else "compiled_only"
        if compiled_exact
        else "neither_correct"
    )
    return {
        "candidate_id": f"{case_id}_{length}_{depth}",
        "source_case_id": case_id,
        "genre": "test record",
        "requested_length": length,
        "seq_len": length - 32,
        "depth": depth,
        "needle_block": 4,
        "prompt_sha256": "A" * 64,
        "paired_outcome": outcome,
        "dense": _mode(dense_exact, 4.0, length),
        "compiled": _mode(
            compiled_exact, 0.5, 900 + int(depth * 100),
            compiled=True),
    }


def _synthetic_cases():
    cases = []
    for length in (16_384, 32_768):
        cases.extend([
            _row("alpha", length, 0.1, True, True),
            _row("alpha", length, 0.5, False, True),
            _row("beta", length, 0.1, True, False),
            _row("beta", length, 0.5, False, False),
        ])
    return cases


def test_paper_bank_is_sealed_diverse_and_well_formed():
    payload = json.loads(PAPER_BANK.read_text(encoding="utf-8"))
    cases = load_prompt_bank(PAPER_BANK)
    assert payload["sealed_utc"]
    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
    assert len({case["genre"] for case in cases}) == 12
    for case in cases:
        assert case["needle"] in case["evidence"]
        assert len(case["distractors"]) == 3
        assert case["answers"]


def test_default_lengths_are_exact_16k_steps_through_128k():
    assert DEFAULT_LENGTHS == tuple(
        16_384 * multiplier for multiplier in range(1, 9))
    assert DEFAULT_LENGTHS[-1] == 131_072


def test_length_resume_marker_requires_finalized_complete_report(tmp_path):
    path = tmp_path / "length.json"
    report = {
        "status": "completed",
        "suite": {"paired_cases_expected": 2},
        "cases": [{}, {}],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not _completed(path)
    report["prompt_build_seconds"] = {"sum": 1.0, "median": 0.5}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _completed(path)
    report["cases"].pop()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not _completed(path)


def test_paired_summary_preserves_wins_losses_and_cluster_bootstrap():
    summary = build_summary(_synthetic_cases(), bootstrap_iterations=100)
    overall = summary["overall"]
    assert overall["dense"]["exact_count"] == 4
    assert overall["compiled"]["exact_count"] == 4
    assert overall["paired"]["compiled_only"] == 2
    assert overall["paired"]["dense_only"] == 2
    assert overall["paired"]["both_correct"] == 2
    assert overall["paired"]["neither_correct"] == 2
    assert overall["sum_weighted_speedup"] == pytest.approx(8.0)
    assert overall["expanded_evidence_recall"] == 1.0
    assert (
        overall["cluster_bootstrap_accuracy_delta"]["clusters"] == 2)
    assert set(summary["by_length"]) == {"16384", "32768"}
    assert set(summary["by_depth"]) == {"0.1", "0.5"}


def test_exact_mcnemar_control():
    assert _exact_binomial_two_sided(0, 0) == 1.0
    assert _exact_binomial_two_sided(1, 0) == 1.0
    assert _exact_binomial_two_sided(10, 0) == pytest.approx(
        2 / (2 ** 10))


def test_tables_markdown_and_all_paper_figures_render(tmp_path):
    cases = _synthetic_cases()
    summary = build_summary(cases, bootstrap_iterations=20)
    tables = write_tables(cases, summary, tmp_path / "tables")
    figures = make_figures(summary, tmp_path / "figures")
    markdown = write_markdown_summary(summary, tmp_path / "SUMMARY.md")

    assert len(tables) == 5
    assert len(figures) == 12
    for path in tables.values():
        assert Path(path).is_file()
    for png, pdf in figures.values():
        assert Path(png).is_file() and Path(png).stat().st_size > 1000
        assert Path(pdf).is_file() and Path(pdf).stat().st_size > 1000
    text = Path(markdown).read_text(encoding="utf-8")
    assert "Dense exact" in text
    assert "Compiler exact" in text
    assert "McNemar" in text
