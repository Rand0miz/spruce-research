"""Route-control parity suite: route mode x backend x candidate budget x target.

Drives ``compare_dense_sparse_live_tree.py`` over a whole target set and
collects per-case and per-combination summaries.

Two jobs this runner is built for:

1. **Held-out validation + kernel parity.** ``--modes learned dense-candidates
   --backends triton pytorch`` answers whether retrieve-then-re-encode is a
   rate or an anecdote, and whether the Triton kernel reproduces the PyTorch
   reference conclusions.
2. **Cost measurement with the densified rows included.** ``--include-dense``
   keeps the paired dense run so speedups are measured on the routes actually
   used, instead of being inherited from runs whose selections lacked dense
   rows. ``--candidate-block-values 4 8 16 32`` sweeps M for the
   accuracy-versus-cost curve.

One child process handles every target for a combination, so the model is
loaded once per combination rather than once per target. ``--resume`` skips
combinations whose JSON already exists, which matters on Colab.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sparse.config import (
    add_residual_summary_arguments,
    residual_summary_config_from_args,
)

DEFAULT_MODES = (
    "learned", "oracle-needle", "dense-reader", "oracle-dense-evidence",
    "dense-candidates",
)
# Only dense-candidates reads --candidate-blocks; every other mode would run
# identical work once per M value.
CANDIDATE_MODES = ("dense-candidates",)


def combo_stem(backend, mode, candidate_blocks, neighborhood=0,
               k_selected=None, sink_blocks=0, dense_layers=(),
               summary_prototypes=None):
    parts = [backend, mode]
    if mode in CANDIDATE_MODES:
        parts.append(f"M{candidate_blocks}")
        if neighborhood:
            parts.append(f"W{neighborhood}")
    if k_selected is not None:
        parts.append(f"K{k_selected}")
    if sink_blocks:
        parts.append(f"S{sink_blocks}")
    if dense_layers:
        parts.append("D" + "-".join(str(layer) for layer in dense_layers))
    if summary_prototypes is not None:
        parts.append(f"RS-P{summary_prototypes}")
    return "__".join(parts)


def run_combo(args, targets, mode, backend, candidate_blocks, neighborhood,
              k_selected, sink_blocks, dense_layers, out_json, log_path):
    command = [
        sys.executable,
        os.path.join(ROOT, "benchmarks", "compare_dense_sparse_live_tree.py"),
        "--gate", args.gate,
        "--targets", *targets,
        "--backend", backend,
        "--route-mode", mode,
        "--k-selected", str(k_selected),
        "--beam", str(args.beam),
        "--repeats", str(args.repeats),
        "--max-new-tokens", str(args.max_new_tokens),
        "--candidate-blocks", str(candidate_blocks),
        "--candidate-neighborhood", str(neighborhood),
        "--sink-blocks", str(sink_blocks),
        "--out", out_json,
    ]
    if not args.include_dense:
        command.append("--skip-dense")
    if args.prompt_bank:
        command.extend(["--prompt-bank", args.prompt_bank])
    if args.model:
        command.extend(["--model", args.model])
    if dense_layers:
        command.extend(["--dense-layers", *map(str, dense_layers)])
    if args.summary_config.enabled:
        command.extend([
            "--residual-summaries",
            "--summary-prototypes", str(args.summary_config.prototypes),
            "--summary-mode", args.summary_config.mode,
        ])
        if args.summary_config.checkpoint:
            command.extend([
                "--summary-checkpoint", args.summary_config.checkpoint])
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    return completed.returncode


def load_report(out_json):
    with open(out_json, encoding="utf-8") as handle:
        report = json.load(handle)
    # Single-K runs write the report directly; multi-K runs nest one report per
    # K under "sweeps". Route-control combinations are always single-K.
    if "cases" not in report and report.get("sweeps"):
        report = report["sweeps"][0]
    return report


def case_rows(report, mode, backend, candidate_blocks, neighborhood=0,
              k_selected=None, sink_blocks=0):
    rows = []
    for case in report["cases"]:
        sparse = case["sparse"]
        dense = case.get("dense")
        sample = (sparse.get("timing_samples") or [{}])[0]
        row = {
            "case_id": case["case_id"],
            "seq_len": case["seq_len"],
            "needle_block": case["needle_block"],
            "mode": mode,
            "backend": backend,
            "k_selected": k_selected,
            "sink_blocks": sink_blocks,
            "candidate_blocks": (
                candidate_blocks if mode in CANDIDATE_MODES else ""),
            "candidate_neighborhood": (
                neighborhood if mode in CANDIDATE_MODES else ""),
            # Measured by the child (overlapping windows collapse), so this is
            # the honest row cost of the route, not M*(2w+1).
            "densified_rows": sample.get("densified_rows"),
            "dense_layers": " ".join(
                str(layer) for layer in sample.get("dense_layers", [])),
            "dense_layer_count": sample.get("dense_layer_count"),
            "dense_layer_fraction": sample.get("dense_layer_fraction"),
            "charged_attention_fraction": sample.get(
                "charged_attention_fraction"),
            "charged_sparsity": sample.get("charged_sparsity"),
            "span_contains_needle": sample.get("span_contains_needle"),
            "answer": sparse["answer"],
            "exact": int(bool(sparse["exact"])),
            "fuzzy": sparse["fuzzy"],
            "needle_hit": sparse.get("needle_layer_group_hit_rate"),
            "any_group": sparse.get("needle_any_group_all_layers_rate"),
            "candidate_contains_needle": sample.get("candidate_contains_needle"),
            "sparse_prefill_seconds": sparse.get("prefill_seconds"),
            "sparse_live_prefill_seconds": sparse.get("live_prefill_seconds"),
            "selector_seconds": sparse.get("selector_seconds"),
            "sparse_peak_gb": sparse.get("peak_memory_allocated_gb"),
        }
        if dense is not None:
            comparison = case.get("comparison") or {}
            row.update({
                "dense_answer": dense["answer"],
                "dense_exact": int(bool(dense["exact"])),
                "dense_prefill_seconds": dense.get("prefill_seconds"),
                "dense_peak_gb": dense.get("peak_memory_allocated_gb"),
                "kernel_prefill_speedup": comparison.get(
                    "kernel_prefill_speedup"),
                "live_prefill_speedup": comparison.get("live_prefill_speedup"),
                "live_total_speedup": comparison.get("live_total_speedup"),
            })
        rows.append(row)
    return rows


def _median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def combo_summary(report, rows, mode, backend, candidate_blocks,
                  neighborhood=0, k_selected=None, sink_blocks=0):
    aggregate = report.get("aggregate", {})
    exact = [row["exact"] for row in rows]
    contains = [
        row["candidate_contains_needle"] for row in rows
        if row["candidate_contains_needle"] is not None]
    span_hits = [
        row["span_contains_needle"] for row in rows
        if row.get("span_contains_needle") is not None]
    return {
        "mode": mode,
        "backend": backend,
        "k_selected": k_selected,
        "sink_blocks": sink_blocks,
        "candidate_blocks": candidate_blocks if mode in CANDIDATE_MODES else "",
        "candidate_neighborhood": (
            neighborhood if mode in CANDIDATE_MODES else ""),
        "median_densified_rows": _median(
            [row.get("densified_rows") for row in rows]),
        "dense_layers": rows[0].get("dense_layers", "") if rows else "",
        "dense_layer_count": (
            rows[0].get("dense_layer_count") if rows else None),
        "median_charged_attention_fraction": _median(
            [row.get("charged_attention_fraction") for row in rows]),
        "median_charged_sparsity": _median(
            [row.get("charged_sparsity") for row in rows]),
        "span_recall": (
            sum(span_hits) / len(span_hits) if span_hits else None),
        "cases": len(rows),
        "exact_count": sum(exact),
        "exact_rate": (sum(exact) / len(exact)) if exact else None,
        "mean_fuzzy": aggregate.get("mean_sparse_fuzzy"),
        "candidate_recall": (
            sum(contains) / len(contains) if contains else None),
        "dense_exact_rate": aggregate.get("dense_exact_rate"),
        "median_kernel_prefill_speedup": aggregate.get(
            "median_kernel_prefill_speedup"),
        "median_live_prefill_speedup": aggregate.get(
            "median_live_prefill_speedup"),
        "sum_kernel_prefill_speedup": aggregate.get(
            "sum_kernel_prefill_speedup"),
        "sum_live_prefill_speedup": aggregate.get("sum_live_prefill_speedup"),
        "max_sparse_peak_gb": aggregate.get(
            "max_sparse_peak_memory_allocated_gb"),
        "max_dense_peak_gb": aggregate.get(
            "max_dense_peak_memory_allocated_gb"),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="teacher target paths or globs; globs are expanded by the child")
    parser.add_argument("--model")
    parser.add_argument("--prompt-bank")
    parser.add_argument(
        "--backends", nargs="+", choices=("triton", "pytorch"),
        default=["pytorch"])
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument(
        "--candidate-block-values", type=int, nargs="+", default=[8],
        help="M sweep for dense-candidates: densified reader-row candidates")
    parser.add_argument(
        "--candidate-neighborhood-values", type=int, nargs="+", default=[0],
        help="W sweep for dense-candidates: densify +-W rows around each "
             "candidate. 0 is scattered top-M; W>0 tests contiguous repair")
    parser.add_argument(
        "--sink-block-values", type=int, nargs="+", default=[0],
        help="attention-sink sweep: force the first N key blocks into every "
             "causal row outside the K budget. 0 is the current policy")
    parser.add_argument(
        "--k-selected-values", type=int, nargs="+", default=None,
        help="route-budget sweep; overrides --k-selected when given")
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument(
        "--dense-layers", type=int, nargs="*", default=[],
        help="zero-based decoder layers dispatched to dense SDPA in every "
             "combination; charged density is copied into CSV summaries")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--include-dense", action="store_true",
        help="keep the paired dense run so speedups are measured on the same "
             "routes (required for any efficiency claim about dense rows)")
    add_residual_summary_arguments(parser)
    parser.add_argument(
        "--resume", action="store_true",
        help="skip combinations whose report JSON already exists")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        args.summary_config = residual_summary_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.summary_config.enabled and any(
            backend != "pytorch" for backend in args.backends):
        parser.error("--residual-summaries requires --backends pytorch")

    if any(value < 1 for value in args.candidate_block_values):
        raise SystemExit("--candidate-block-values must be >= 1")
    if any(value < 0 for value in args.candidate_neighborhood_values):
        raise SystemExit("--candidate-neighborhood-values must be >= 0")
    if any(value < 0 for value in args.sink_block_values):
        raise SystemExit("--sink-block-values must be >= 0")
    k_values = args.k_selected_values or [args.k_selected]
    if any(value < 1 for value in k_values):
        raise SystemExit("--k-selected-values must be >= 1")
    args.dense_layers = sorted(set(args.dense_layers))
    if any(value < 0 for value in args.dense_layers):
        raise SystemExit("--dense-layers must be non-negative")

    os.makedirs(args.out_dir, exist_ok=True)
    combos = []
    for backend in args.backends:
        for mode in args.modes:
            # Non-candidate modes ignore M and W, so collapse them to one run
            # instead of repeating identical work per swept value.
            block_values = (
                args.candidate_block_values if mode in CANDIDATE_MODES else [0])
            width_values = (
                args.candidate_neighborhood_values
                if mode in CANDIDATE_MODES else [0])
            for k_selected in k_values:
                for sink_blocks in args.sink_block_values:
                    for candidate_blocks in block_values:
                        for neighborhood in width_values:
                            combos.append((backend, mode, candidate_blocks,
                                           neighborhood, k_selected,
                                           sink_blocks))

    case_rows_all, summaries, failures = [], [], []
    for index, combo in enumerate(combos, start=1):
        (backend, mode, candidate_blocks, neighborhood, k_selected,
         sink_blocks) = combo
        stem = combo_stem(
            backend, mode, candidate_blocks, neighborhood,
            k_selected, sink_blocks, args.dense_layers,
            summary_prototypes=(
                args.summary_config.prototypes
                if args.summary_config.enabled else None
            ))
        out_json = os.path.join(args.out_dir, f"{stem}.json")
        log_path = os.path.join(args.out_dir, f"{stem}.log")
        if args.resume and os.path.isfile(out_json):
            print(f"[{index}/{len(combos)}] {stem}: resume, reusing report",
                  flush=True)
        else:
            print(f"[{index}/{len(combos)}] {stem}: running", flush=True)
            code = run_combo(args, args.targets, mode, backend,
                             candidate_blocks, neighborhood, k_selected,
                             sink_blocks, args.dense_layers,
                             out_json, log_path)
            if code != 0:
                print(f"[{index}/{len(combos)}] {stem}: FAILED exit {code} "
                      f"(see {log_path})", flush=True)
                failures.append({"combo": stem, "exit_code": code,
                                 "log": log_path})
                continue
        try:
            report = load_report(out_json)
            rows = case_rows(report, mode, backend, candidate_blocks,
                             neighborhood, k_selected, sink_blocks)
        except (OSError, ValueError, KeyError) as error:
            print(f"[{index}/{len(combos)}] {stem}: unreadable report: {error}",
                  flush=True)
            failures.append({"combo": stem, "exit_code": "unreadable",
                             "log": log_path})
            continue
        case_rows_all.extend(rows)
        summary = combo_summary(report, rows, mode, backend, candidate_blocks,
                                neighborhood, k_selected, sink_blocks)
        summaries.append(summary)
        print(f"[{index}/{len(combos)}] {stem}: exact "
              f"{summary['exact_count']}/{summary['cases']} "
              f"rows={summary['median_densified_rows']} "
              f"candidate_recall={summary['candidate_recall']} "
              f"median_kernel_speedup="
              f"{summary['median_kernel_prefill_speedup']}",
              flush=True)

    case_fields = [
        "case_id", "seq_len", "needle_block", "mode", "backend", "k_selected",
        "sink_blocks",
        "candidate_blocks", "candidate_neighborhood", "densified_rows",
        "dense_layers", "dense_layer_count", "dense_layer_fraction",
        "charged_attention_fraction", "charged_sparsity",
        "span_contains_needle",
        "answer", "exact", "fuzzy", "needle_hit",
        "any_group", "candidate_contains_needle", "sparse_prefill_seconds",
        "sparse_live_prefill_seconds", "selector_seconds", "sparse_peak_gb",
        "dense_answer", "dense_exact", "dense_prefill_seconds", "dense_peak_gb",
        "kernel_prefill_speedup", "live_prefill_speedup", "live_total_speedup",
    ]
    summary_fields = [
        "mode", "backend", "k_selected", "sink_blocks", "candidate_blocks",
        "candidate_neighborhood", "median_densified_rows", "dense_layers",
        "dense_layer_count", "median_charged_attention_fraction",
        "median_charged_sparsity", "span_recall",
        "cases", "exact_count",
        "exact_rate", "mean_fuzzy", "candidate_recall", "dense_exact_rate",
        "median_kernel_prefill_speedup", "median_live_prefill_speedup",
        "sum_kernel_prefill_speedup", "sum_live_prefill_speedup",
        "max_sparse_peak_gb", "max_dense_peak_gb",
    ]
    cases_path = os.path.join(args.out_dir, "cases.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    write_csv(cases_path, case_rows_all, case_fields)
    write_csv(summary_path, summaries, summary_fields)
    print(f"cases -> {cases_path}")
    print(f"summary -> {summary_path}")
    if failures:
        print("FAILED combinations: "
              + ", ".join(failure["combo"] for failure in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
