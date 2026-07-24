"""Run a resumable held-out 64K-128K YaRN extraction and benchmark suite.

Each requested length is extracted and benchmarked independently. A completed
per-length JSON report is the resume marker, so interrupted Colab sessions can
continue without repeating successful lengths.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_LENGTHS = (
    64_000, 72_000, 80_000, 88_000, 96_000,
    104_000, 112_000, 120_000, 128_000,
)
DEFAULT_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
HELDOUT_CASES = 6


def _run(command):
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _files_in(directory):
    return sorted(glob.glob(os.path.join(directory, "*.pt")))


def _safe_remove_target_bucket(targets_dir, bucket):
    targets_root = os.path.realpath(targets_dir)
    bucket_root = os.path.realpath(bucket)
    if bucket_root == targets_root:
        raise RuntimeError("refusing to remove the complete targets directory")
    if os.path.commonpath((targets_root, bucket_root)) != targets_root:
        raise RuntimeError(f"target bucket escaped work directory: {bucket}")
    if os.path.isdir(bucket_root):
        shutil.rmtree(bucket_root)
        print(
            f"removed regenerable feature targets -> {bucket_root}",
            flush=True,
        )


def combine_reports(paths, output_path, plot_path, suite):
    """Combine per-length reports into one accuracy/efficiency artifact."""
    reports = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            reports.append(json.load(handle))
    if not reports:
        raise ValueError("at least one per-length report is required")

    first = reports[0]
    cases = [
        case
        for report in reports
        for case in report["cases"]
    ]
    from benchmarks.compare_dense_sparse_live_tree import aggregate_results
    from benchmarks.plot_efficiency_accuracy import (
        save_efficiency_accuracy_plot,
    )

    combined = {
        key: value
        for key, value in first.items()
        if key not in {"cases", "aggregate"}
    }
    combined.update({
        "suite": suite,
        "length_reports": [os.path.abspath(path) for path in paths],
        "cases": cases,
        "aggregate": aggregate_results(cases),
    })
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)
    save_efficiency_accuracy_plot(combined, plot_path)
    print(f"combined report -> {output_path}", flush=True)
    print(f"combined plot   -> {plot_path}", flush=True)
    return combined


def _extract_length(args, length, target_dir):
    command = [
        sys.executable,
        os.path.join("scripts", "extract_teacher_targets.py"),
        "--model", args.model,
        "--lengths", str(length),
        "--yarn-factor", "4.0",
        "--features-only",
        "--heldout",
        "--all",
        "--depths",
    ]
    command.extend(str(depth) for depth in args.depths)
    command.extend([
        "--depth-mode", "grid",
        "--store-dtype", "float16",
        "--no-heatmap",
        "--skip-existing",
        "--partition-by-length",
        "--out", os.path.dirname(target_dir),
    ])
    _run(command)


def _benchmark_length(args, length, target_dir, report, plot):
    _run([
        sys.executable,
        os.path.join("benchmarks", "compare_dense_sparse_live_tree.py"),
        "--gate", args.gate,
        "--targets", os.path.join(target_dir, "*.pt"),
        "--model", args.model,
        "--yarn-factor", "4.0",
        "--beam", str(args.beam),
        "--k-selected", str(args.k_selected),
        "--local-window", "1",
        "--selector-dtype", "float16",
        "--selector-layer-chunk", str(args.selector_layer_chunk),
        "--dtype", "auto",
        "--max-new-tokens", str(args.max_new_tokens),
        "--warmup-tokens", "64",
        "--repeats", str(args.repeats),
        "--out", report,
        "--plot", plot,
        "--load-offload-dir", "/tmp/spruce_hf_load_offload",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--rerun-completed", action="store_true",
        help="replace per-length reports that already exist")
    parser.add_argument(
        "--cleanup-targets", action="store_true",
        help=(
            "delete each length's regenerable feature artifacts only after "
            "its benchmark report is saved"
        ),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not args.lengths or any(length < 1 for length in args.lengths):
        raise SystemExit("--lengths must contain positive integers")
    if not args.depths or any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0, 1]")

    os.makedirs(args.work_dir, exist_ok=True)
    targets_dir = os.path.join(args.work_dir, "targets")
    outputs_dir = os.path.join(args.work_dir, "outputs")
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    expected_per_length = len(args.depths) * HELDOUT_CASES
    report_paths = []
    for length in args.lengths:
        target_dir = os.path.join(targets_dir, str(length))
        report = os.path.join(outputs_dir, f"length_{length}.json")
        plot = os.path.join(outputs_dir, f"length_{length}.png")
        report_paths.append(report)

        report_complete = os.path.isfile(report) and not args.rerun_completed
        if report_complete:
            print(f"resume: keeping completed report -> {report}", flush=True)
        elif not args.skip_benchmark:
            target_count = len(_files_in(target_dir))
            if target_count != expected_per_length and not args.skip_extraction:
                _extract_length(args, length, target_dir)
                target_count = len(_files_in(target_dir))
            if target_count != expected_per_length:
                raise SystemExit(
                    f"expected {expected_per_length} targets for {length}, "
                    f"found {target_count}; rerun without --skip-extraction"
                )
            _benchmark_length(args, length, target_dir, report, plot)

        if args.cleanup_targets and os.path.isfile(report):
            _safe_remove_target_bucket(targets_dir, target_dir)

    missing_reports = [
        path for path in report_paths if not os.path.isfile(path)
    ]
    if missing_reports:
        raise SystemExit(
            f"missing {len(missing_reports)} per-length reports; rerun "
            "without --skip-benchmark"
        )

    stem = (
        f"full_yarn4_K{args.k_selected}_"
        f"{min(args.lengths) // 1000}k_{max(args.lengths) // 1000}k"
    )
    combined_report = os.path.join(outputs_dir, f"{stem}.json")
    combined_plot = os.path.join(outputs_dir, f"{stem}.png")
    combine_reports(
        report_paths,
        combined_report,
        combined_plot,
        suite={
            "lengths": args.lengths,
            "depths": args.depths,
            "heldout_cases": HELDOUT_CASES,
            "paired_trials": (
                len(args.lengths) * len(args.depths) * HELDOUT_CASES
            ),
            "repeats": args.repeats,
            "resumable_by_length": True,
        },
    )


if __name__ == "__main__":
    main()
