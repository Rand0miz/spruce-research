"""Run the held-out diverse-prose retrieval suite with the current gate.

This is a distribution-shift test for the repeated-filler needle harness. It
uses non-repeating essay-style background sentences, near-miss facts, concise
answer scoring, and the same fixed-K live tree traversal as the main benchmark.
"""
import argparse
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.run_long_context_suite import (
    _files_in,
    _safe_remove_target_bucket,
    combine_reports,
)


DEFAULT_LENGTHS = (64_000, 80_000, 96_000, 112_000, 128_000)
DEFAULT_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_BANK = os.path.join(
    ROOT, "scripts", "prompt_banks", "natural_heldout.json")


def _run(command):
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _case_count(prompt_bank):
    with open(prompt_bank, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data["cases"] if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("natural prompt bank must contain cases")
    return len(cases)


def _extract_length(args, length, target_dir):
    command = [
        sys.executable,
        os.path.join("scripts", "extract_teacher_targets.py"),
        "--model", args.model,
        "--prompt-bank", args.prompt_bank,
        "--lengths", str(length),
        "--yarn-factor", "4.0",
        "--features-only",
        "--all",
        "--depths",
    ]
    command.extend(str(depth) for depth in args.depths)
    command.extend([
        "--depth-mode", "grid",
        "--seed", str(args.seed),
        "--store-dtype", "float16",
        "--no-heatmap",
        "--skip-existing",
        "--partition-by-length",
        "--out", os.path.dirname(target_dir),
    ])
    _run(command)


def _benchmark_length(args, target_dir, report, plot):
    _run([
        sys.executable,
        os.path.join("benchmarks", "compare_dense_sparse_live_tree.py"),
        "--gate", args.gate,
        "--targets", os.path.join(target_dir, "*.pt"),
        "--model", args.model,
        "--prompt-bank", args.prompt_bank,
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
    parser.add_argument("--prompt-bank", default=DEFAULT_BANK)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--cleanup-targets", action="store_true",
        help="delete a length's feature artifacts after its report is saved")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not os.path.isfile(args.prompt_bank):
        raise SystemExit(f"prompt bank not found: {args.prompt_bank}")
    case_count = _case_count(args.prompt_bank)
    expected_per_length = case_count * len(args.depths)

    targets_dir = os.path.join(args.work_dir, "targets")
    outputs_dir = os.path.join(args.work_dir, "outputs")
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    report_paths = []
    for length in args.lengths:
        target_dir = os.path.join(targets_dir, str(length))
        report = os.path.join(outputs_dir, f"natural_length_{length}.json")
        plot = os.path.join(outputs_dir, f"natural_length_{length}.png")
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
                    f"expected {expected_per_length} natural targets for "
                    f"{length}, found {target_count}")
            _benchmark_length(args, target_dir, report, plot)

        if args.cleanup_targets and os.path.isfile(report):
            _safe_remove_target_bucket(targets_dir, target_dir)

    missing = [path for path in report_paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} per-length natural reports")

    stem = (
        f"natural_yarn4_K{args.k_selected}_"
        f"{min(args.lengths) // 1000}k_{max(args.lengths) // 1000}k"
    )
    combine_reports(
        report_paths,
        os.path.join(outputs_dir, f"{stem}.json"),
        os.path.join(outputs_dir, f"{stem}.png"),
        suite={
            "test_type": "heldout_diverse_prose_retrieval",
            "prompt_bank": os.path.abspath(args.prompt_bank),
            "generator": "natural_prose_v1",
            "seed": args.seed,
            "lengths": args.lengths,
            "depths": args.depths,
            "heldout_cases": case_count,
            "paired_trials": (
                len(args.lengths) * len(args.depths) * case_count),
            "repeats": args.repeats,
            "resumable_by_length": True,
            "selector_gate_unchanged": True,
        },
    )


if __name__ == "__main__":
    main()
