"""Run the resumable 16K-128K unscreened natural YaRN paper suite."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEFAULT_LENGTHS = tuple(16_384 * multiplier for multiplier in range(1, 9))
DEFAULT_DEPTHS = (0.1, 0.5, 0.9)
DEFAULT_BANK = os.path.join(
    ROOT, "scripts", "prompt_banks", "natural_paper_untouched.json")


def _run(command):
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _completed(path):
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    expected = report.get("suite", {}).get("paired_cases_expected")
    return (
        report.get("status") == "completed"
        and report.get("prompt_build_seconds") is not None
        and expected is not None
        and len(report.get("cases", [])) == int(expected)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--prompt-bank", default=DEFAULT_BANK)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"),
        default="paragraph")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"),
        default="float16")
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=32_768)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="only combine already-completed per-length reports")
    args = parser.parse_args()

    if not os.path.isfile(args.prompt_bank):
        raise SystemExit(f"prompt bank not found: {args.prompt_bank}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.beam < args.candidate_blocks:
        raise SystemExit("--beam must be >= --candidate-blocks")
    if not args.lengths or len(args.lengths) != len(set(args.lengths)):
        raise SystemExit("--lengths must be non-empty and duplicate-free")
    if args.lengths != sorted(args.lengths):
        raise SystemExit("--lengths must be strictly increasing")
    if not args.depths or any(
            not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0,1]")

    work = Path(args.work_dir)
    lengths_dir = work / "length_reports"
    paper_dir = work / "paper_artifacts"
    lengths_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for index, length in enumerate(args.lengths, start=1):
        report = lengths_dir / f"length_{length}.json"
        reports.append(report)
        if _completed(report):
            print(
                f"[{index}/{len(args.lengths)}] resume complete: {length}",
                flush=True)
        else:
            if args.skip_benchmark:
                raise SystemExit(f"incomplete length report: {report}")
            command = [
                sys.executable, "-u",
                os.path.join(
                    "benchmarks",
                    "benchmark_pre_qwen_natural_yarn_length.py"),
                "--prompt-bank", os.path.abspath(args.prompt_bank),
                "--length", str(length),
                "--depths", *[str(depth) for depth in args.depths],
                "--seed", str(args.seed),
                "--model", args.model,
                "--block-size", str(args.block_size),
                "--candidate-blocks", str(args.candidate_blocks),
                "--block-radius", str(args.block_radius),
                "--boundary", args.boundary,
                "--beam", str(args.beam),
                "--feature-dim", str(args.feature_dim),
                "--unigram-fraction", str(args.unigram_fraction),
                "--idf-power", str(args.idf_power),
                "--radix", str(args.radix),
                "--repeats", str(args.repeats),
                "--max-new-tokens", str(args.max_new_tokens),
                "--dtype", args.dtype,
                "--yarn-factor", str(args.yarn_factor),
                "--original-max-position-embeddings",
                str(args.original_max_position_embeddings),
                "--out", str(report),
                "--load-offload-dir", str(
                    Path(tempfile.gettempdir())
                    / "spruce_natural_yarn_offload" / str(length)),
            ]
            _run(command)
        if not _completed(report):
            raise SystemExit(f"length did not complete: {report}")
        # Refresh partial tables and figures after every completed length so a
        # Colab interruption still leaves immediately inspectable artifacts.
        _run([
            sys.executable, "-u",
            os.path.join(
                "benchmarks", "report_pre_qwen_natural_yarn.py"),
            "--reports", *[str(path) for path in reports],
            "--out-dir", str(work / "paper_artifacts_partial"),
            "--bootstrap-iterations", str(
                min(1000, args.bootstrap_iterations)),
        ])

    _run([
        sys.executable, "-u",
        os.path.join(
            "benchmarks", "report_pre_qwen_natural_yarn.py"),
        "--reports", *[str(path) for path in reports],
        "--out-dir", str(paper_dir),
        "--bootstrap-iterations", str(args.bootstrap_iterations),
    ])
    print(f"combined report -> {paper_dir / 'combined_report.json'}")
    print(f"summary         -> {paper_dir / 'SUMMARY.md'}")
    print(f"figures         -> {paper_dir / 'figures'}")
    print(f"tables          -> {paper_dir / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
