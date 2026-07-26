"""Route-control parity suite: every diagnostic route mode x backend x target.

Reruns the 2026-07-26 laptop diagnostics (learned / oracle-needle /
dense-reader / oracle-dense-evidence / dense-candidates) through
``compare_dense_sparse_live_tree.py`` and collects one summary CSV. On a
Colab GPU run it with ``--backends triton pytorch`` to close the
reference/kernel parity gap for these findings; on the 8GB laptop use
``--backends pytorch`` (dense is skipped either way — dense answers come
from the targets' dense-screen records).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODES = (
    "learned", "oracle-needle", "dense-reader", "oracle-dense-evidence",
    "dense-candidates",
)


def run_one(args, target, mode, backend, out_json):
    command = [
        sys.executable,
        os.path.join(ROOT, "benchmarks", "compare_dense_sparse_live_tree.py"),
        "--gate", args.gate,
        "--targets", target,
        "--backend", backend,
        "--route-mode", mode,
        "--skip-dense",
        "--k-selected", str(args.k_selected),
        "--beam", str(args.beam),
        "--repeats", "1",
        "--max-new-tokens", str(args.max_new_tokens),
        "--candidate-blocks", str(args.candidate_blocks),
        "--out", out_json,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-5:])
        err = "\n".join(completed.stderr.splitlines()[-5:])
        return {"error": f"exit {completed.returncode}: {tail} {err}"}
    with open(out_json, encoding="utf-8") as handle:
        report = json.load(handle)
    case = report["cases"][0]
    sparse = case["sparse"]
    return {
        "answer": sparse["answer"],
        "exact": int(bool(sparse["exact"])),
        "fuzzy": sparse["fuzzy"],
        "needle_hit": sparse.get("needle_layer_group_hit_rate"),
        "any_group": sparse.get("needle_any_group_all_layers_rate"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=("triton", "pytorch"),
        default=["pytorch"])
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--candidate-blocks", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for target in args.targets:
        stem = os.path.splitext(os.path.basename(target))[0]
        for backend in args.backends:
            for mode in args.modes:
                out_json = os.path.join(
                    args.out_dir, f"{stem}__{backend}__{mode}.json")
                result = run_one(args, target, mode, backend, out_json)
                row = {"target": stem, "mode": mode, "backend": backend,
                       **result}
                rows.append(row)
                print(
                    f"{stem} {backend} {mode}: "
                    + (result.get("error")
                       or f"answer {result['answer']!r} exact {result['exact']}"),
                    flush=True)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    fields = ["target", "mode", "backend", "answer", "exact", "fuzzy",
              "needle_hit", "any_group", "error"]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
