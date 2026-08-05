"""Fixed-L4 baseline and YaRN-confound runs for the paper rewrite.

Two phases, both on the sealed natural bank, both reporting generated exact
accuracy with every live request cost charged.

**Phase ``yarn``** — the confound check.  The published configuration applies
static YaRN factor 4 at every length, including lengths at or below the
32,768-token native context where no scaling is needed.  The dense arm reads
the whole scaled prompt; the compiled arm reads a roughly 1.6K-token packet
and is barely affected.  If dense recovers accuracy at factor 1, part of the
published dense-versus-compiled gap is rotary scaling, not compilation.  This
phase reruns dense and compiled at factor 4 and factor 1 over the lengths
where factor 1 is legal.

**Phase ``baselines``** — the matched-budget check.  Every selector spends the
same block budget and feeds the identical compiler, paragraph repair, and
packet format: the recursive ``tree``, a ``flat`` scan of every leaf with the
same query weights, ``bm25`` over the same block grid, and the
question-independent ``lead`` / ``tail`` / ``stride`` / ``random`` arms.  A
naive arm that matches the tree would mean the result is packet size, not
selection.

Peak allocated and reserved GPU memory are recorded per arm, which is also the
L4 replacement for the superseded A100 memory figure measured at
``feature_dim`` 512.

The runner asserts it is on the fixed reporting GPU unless ``--allow-any-gpu``
is passed.  Prompt synthesis is deterministic harness work, timed separately
and excluded from every request timer.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from benchmarks.benchmark_pre_qwen_e2e import (
    _model_read,
    _summarize_samples,
    run_dense_request,
)
from benchmarks.compare_dense_sparse import (
    _free_model,
    _load_model,
    _warmup,
    runtime_metadata,
)
from benchmarks.selector_baselines_cpu import (
    _atomic_json,
    _build_prompt,
    _case_id,
    _sha256,
)
from configs.long_context import (
    QWEN_NATIVE_CONTEXT,
    configure_tokenizer,
    context_limit,
    yarn_metadata,
)
from interfaces.evidence_compiler import (
    compile_evidence_packet_from_layout,
    locate_prompt_layout,
)
from scripts.extract_teacher_targets import (
    load_prompt_bank,
    needle_block_index,
)
from selector.baselines import (
    ALL_METHODS,
    POSITIONAL_METHODS,
    bm25_select,
    flat_select,
    positional_select,
)
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


REPORT_KIND = "spruce_paper_baselines_l4_v1"
REPORTING_GPU = "NVIDIA L4"


def _select_blocks(tokenizer, layout, index, weights, question, method, args):
    """Dispatch one selection rule; returns blocks and traversal accounting."""
    if method == "tree":
        selection = select_pre_qwen_blocks(
            index, weights, top_m=args.candidate_blocks,
            beam=args.beam, radix=args.radix)
        return selection.blocks, {
            "visited_nodes": int(selection.visited_nodes),
            "final_candidates": int(selection.final_candidates),
        }
    if method == "flat":
        selection = flat_select(index, weights, top_m=args.candidate_blocks)
    elif method == "bm25":
        selection = bm25_select(
            tokenizer, layout, question, args.block_size,
            top_m=args.candidate_blocks, k1=args.bm25_k1, b=args.bm25_b)
    elif method in POSITIONAL_METHODS:
        selection = positional_select(
            layout, args.block_size, method=method,
            top_m=args.candidate_blocks, seed=args.seed)
    else:
        raise ValueError(f"unknown selection method: {method!r}")
    return selection.blocks, {
        "visited_nodes": int(selection.scored_blocks),
        "final_candidates": int(selection.scored_blocks),
    }


def run_selected_request(model, tokenizer, target, method, args):
    """Charge the complete live request for one selection rule."""
    request_started = time.perf_counter()
    prompt = target["prompt_text"]
    user_prompt = target["user_prompt_text"]
    question = target["question"]

    started = time.perf_counter()
    layout = locate_prompt_layout(tokenizer, prompt, user_prompt, question)
    layout_tokenize_seconds = time.perf_counter() - started

    index = None
    weights = None
    index_seconds = 0.0
    if method in ("tree", "flat"):
        started = time.perf_counter()
        index = build_pre_qwen_index(
            layout, int(target["block_size"]),
            feature_dim=args.feature_dim,
            unigram_fraction=args.unigram_fraction,
            radix=args.radix)
        weights = question_feature_weights(
            tokenizer, question, index, idf_power=args.idf_power)
        index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    blocks, traversal = _select_blocks(
        tokenizer, layout, index, weights, question, method, args)
    selection_seconds = time.perf_counter() - started

    started = time.perf_counter()
    packet = compile_evidence_packet_from_layout(
        tokenizer, prompt, layout, question, blocks,
        int(target["block_size"]),
        block_radius=args.block_radius, boundary=args.boundary)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    encoded = tokenizer(packet.prompt, return_tensors="pt")
    compact_tokenize_seconds = time.perf_counter() - started

    needle_block = int(target["needle_block"])
    return _model_read(
        model, tokenizer, encoded, target,
        max_new_tokens=args.max_new_tokens,
        request_started=request_started,
        preprocessing={
            "method": method,
            "layout_tokenize_seconds": layout_tokenize_seconds,
            "index_seconds": index_seconds,
            "selection_seconds": selection_seconds,
            "compile_seconds": compile_seconds,
            "compact_tokenize_seconds": compact_tokenize_seconds,
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "original_input_tokens": len(layout.input_ids),
            "compression_fraction": float(packet.compression_fraction),
            "selected_blocks": [int(block) for block in blocks],
            "expanded_blocks": [
                int(block) for block in packet.expanded_blocks],
            "selected_contains_needle": needle_block in set(
                int(block) for block in blocks),
            "expanded_contains_needle": needle_block in set(
                int(block) for block in packet.expanded_blocks),
            "tree_levels": len(index.levels) if index is not None else 0,
            **traversal,
        },
    )


def _plan(args):
    """Build the ordered work plan, grouped so each model loads once.

    Factor-4 groups come first so the baselines phase and the factor-4 half of
    the yarn phase share a single model load; the factor-1 load happens last.
    """
    groups = []
    methods = [method for method in args.methods if method in ALL_METHODS]
    if "baselines" in args.phases:
        groups.append({
            "phase": "baselines",
            "yarn_factor": float(args.yarn_factor),
            "lengths": [int(length) for length in args.lengths],
            "arms": (["dense"] if args.include_dense else []) + methods,
        })
    if "yarn" in args.phases:
        native = context_limit(
            yarn_factor=1.0,
            original_max_position_embeddings=(
                args.original_max_position_embeddings))
        legal = [
            int(length) for length in args.yarn_lengths
            if int(length) <= native
        ]
        if not legal:
            raise SystemExit(
                "--yarn-lengths must contain a length within the native "
                f"context of {native} tokens")
        for factor in (float(args.yarn_factor), 1.0):
            groups.append({
                "phase": "yarn",
                "yarn_factor": factor,
                "lengths": legal,
                "arms": ["dense", "tree"],
            })
    return groups


def _row_id(phase, yarn_factor, arm, candidate_id):
    return f"{phase}|y{float(yarn_factor):g}|{arm}|{candidate_id}"


def _expected_config(args, prompt_bank_hash):
    return {
        "prompt_bank": os.path.abspath(args.prompt_bank),
        "prompt_bank_sha256": prompt_bank_hash,
        "model": args.model,
        "phases": list(args.phases),
        "lengths": [int(length) for length in args.lengths],
        "yarn_lengths": [int(length) for length in args.yarn_lengths],
        "depths": [float(depth) for depth in args.depths],
        "seed": int(args.seed),
        "block_size": int(args.block_size),
        "candidate_blocks": int(args.candidate_blocks),
        "block_radius": int(args.block_radius),
        "boundary": args.boundary,
        "beam": int(args.beam),
        "feature_dim": int(args.feature_dim),
        "unigram_fraction": float(args.unigram_fraction),
        "idf_power": float(args.idf_power),
        "radix": int(args.radix),
        "bm25_k1": float(args.bm25_k1),
        "bm25_b": float(args.bm25_b),
        "repeats": int(args.repeats),
        "max_new_tokens": int(args.max_new_tokens),
        "dtype": args.dtype,
        "yarn_factor": float(args.yarn_factor),
        "original_max_position_embeddings": int(
            args.original_max_position_embeddings),
        "methods": list(args.methods),
        "include_dense": bool(args.include_dense),
    }


def _load_resume(path, config):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"{path} is not a {REPORT_KIND} report")
    if report.get("config") != config:
        raise ValueError(
            "existing report configuration differs from requested run")
    return report


def _accuracy(rows):
    exact = sum(int(row["exact"]) for row in rows)
    return {
        "prompts": len(rows),
        "exact_count": exact,
        "exact_rate": exact / max(1, len(rows)),
    }


def aggregate_rows(rows):
    """Group generated accuracy, cost and memory by phase, YaRN factor, arm."""
    groups = {}
    for row in rows:
        key = f"{row['phase']}|yarn{float(row['yarn_factor']):g}|{row['arm']}"
        groups.setdefault(key, []).append(row)

    by_arm = {}
    for key, bucket in sorted(groups.items()):
        entry = {
            **_accuracy(bucket),
            "sum_request_seconds": sum(
                float(row["request_seconds"]) for row in bucket),
            "median_request_seconds": statistics.median(
                row["request_seconds"] for row in bucket),
            "median_input_tokens": statistics.median(
                row["input_tokens"] for row in bucket),
            "median_peak_memory_allocated_gb": statistics.median(
                row["peak_memory_allocated_gb"] for row in bucket),
            "median_peak_memory_reserved_gb": statistics.median(
                row["peak_memory_reserved_gb"] for row in bucket),
            "by_length": {
                str(length): _accuracy([
                    row for row in bucket
                    if int(row["requested_length"]) == int(length)])
                for length in sorted({
                    int(row["requested_length"]) for row in bucket})
            },
        }
        recalls = [
            row for row in bucket
            if row.get("expanded_contains_needle") is not None
        ]
        if recalls:
            entry["expanded_needle_recall"] = sum(
                int(row["expanded_contains_needle"]) for row in recalls
            ) / len(recalls)
            entry["direct_needle_recall"] = sum(
                int(row["selected_contains_needle"]) for row in recalls
            ) / len(recalls)
        by_arm[key] = entry

    dense_by_prompt = {
        (row["phase"], float(row["yarn_factor"]), row["candidate_id"]):
            bool(row["exact"])
        for row in rows if row["arm"] == "dense"
    }
    paired = {}
    for row in rows:
        if row["arm"] == "dense":
            continue
        key = (row["phase"], float(row["yarn_factor"]), row["candidate_id"])
        if key not in dense_by_prompt:
            continue
        label = (
            f"{row['phase']}|yarn{float(row['yarn_factor']):g}|{row['arm']}")
        counts = paired.setdefault(label, {
            "prompts": 0, "both_correct": 0, "arm_only": 0,
            "dense_only": 0, "neither_correct": 0})
        dense_exact = dense_by_prompt[key]
        arm_exact = bool(row["exact"])
        counts["prompts"] += 1
        if dense_exact and arm_exact:
            counts["both_correct"] += 1
        elif arm_exact:
            counts["arm_only"] += 1
        elif dense_exact:
            counts["dense_only"] += 1
        else:
            counts["neither_correct"] += 1

    return {"by_arm": by_arm, "paired_against_dense": paired}


def write_summary(report, path):
    """Write the human-readable L4 summary."""
    aggregate = report.get("aggregate") or {}
    by_arm = aggregate.get("by_arm", {})
    config = report["config"]
    lines = [
        "# SPRUCE paper baselines on the fixed L4",
        "",
        f"- generated: {report.get('created_utc')}",
        f"- GPU: {report.get('runtime', {}).get('gpu_name')}",
        f"- model: {config['model']}",
        f"- feature_dim: {config['feature_dim']}, "
        f"M={config['candidate_blocks']}, beam={config['beam']}, "
        f"B={config['block_size']}",
        f"- rows: {len(report.get('rows', []))} "
        f"(status {report.get('status')})",
        "",
        "## Per arm",
        "",
        "| phase / yarn / arm | n | exact | median req s | median tok | "
        "peak alloc GB | expanded recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in by_arm.items():
        recall = values.get("expanded_needle_recall")
        recall_cell = f"{recall:.4f}" if recall is not None else "n/a"
        lines.append(
            f"| {key} | {values['prompts']} | "
            f"{values['exact_count']}/{values['prompts']} "
            f"({values['exact_rate']:.4f}) | "
            f"{values['median_request_seconds']:.4f} | "
            f"{values['median_input_tokens']:.0f} | "
            f"{values['median_peak_memory_allocated_gb']:.3f} | "
            f"{recall_cell} |")

    paired = aggregate.get("paired_against_dense", {})
    if paired:
        lines += [
            "",
            "## Paired against dense, same phase and YaRN factor",
            "",
            "| arm | n | both | arm only | dense only | neither |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, counts in paired.items():
            lines.append(
                f"| {label} | {counts['prompts']} | "
                f"{counts['both_correct']} | {counts['arm_only']} | "
                f"{counts['dense_only']} | {counts['neither_correct']} |")

    lines += [
        "",
        "## Reading this",
        "",
        "- The yarn phase is a confound check. If dense at factor 1 scores",
        "  materially above dense at factor 4 on the same prompts, part of",
        "  the published dense-versus-compiled gap is rotary scaling.",
        "- The baselines phase is a matched-budget check. Every arm spends",
        "  the same block budget through the same compiler, so a naive arm",
        "  that matches the tree means the result is packet size, not",
        "  selection.",
        "- Memory columns are the L4 replacement for the superseded A100",
        "  figure measured at feature_dim 512.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(report, directory):
    """Emit PNG and PDF figures; returns written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = report.get("aggregate") or {}
    by_arm = aggregate.get("by_arm", {})
    if not by_arm:
        return []
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = []

    def _save(figure, stem):
        for suffix in ("png", "pdf"):
            path = target / f"{stem}.{suffix}"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            written.append(str(path))
        plt.close(figure)

    baselines = {
        key: values for key, values in by_arm.items()
        if key.startswith("baselines|")
    }
    if baselines:
        names = [key.split("|")[-1] for key in baselines]
        figure, axis = plt.subplots(figsize=(6.4, 3.4))
        axis.bar(
            range(len(names)),
            [values["exact_rate"] for values in baselines.values()])
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=20)
        axis.set_ylabel("exact accuracy")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("Generated accuracy at a matched block budget (L4)")
        _save(figure, "fig_l4_baseline_accuracy")

    yarn = {
        key: values for key, values in by_arm.items()
        if key.startswith("yarn|")
    }
    if yarn:
        names = list(yarn)
        figure, axis = plt.subplots(figsize=(6.4, 3.4))
        axis.bar(
            range(len(names)),
            [values["exact_rate"] for values in yarn.values()])
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(
            [name.replace("yarn|", "") for name in names], rotation=20)
        axis.set_ylabel("exact accuracy")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("Static YaRN confound check (L4)")
        _save(figure, "fig_l4_yarn_confound")

    names = list(by_arm)
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    axis.bar(
        range(len(names)),
        [values["median_peak_memory_allocated_gb"]
         for values in by_arm.values()])
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=35, fontsize=6)
    axis.set_ylabel("median peak allocated (GB)")
    axis.set_title("Peak allocated GPU memory per arm (L4)")
    _save(figure, "fig_l4_memory")

    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--phases", nargs="+", default=["baselines", "yarn"],
        choices=("baselines", "yarn"))
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=[16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072])
    parser.add_argument(
        "--yarn-lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument(
        "--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"), default="paragraph")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"),
        default="float16")
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument(
        "--methods", nargs="+", default=list(ALL_METHODS),
        choices=list(ALL_METHODS))
    parser.add_argument(
        "--include-dense", action="store_true",
        help="also run a dense arm inside the baselines phase")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--figures-dir")
    parser.add_argument("--session-budget-minutes", type=float, default=280.0)
    parser.add_argument("--allow-any-gpu", action="store_true")
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_paper_baselines_offload"))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    gpu_name = torch.cuda.get_device_name(0)
    if not args.allow_any_gpu and REPORTING_GPU not in gpu_name:
        raise SystemExit(
            f"this run must use the fixed reporting GPU {REPORTING_GPU!r}, "
            f"found {gpu_name!r}; pass --allow-any-gpu only for smoke tests")
    if not os.path.isfile(args.prompt_bank):
        raise SystemExit(f"prompt bank not found: {args.prompt_bank}")
    if args.beam < args.candidate_blocks:
        raise SystemExit("--beam must be >= --candidate-blocks")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise SystemExit("--repeats and --max-new-tokens must be >= 1")
    if any(not 0.0 <= float(depth) <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0, 1]")

    started_at = time.perf_counter()
    cases = load_prompt_bank(args.prompt_bank)
    config = _expected_config(args, _sha256(args.prompt_bank))
    try:
        report = _load_resume(args.out, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if report is None:
        report = {
            "kind": REPORT_KIND,
            "status": "running",
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "runtime": runtime_metadata(),
            "config": config,
            "rope": yarn_metadata(
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=(
                    args.original_max_position_embeddings)),
            "suite": {
                "distribution": "sealed_unscreened_natural_prose",
                "prompt_bank_cases": len(cases),
                "reporting_gpu": REPORTING_GPU,
            },
            "rows": [],
            "aggregate": None,
        }
        _atomic_json(args.out, report)

    completed = {row["row_id"] for row in report["rows"]}
    work = [
        (group, case, length, depth, arm)
        for group in _plan(args)
        for case in cases
        for length in group["lengths"]
        for depth in args.depths
        for arm in group["arms"]
    ]
    total = len(work)
    budget_seconds = float(args.session_budget_minutes) * 60.0
    stopped_early = False

    model = None
    loaded_factor = None
    prompt_cache = {}
    try:
        for position, (group, case, length, depth, arm) in enumerate(
                work, start=1):
            candidate_id = _case_id(case, length, depth, args.seed)
            row_id = _row_id(
                group["phase"], group["yarn_factor"], arm, candidate_id)
            if row_id in completed:
                continue
            if budget_seconds > 0 and (
                    time.perf_counter() - started_at) > budget_seconds:
                stopped_early = True
                print(
                    "session budget reached, stopping at "
                    f"{len(completed)}/{total}", flush=True)
                break

            if loaded_factor != group["yarn_factor"]:
                if model is not None:
                    model = _free_model(model)
                configure_tokenizer(
                    tokenizer, yarn_factor=group["yarn_factor"],
                    original_max_position_embeddings=(
                        args.original_max_position_embeddings))
                model = _load_model(
                    args.model, "sdpa", dtype, args.load_offload_dir,
                    yarn_factor=group["yarn_factor"],
                    original_max_position_embeddings=(
                        args.original_max_position_embeddings))
                _warmup(
                    model,
                    tokenizer(
                        "Read the supplied document and answer briefly.",
                        return_tensors="pt"))
                loaded_factor = group["yarn_factor"]
                # Prompt text depends on the tokenizer configuration, so the
                # cache cannot survive a rotary reconfiguration.
                prompt_cache.clear()

            print(
                f"[{position}/{total}] {group['phase']} "
                f"yarn={group['yarn_factor']:g} {arm} {candidate_id}",
                flush=True)

            if candidate_id not in prompt_cache:
                target = _build_prompt(
                    tokenizer, case, length, depth, args.seed,
                    args.max_new_tokens)
                target["question"] = case["question"]
                target["block_size"] = int(args.block_size)
                target["answers"] = list(case["answers"])
                target["reference_answers"] = list(case["answers"])
                target["prompt_format"] = "qwen_chat_v1"
                target["needle_block"] = needle_block_index(
                    tokenizer, target["prompt_text"], target["needle"],
                    args.block_size)
                prompt_cache[candidate_id] = target
            target = prompt_cache[candidate_id]

            samples = []
            for _ in range(args.repeats):
                if arm == "dense":
                    samples.append(run_dense_request(
                        model, tokenizer, target,
                        max_new_tokens=args.max_new_tokens))
                else:
                    samples.append(run_selected_request(
                        model, tokenizer, target, arm, args))
            summary = _summarize_samples(samples)

            row = {
                "row_id": row_id,
                "phase": group["phase"],
                "yarn_factor": float(group["yarn_factor"]),
                "arm": arm,
                "candidate_id": candidate_id,
                "source_case_id": case["id"],
                "genre": case.get("genre"),
                "requested_length": int(length),
                "depth": float(depth),
                "prompt_token_budget": target["prompt_token_budget"],
                "needle_block": int(target["needle_block"]),
                "prompt_sha256": target["prompt_sha256"],
                "exact": bool(summary["exact"]),
                "fuzzy": float(summary["fuzzy"]),
                "answer": summary["answer"],
                "answer_repeat_match": bool(summary["answer_repeat_match"]),
                "request_seconds": float(summary["request_seconds"]),
                "prefill_seconds": float(summary["prefill_seconds"]),
                "decode_seconds": float(summary["decode_seconds"]),
                "input_tokens": int(summary["input_tokens"]),
                "peak_memory_allocated_gb": float(
                    summary["peak_memory_allocated_gb"]),
                "peak_memory_reserved_gb": float(
                    summary["peak_memory_reserved_gb"]),
                "selected_blocks": summary.get("selected_blocks"),
                "expanded_blocks": summary.get("expanded_blocks"),
                "selected_contains_needle": summary.get(
                    "selected_contains_needle"),
                "expanded_contains_needle": summary.get(
                    "expanded_contains_needle"),
                "compression_fraction": summary.get("compression_fraction"),
            }
            report["rows"].append(row)
            completed.add(row_id)
            report["aggregate"] = aggregate_rows(report["rows"])
            report["status"] = "running"
            _atomic_json(args.out, report)
            print(
                f"  exact={int(row['exact'])} "
                f"tokens={row['input_tokens']} "
                f"request={row['request_seconds']:.3f}s",
                flush=True)
    finally:
        if model is not None:
            model = _free_model(model)

    report["aggregate"] = aggregate_rows(report["rows"])
    report["status"] = (
        "running" if stopped_early or len(completed) < total else "completed")
    report["completed_rows"] = len(report["rows"])
    report["expected_rows"] = total
    _atomic_json(args.out, report)

    if args.summary:
        write_summary(report, args.summary)
    if args.figures_dir:
        write_figures(report, args.figures_dir)

    print(json.dumps(report["aggregate"]["by_arm"], indent=2), flush=True)
    print(
        f"wrote {args.out} ({report['status']}, "
        f"{len(report['rows'])}/{total} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
