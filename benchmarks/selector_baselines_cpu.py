"""Selection-level baseline comparison on the sealed natural bank (CPU only).

Answers two questions without loading model weights:

1. **Is the tree doing anything a flat scan does not?**  The ``flat`` arm
   scores every leaf block with the identical query weights and takes the same
   top-M.  If ``flat`` and ``tree`` agree on every prompt, the tree's
   contribution is cost, not accuracy, and the paper must say so.
2. **Does an ordinary retriever already solve this?**  The ``bm25`` arm ranks
   the same block grid with Okapi BM25.  ``lead`` / ``tail`` / ``stride`` /
   ``random`` spend the same block budget without reading the question at all.

Every arm feeds the same evidence compiler, the same paragraph repair, and the
same packet format, so the only variable is the selection rule.  Metrics are
evidence recall and packet size, not generated accuracy: this runner never
loads Qwen.  Generated accuracy for the same arms is measured on the fixed L4
by ``benchmarks/paper_baselines_l4.py``.

Prompt synthesis is deterministic harness work and is timed separately from
selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.natural_context import (
    build_natural_prompt_calibrated,
    format_instruct_chat_prompt,
)
from interfaces.evidence_compiler import (
    compile_evidence_packet_from_layout,
    locate_prompt_layout,
)
from scripts.extract_teacher_targets import (
    depth_tag,
    load_prompt_bank,
    needle_block_index,
)
from selector.baselines import (
    ALL_METHODS,
    bm25_select,
    flat_select,
    positional_select,
    selection_agreement,
)
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


REPORT_KIND = "spruce_selector_baselines_cpu_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)


def _case_id(case, length, depth, seed):
    return (
        f"{case['id']}_L{int(length)}_"
        f"d{depth_tag(float(depth))}_s{int(seed)}"
    )


def _build_prompt(tokenizer, case, length, depth, seed, generation_reserve):
    """Synthesize one paper-format prompt; returns the prompt and its needle."""
    budget = int(length) - int(generation_reserve)
    if budget < 1:
        raise ValueError("generation reserve consumes the prompt budget")
    started = time.perf_counter()
    prompt, needle, natural_units, user_prompt = (
        build_natural_prompt_calibrated(
            tokenizer,
            budget,
            case,
            float(depth),
            seed=int(seed),
            prompt_formatter=lambda content: format_instruct_chat_prompt(
                tokenizer, content),
            return_content=True,
        )
    )
    return {
        "prompt_text": prompt,
        "user_prompt_text": user_prompt,
        "needle": needle,
        "natural_units": int(natural_units),
        "prompt_token_budget": budget,
        "build_seconds": time.perf_counter() - started,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest().upper(),
    }


def _packet_metrics(
        tokenizer, prompt, layout, question, blocks, block_size, *,
        block_radius, boundary, needle_block):
    """Compile one packet and report evidence survival and packet size."""
    started = time.perf_counter()
    packet = compile_evidence_packet_from_layout(
        tokenizer, prompt, layout, question, tuple(blocks), int(block_size),
        block_radius=int(block_radius), boundary=boundary)
    compile_seconds = time.perf_counter() - started
    packet_tokens = len(tokenizer(packet.prompt)["input_ids"])
    return {
        "compile_seconds": compile_seconds,
        "expanded_blocks": [int(block) for block in packet.expanded_blocks],
        "packet_tokens": int(packet_tokens),
        "compression_fraction": float(packet.compression_fraction),
        "selected_contains_needle": int(needle_block) in set(
            int(block) for block in blocks),
        "expanded_contains_needle": int(needle_block) in set(
            int(block) for block in packet.expanded_blocks),
    }


def _run_arms(tokenizer, target, layout, index, weights, args, needle_block):
    """Run every requested selector over one already-built prompt layout."""
    prompt = target["prompt_text"]
    question = target["question"]
    arms = {}
    for method in args.methods:
        started = time.perf_counter()
        if method == "tree":
            selection = select_pre_qwen_blocks(
                index, weights, top_m=args.candidate_blocks,
                beam=args.beam, radix=args.radix)
            scored = int(selection.visited_nodes)
        elif method == "flat":
            selection = flat_select(
                index, weights, top_m=args.candidate_blocks)
            scored = int(selection.scored_blocks)
        elif method == "bm25":
            selection = bm25_select(
                tokenizer, layout, question, args.block_size,
                top_m=args.candidate_blocks, k1=args.bm25_k1, b=args.bm25_b)
            scored = int(selection.scored_blocks)
        else:
            selection = positional_select(
                layout, args.block_size, method=method,
                top_m=args.candidate_blocks, seed=args.seed)
            scored = int(selection.scored_blocks)
        select_seconds = time.perf_counter() - started

        arms[method] = {
            "method": method,
            "selected_blocks": [int(block) for block in selection.blocks],
            "select_seconds": select_seconds,
            "scored_blocks": scored,
            **_packet_metrics(
                tokenizer, prompt, layout, question, selection.blocks,
                args.block_size, block_radius=args.block_radius,
                boundary=args.boundary, needle_block=needle_block),
        }
    return arms


def _expected_config(args, prompt_bank_hash):
    return {
        "prompt_bank": os.path.abspath(args.prompt_bank),
        "prompt_bank_sha256": prompt_bank_hash,
        "tokenizer": args.model,
        "lengths": [int(length) for length in args.lengths],
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
        "generation_reserve": int(args.generation_reserve),
        "methods": list(args.methods),
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


def _rate(rows, key):
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def _arm_rows(rows, method):
    return [row["arms"][method] for row in rows if method in row["arms"]]


def aggregate_rows(rows, methods):
    """Summarize evidence recall, packet size and cost per arm."""
    per_method = {}
    for method in methods:
        arms = _arm_rows(rows, method)
        if not arms:
            continue
        per_method[method] = {
            "prompts": len(arms),
            "direct_needle_recall": _rate(arms, "selected_contains_needle"),
            "expanded_needle_recall": _rate(arms, "expanded_contains_needle"),
            "median_packet_tokens": statistics.median(
                arm["packet_tokens"] for arm in arms),
            "median_compression_fraction": statistics.median(
                arm["compression_fraction"] for arm in arms),
            "median_select_seconds": statistics.median(
                arm["select_seconds"] for arm in arms),
            "median_scored_blocks": statistics.median(
                arm["scored_blocks"] for arm in arms),
        }

    buckets = {}
    for row in rows:
        buckets.setdefault(str(row["requested_length"]), []).append(row)
    by_length = {
        length: {
            method: {
                "prompts": len(_arm_rows(bucket, method)),
                "direct_needle_recall": _rate(
                    _arm_rows(bucket, method), "selected_contains_needle"),
                "expanded_needle_recall": _rate(
                    _arm_rows(bucket, method), "expanded_contains_needle"),
            }
            for method in per_method
        }
        for length, bucket in sorted(
            buckets.items(), key=lambda item: int(item[0]))
    }

    agreement = None
    comparisons = [
        row["tree_vs_flat"] for row in rows
        if row.get("tree_vs_flat") is not None
    ]
    if comparisons:
        agreement = {
            "prompts": len(comparisons),
            "identical_set_rate": _rate(comparisons, "identical_set"),
            "identical_order_rate": _rate(comparisons, "identical_order"),
            "top1_match_rate": _rate(comparisons, "top1_match"),
            "mean_jaccard": sum(
                float(item["jaccard"]) for item in comparisons
            ) / len(comparisons),
            "disagreeing_prompts": [
                row["candidate_id"] for row in rows
                if row.get("tree_vs_flat")
                and not row["tree_vs_flat"]["identical_set"]
            ],
        }

    return {
        "prompts": len(rows),
        "by_method": per_method,
        "by_length": by_length,
        "tree_vs_flat": agreement,
    }


def write_summary(report, path):
    """Write the human-readable summary next to the JSON report."""
    aggregate = report.get("aggregate") or {}
    methods = aggregate.get("by_method", {})
    config = report["config"]
    lines = [
        "# SPRUCE selector baselines (CPU, selection level)",
        "",
        f"- generated: {report.get('created_utc')}",
        f"- prompts: {aggregate.get('prompts', 0)}",
        f"- tokenizer: {config['tokenizer']}",
        f"- feature_dim: {config['feature_dim']}, "
        f"M={config['candidate_blocks']}, beam={config['beam']}, "
        f"B={config['block_size']}",
        "",
        "No model weights are loaded. These are evidence-recall and",
        "packet-size numbers, not generated accuracy.",
        "",
        "## Per arm",
        "",
        "| arm | direct recall | expanded recall | median packet tok | "
        "median compression | median select s | blocks scored |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, values in methods.items():
        lines.append(
            f"| {method} | {values['direct_needle_recall']:.4f} | "
            f"{values['expanded_needle_recall']:.4f} | "
            f"{values['median_packet_tokens']:.0f} | "
            f"{values['median_compression_fraction']:.4f} | "
            f"{values['median_select_seconds']:.6f} | "
            f"{values['median_scored_blocks']:.0f} |")

    agreement = aggregate.get("tree_vs_flat")
    if agreement:
        lines += [
            "",
            "## Tree versus flat scan",
            "",
            f"- identical block set: {agreement['identical_set_rate']:.4f} "
            f"of {agreement['prompts']} prompts",
            f"- identical ranked order: "
            f"{agreement['identical_order_rate']:.4f}",
            f"- top-1 match: {agreement['top1_match_rate']:.4f}",
            f"- mean Jaccard: {agreement['mean_jaccard']:.4f}",
            "",
            "If these rates are 1.0 the tree is a cost result, not an",
            "accuracy result, and the paper must frame it that way.",
        ]

    if methods:
        names = list(methods)
        lines += [
            "",
            "## Per length, expanded evidence recall",
            "",
            "| length | " + " | ".join(names) + " |",
            "| --- | " + " | ".join("---:" for _ in names) + " |",
        ]
        for length, bucket in aggregate.get("by_length", {}).items():
            cells = " | ".join(
                f"{bucket[name]['expanded_needle_recall']:.4f}"
                for name in names)
            lines.append(f"| {length} | {cells} |")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(report, directory):
    """Emit PNG and PDF figures for the paper; returns written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = report.get("aggregate") or {}
    methods = list(aggregate.get("by_method", {}))
    if not methods:
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

    values = aggregate["by_method"]
    positions = list(range(len(methods)))

    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.bar(
        [position - 0.2 for position in positions],
        [values[method]["direct_needle_recall"] for method in methods],
        width=0.4, label="direct")
    axis.bar(
        [position + 0.2 for position in positions],
        [values[method]["expanded_needle_recall"] for method in methods],
        width=0.4, label="after paragraph repair")
    axis.set_xticks(positions)
    axis.set_xticklabels(methods, rotation=20)
    axis.set_ylabel("evidence recall")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.set_title("Evidence recall at a matched block budget")
    _save(figure, "fig_baseline_recall")

    lengths = sorted(aggregate.get("by_length", {}), key=int)
    if lengths:
        figure, axis = plt.subplots(figsize=(6.0, 3.4))
        for method in methods:
            axis.plot(
                [int(length) for length in lengths],
                [aggregate["by_length"][length][method][
                    "expanded_needle_recall"] for length in lengths],
                marker="o", label=method)
        axis.set_xlabel("requested context length (tokens)")
        axis.set_ylabel("expanded evidence recall")
        axis.set_ylim(0.0, 1.05)
        axis.legend(fontsize=7)
        axis.set_title("Evidence recall against context length")
        _save(figure, "fig_baseline_recall_by_length")

    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.bar(
        positions,
        [values[method]["median_packet_tokens"] for method in methods])
    axis.set_xticks(positions)
    axis.set_xticklabels(methods, rotation=20)
    axis.set_ylabel("median packet tokens")
    axis.set_title("Packet size at a matched block budget")
    _save(figure, "fig_baseline_packet_tokens")

    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="tokenizer only; no weights are loaded")
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=[16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072])
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
    parser.add_argument("--generation-reserve", type=int, default=32)
    parser.add_argument(
        "--methods", nargs="+", default=list(ALL_METHODS),
        choices=list(ALL_METHODS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--figures-dir")
    parser.add_argument(
        "--session-budget-minutes", type=float, default=0.0,
        help="stop cleanly after this many minutes; 0 disables the budget")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.prompt_bank):
        raise SystemExit(f"prompt bank not found: {args.prompt_bank}")
    if args.beam < args.candidate_blocks:
        raise SystemExit("--beam must be >= --candidate-blocks")
    if any(not 0.0 <= float(depth) <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0, 1]")
    if "tree" not in args.methods:
        raise SystemExit("--methods must include tree, the reference arm")

    started_at = time.perf_counter()
    cases = load_prompt_bank(args.prompt_bank)
    config = _expected_config(args, _sha256(args.prompt_bank))
    try:
        report = _load_resume(args.out, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if report is None:
        report = {
            "kind": REPORT_KIND,
            "status": "running",
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "config": config,
            "suite": {
                "distribution": "sealed_unscreened_natural_prose",
                "prompt_bank_cases": len(cases),
                "model_weights_loaded": False,
                "measures": "evidence recall and packet size, not accuracy",
            },
            "rows": [],
            "aggregate": None,
        }
        _atomic_json(args.out, report)

    completed = {row["candidate_id"] for row in report["rows"]}
    expected = [
        (case, length, depth)
        for case in cases
        for length in args.lengths
        for depth in args.depths
    ]
    total = len(expected)
    budget_seconds = float(args.session_budget_minutes) * 60.0
    stopped_early = False

    for position, (case, length, depth) in enumerate(expected, start=1):
        candidate_id = _case_id(case, length, depth, args.seed)
        if candidate_id in completed:
            continue
        if budget_seconds > 0 and (
                time.perf_counter() - started_at) > budget_seconds:
            stopped_early = True
            print(
                "session budget reached, stopping at "
                f"{len(completed)}/{total}", flush=True)
            break

        print(f"[{position}/{total}] {candidate_id}", flush=True)
        target = _build_prompt(
            tokenizer, case, length, depth, args.seed, args.generation_reserve)
        target["question"] = case["question"]
        prompt = target["prompt_text"]

        layout_started = time.perf_counter()
        layout = locate_prompt_layout(
            tokenizer, prompt, target["user_prompt_text"], case["question"])
        layout_seconds = time.perf_counter() - layout_started

        index_started = time.perf_counter()
        index = build_pre_qwen_index(
            layout, args.block_size,
            feature_dim=args.feature_dim,
            unigram_fraction=args.unigram_fraction,
            radix=args.radix)
        index_seconds = time.perf_counter() - index_started
        weights = question_feature_weights(
            tokenizer, case["question"], index, idf_power=args.idf_power)

        needle_block = needle_block_index(
            tokenizer, prompt, target["needle"], args.block_size)
        arms = _run_arms(
            tokenizer, target, layout, index, weights, args, needle_block)

        row = {
            "candidate_id": candidate_id,
            "source_case_id": case["id"],
            "genre": case.get("genre"),
            "requested_length": int(length),
            "depth": float(depth),
            "seed": int(args.seed),
            "seq_len": len(layout.input_ids),
            "needle_block": int(needle_block),
            "document_blocks": len(index.document_blocks),
            "tree_levels": len(index.levels),
            "prompt_sha256": target["prompt_sha256"],
            "prompt_build_seconds": target["build_seconds"],
            "layout_seconds": layout_seconds,
            "index_seconds": index_seconds,
            "arms": arms,
            "tree_vs_flat": (
                selection_agreement(
                    tuple(arms["tree"]["selected_blocks"]),
                    tuple(arms["flat"]["selected_blocks"]))
                if "flat" in arms and "tree" in arms else None),
        }
        report["rows"].append(row)
        completed.add(candidate_id)
        report["aggregate"] = aggregate_rows(report["rows"], args.methods)
        report["status"] = "running"
        _atomic_json(args.out, report)

        recalls = " ".join(
            f"{method}={int(arm['expanded_contains_needle'])}"
            for method, arm in arms.items())
        agreement = row["tree_vs_flat"]
        print(
            f"  expanded_recall {recalls}"
            + (f" tree==flat={int(agreement['identical_set'])}"
               if agreement else ""),
            flush=True)

    report["aggregate"] = aggregate_rows(report["rows"], args.methods)
    report["status"] = (
        "running" if stopped_early or len(completed) < total else "completed")
    report["completed_rows"] = len(report["rows"])
    report["expected_rows"] = total
    _atomic_json(args.out, report)

    if args.summary:
        write_summary(report, args.summary)
    if args.figures_dir:
        write_figures(report, args.figures_dir)

    print(
        json.dumps(report["aggregate"]["by_method"], indent=2), flush=True)
    print(
        f"wrote {args.out} ({report['status']}, "
        f"{len(report['rows'])}/{total} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
