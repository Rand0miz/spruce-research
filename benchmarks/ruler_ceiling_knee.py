"""Compiler-only beam x budget ceiling sweep: is the compiler still the bottleneck?

"Is SPRUCE more accurate than dense" needs a model to answer.  This runner asks
a different, model-independent question: **how well could any model possibly do
given the packet the compiler hands it?**  That is the compiler ceiling, and it
is what decides whether a stronger backbone would be held back by the compiler
rather than by itself.

Two levers, because they fix different failures:

``M`` (block budget)
    Recovers references that survived the descent into the final beam but
    ranked below the cut.  At tail=3 / D=1024 / M=4 that was 618 of 1,212
    selector misses.

``beam`` (traversal slack)
    Recovers references the descent discarded before final ranking at all.
    That was the other 594, and the count grows with length: 36 at 4K rising to
    187 at 128K.  **Raising M cannot fix any of them.**  Beam has been pinned at
    16 in every RULER run so far, so it is the unswept lever.

Compiler-only: a tokenizer, no model weights, no GPU.  Run it on a CPU runtime.
Timings it emits are CPU-dependent diagnostics and must never enter a paper
latency table; ceiling and packet fraction are hardware-independent.

It reports three things a global ceiling average hides:

1. **The minimum task-length cell ceiling.**  An 85% aggregate can conceal a
   cell at 40% where the compiler caps everything downstream.
2. **A pre-declared pass/fail** against ``--target-accuracy + --margin`` over
   in-scope tasks.  Declare the target before looking at the result.
3. **The ceiling-versus-packet-fraction knee.**  Ceiling can always be bought
   with a larger packet; the useful configuration is the knee, not the maximum.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ruler_cached_factorial import (
    DEFAULT_LENGTHS,
    TASK_TYPES,
    atomic_json,
    config_fingerprint,
    parse_arm as parse_factorial_arm,
    run_case,
)


REPORT_KIND = "spruce_ruler_ceiling_knee_v1"
VERDICT_FIELDS = (
    "ok", "data_absent", "selector_in_beam", "selector_outside_beam", "stitch")
# Aggregation tasks whose instruction names no distinctive term for a lexical
# sketch to hash. Declared out of scope here, before measurement, so a scoped
# claim is a stated design choice rather than a post-hoc rescue.
DEFAULT_OUT_OF_SCOPE = ("vt", "cwe", "fwe")


def arm_key(beam, budget):
    return f"beam{int(beam)}_M{int(budget)}"


def parse_arm(name):
    beam, budget = name.split("_M")
    return int(beam.removeprefix("beam")), int(budget)


def budgets_for_beam(beam, budgets):
    """A block budget wider than the beam is unreachable, so drop it."""
    return [int(budget) for budget in budgets if int(budget) <= int(beam)]


def summarize_verdicts(per_reference):
    """Split selector misses by whether the beam ever saw the reference.

    ``rank_in_beam`` is the reference's position among the blocks the traversal
    returned.  ``None`` means the descent discarded it before ranking, which
    only a wider beam can fix.  An integer at or beyond the budget means it
    survived the descent but lost the cut, which only a larger M fixes.
    """
    counts = {field: 0 for field in VERDICT_FIELDS}
    for reference in per_reference:
        verdict = reference["verdict"]
        if verdict == "ok":
            counts["ok"] += 1
        elif verdict == "data":
            counts["data_absent"] += 1
        elif verdict == "stitch":
            counts["stitch"] += 1
        elif reference.get("rank_in_beam") is None:
            counts["selector_outside_beam"] += 1
        else:
            counts["selector_in_beam"] += 1
    return counts


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_rows(rows, arms, *, in_scope_tasks, target, margin):
    """Per-arm ceiling, per-cell floor, miss causes, and the pass/fail rule."""
    threshold = float(target) + float(margin)
    in_scope_tasks = set(in_scope_tasks)
    by_arm = {}
    for arm in arms:
        entries = [(row, row["arms"][arm]) for row in rows if arm in row["arms"]]
        if not entries:
            continue

        cells = {}
        for row, values in entries:
            cells.setdefault(
                (row["task"], int(row["length"])), []).append(values["ceiling"])
        cell_ceilings = {
            f"{task}@{length}": _mean(scores)
            for (task, length), scores in sorted(cells.items())
        }
        in_scope_cells = {
            key: value for key, value in cell_ceilings.items()
            if key.split("@")[0] in in_scope_tasks
        }
        failing = sorted(
            (key for key, value in in_scope_cells.items() if value < threshold),
            key=lambda key: in_scope_cells[key])

        verdicts = {field: 0 for field in VERDICT_FIELDS}
        for _, values in entries:
            for field, count in values["verdicts"].items():
                verdicts[field] += count

        lengths = {}
        for row, values in entries:
            lengths.setdefault(
                str(int(row["length"])), []).append(values["ceiling"])

        beam, budget = parse_arm(arm)
        by_arm[arm] = {
            "beam": beam,
            "budget": budget,
            "samples": len(entries),
            "ceiling": _mean(values["ceiling"] for _, values in entries),
            "in_scope_ceiling": _mean(
                values["ceiling"] for row, values in entries
                if row["task"] in in_scope_tasks),
            "median_compression_fraction": statistics.median(
                values["compression_fraction"] for _, values in entries),
            "median_packet_tokens": statistics.median(
                values["compiled_prompt_tokens"] for _, values in entries),
            "cell_ceilings": cell_ceilings,
            "min_in_scope_cell": (
                min(in_scope_cells.values()) if in_scope_cells else 0.0),
            "min_in_scope_cell_name": (
                min(in_scope_cells, key=in_scope_cells.get)
                if in_scope_cells else None),
            "by_length": {
                length: _mean(scores)
                for length, scores in sorted(
                    lengths.items(), key=lambda item: int(item[0]))
            },
            "verdicts": verdicts,
            "failing_cells": failing,
            "passes_rule": bool(in_scope_cells) and not failing,
        }

    passing = [name for name, values in by_arm.items() if values["passes_rule"]]
    recommended = None
    if passing:
        # Cheapest arm that clears the rule: ceiling bought with packet size is
        # not free, so the knee is the smallest packet that passes.
        recommended = min(
            passing, key=lambda name: by_arm[name]["median_compression_fraction"])
    elif by_arm:
        # Nothing passes. The honest report is the highest cell floor reached,
        # which is the evidence that the compiler is still binding.
        recommended = max(
            by_arm, key=lambda name: by_arm[name]["min_in_scope_cell"])

    return {
        "samples": len(rows),
        "decision_rule": {
            "target_accuracy": float(target),
            "margin": float(margin),
            "threshold": threshold,
            "in_scope_tasks": sorted(in_scope_tasks),
            "statement": (
                "the compiler is not the bottleneck when every in-scope "
                f"task-length cell ceiling is at least {threshold:.4f}"),
        },
        "by_arm": by_arm,
        "passing_arms": passing,
        "recommended_arm": recommended,
        "prompt_ceiling": _mean(
            row["prompt_ceiling"] for row in rows if "prompt_ceiling" in row),
    }


def _ordered_arms(by_arm):
    return sorted(
        by_arm.items(), key=lambda item: (item[1]["beam"], item[1]["budget"]))


def write_summary(report, path):
    """Write the human-readable verdict next to the JSON report."""
    aggregate = report.get("aggregate") or {}
    by_arm = aggregate.get("by_arm", {})
    rule = aggregate.get("decision_rule", {})
    config = report["config"]
    lines = [
        "# SPRUCE compiler ceiling: beam x budget knee",
        "",
        f"- generated: {report.get('created_utc')}",
        f"- samples: {aggregate.get('samples', 0)} "
        f"(status {report.get('status')})",
        f"- fixed: tail={config['tail']}, D={config['feature_dim']}, "
        f"B={config['block_size']}, radius={config['block_radius']}",
        f"- full-prompt ceiling: {aggregate.get('prompt_ceiling', 0.0):.4f}",
        "",
        "Compiler-only. No model weights, no GPU. Ceiling is what a perfect",
        "model could score on the compiled packet, so it is the model-",
        "independent measure of whether the compiler is the bottleneck.",
        "",
        "## Pre-declared decision rule",
        "",
        f"- {rule.get('statement', 'n/a')}",
        f"- target {rule.get('target_accuracy')} + margin "
        f"{rule.get('margin')} = **{rule.get('threshold')}**",
        f"- in scope: {', '.join(rule.get('in_scope_tasks', []))}",
        "",
        "## Per arm",
        "",
        "| arm | beam | M | ceiling | in-scope | min cell | worst cell | "
        "packet frac | passes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for arm, values in _ordered_arms(by_arm):
        lines.append(
            f"| {arm} | {values['beam']} | {values['budget']} | "
            f"{values['ceiling']:.4f} | {values['in_scope_ceiling']:.4f} | "
            f"{values['min_in_scope_cell']:.4f} | "
            f"{values['min_in_scope_cell_name']} | "
            f"{values['median_compression_fraction']:.4f} | "
            f"{'yes' if values['passes_rule'] else 'no'} |")

    lines += [
        "",
        "## Why references are still missed",
        "",
        "`selector_outside_beam` is fixable only by a wider beam;",
        "`selector_in_beam` only by a larger M; `data_absent` by neither —",
        "it is a property of the benchmark, not of the compiler, and should",
        "be excluded before computing how much headroom is really left.",
        "",
        "| arm | ok | data absent | in beam | outside beam | stitch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, values in _ordered_arms(by_arm):
        counts = values["verdicts"]
        lines.append(
            f"| {arm} | {counts['ok']} | {counts['data_absent']} | "
            f"{counts['selector_in_beam']} | "
            f"{counts['selector_outside_beam']} | {counts['stitch']} |")

    recommended = aggregate.get("recommended_arm")
    if recommended and recommended in by_arm:
        values = by_arm[recommended]
        lines += ["", "## Recommendation", ""]
        if values["passes_rule"]:
            lines.append(
                f"- **{recommended}** clears the rule at the smallest packet. "
                "Run the end-to-end generation at this configuration.")
        else:
            lines.append(
                f"- **No arm clears the rule.** {recommended} reaches the "
                "highest in-scope cell floor, so the compiler is still the "
                "bottleneck; widen the sweep before spending GPU time.")
        lines.append(
            f"- ceiling {values['ceiling']:.4f}, in-scope cell floor "
            f"{values['min_in_scope_cell']:.4f}, median packet "
            f"{values['median_compression_fraction']:.4f} of the source prompt")
        if values["failing_cells"]:
            lines.append(
                "- worst failing cells: "
                + ", ".join(values["failing_cells"][:8]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(report, directory):
    """Emit PNG and PDF figures; returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = report.get("aggregate") or {}
    by_arm = aggregate.get("by_arm", {})
    if not by_arm:
        return []
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    threshold = aggregate["decision_rule"]["threshold"]
    written = []

    def _save(figure, stem):
        for suffix in ("png", "pdf"):
            path = target / f"{stem}.{suffix}"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            written.append(str(path))
        plt.close(figure)

    knee = sorted(
        by_arm.items(),
        key=lambda item: item[1]["median_compression_fraction"])
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        [values["median_compression_fraction"] for _, values in knee],
        [values["ceiling"] for _, values in knee],
        marker="o", label="ceiling, all tasks")
    axis.plot(
        [values["median_compression_fraction"] for _, values in knee],
        [values["min_in_scope_cell"] for _, values in knee],
        marker="s", label="worst in-scope cell")
    for name, values in knee:
        axis.annotate(
            name, (values["median_compression_fraction"], values["ceiling"]),
            fontsize=6, xytext=(3, 4), textcoords="offset points")
    axis.axhline(
        threshold, linestyle="--", linewidth=1, color="black",
        label=f"rule threshold {threshold:.2f}")
    axis.set_xlabel("median packet fraction of the source prompt")
    axis.set_ylabel("compiler ceiling")
    axis.legend(fontsize=7)
    axis.set_title("Ceiling versus packet cost: take the knee, not the maximum")
    _save(figure, "fig_ceiling_knee")

    lengths = sorted({
        int(length)
        for values in by_arm.values() for length in values["by_length"]})
    if lengths:
        figure, axis = plt.subplots(figsize=(6.4, 3.6))
        for name, values in _ordered_arms(by_arm):
            axis.plot(
                lengths,
                [values["by_length"].get(str(length), 0.0)
                 for length in lengths],
                marker="o", label=name)
        axis.axhline(threshold, linestyle="--", linewidth=1, color="black")
        axis.set_xlabel("context length (tokens)")
        axis.set_ylabel("compiler ceiling")
        axis.legend(fontsize=6)
        axis.set_title("Ceiling against length")
        _save(figure, "fig_ceiling_by_length")

    names = [name for name, _ in _ordered_arms(by_arm)]
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    bottom = [0.0] * len(names)
    for field, label in (
            ("selector_outside_beam", "outside beam (needs wider beam)"),
            ("selector_in_beam", "in beam, below cut (needs larger M)"),
            ("stitch", "stitch"),
            ("data_absent", "reference absent from document")):
        values = [by_arm[name]["verdicts"][field] for name in names]
        axis.bar(names, values, bottom=bottom, label=label)
        bottom = [carry + value for carry, value in zip(bottom, values)]
    axis.set_ylabel("missed references")
    axis.tick_params(axis="x", labelrotation=20, labelsize=7)
    axis.legend(fontsize=6)
    axis.set_title("What the compiler still misses, by cause")
    _save(figure, "fig_miss_causes")

    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="tokenizer only; no weights are loaded")
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--beams", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--budgets", type=int, nargs="+", default=[9, 16, 32])
    parser.add_argument("--tasks", nargs="+", default=sorted(TASK_TYPES))
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"), default="paragraph")
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--max-occurrences", type=int, default=8)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--out-of-scope-tasks", nargs="*", default=list(DEFAULT_OUT_OF_SCOPE))
    parser.add_argument("--session-budget-minutes", type=float, default=240.0)
    return parser.parse_args(argv)


def build_plan(beams, budgets):
    """Map each beam to the budgets it can actually reach."""
    beams = sorted({int(beam) for beam in beams})
    budgets = sorted({int(budget) for budget in budgets})
    return {beam: budgets_for_beam(beam, budgets) for beam in beams}


def main(argv=None):
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        raise SystemExit(f"data directory not found: {args.data_dir}")
    plan = build_plan(args.beams, args.budgets)
    if not any(plan.values()):
        raise SystemExit("no block budget is reachable with the given beams")
    in_scope = sorted(set(args.tasks) - set(args.out_of_scope_tasks or ()))
    if not in_scope:
        raise SystemExit("every requested task was declared out of scope")

    config = {
        "tail": int(args.tail),
        "feature_dim": int(args.feature_dim),
        "beams": sorted(plan),
        "budgets": sorted({int(budget) for budget in args.budgets}),
        "tasks": list(args.tasks),
        "lengths": [int(length) for length in args.lengths],
        "max_samples": int(args.max_samples),
        "block_size": int(args.block_size),
        "block_radius": int(args.block_radius),
        "boundary": args.boundary,
        "unigram_fraction": float(args.unigram_fraction),
        "idf_power": float(args.idf_power),
        "radix": int(args.radix),
        "max_occurrences": int(args.max_occurrences),
        "target_accuracy": float(args.target_accuracy),
        "margin": float(args.margin),
        "in_scope_tasks": in_scope,
    }
    fingerprint = config_fingerprint(config)
    arms = [
        arm_key(beam, budget)
        for beam in sorted(plan) for budget in plan[beam]
    ]

    report = None
    if args.out.is_file():
        report = json.loads(args.out.read_text(encoding="utf-8"))
        if report.get("kind") != REPORT_KIND:
            raise SystemExit(f"{args.out} is not a {REPORT_KIND} report")
        if report.get("fingerprint") != fingerprint:
            raise SystemExit(
                "existing report configuration differs from requested run")
    if report is None:
        report = {
            "kind": REPORT_KIND,
            "status": "running",
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "config": config,
            "fingerprint": fingerprint,
            "arms": arms,
            "rows": [],
            "aggregate": None,
        }
        atomic_json(args.out, report)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    done = {
        (row["task"], int(row["length"]), int(row["sample"]))
        for row in report["rows"]
    }
    # The M-prefix assertion inside run_case only has to hold once per beam,
    # so the validated set persists across samples instead of being rebuilt.
    prefix_validation = {beam: set() for beam in plan}
    started = time.perf_counter()
    budget_seconds = float(args.session_budget_minutes) * 60.0
    stopped_early = False

    for length in config["lengths"]:
        if stopped_early:
            break
        for task in config["tasks"]:
            if stopped_early:
                break
            path = args.data_dir / str(length) / task / "validation.jsonl"
            if not path.is_file():
                print(f"skip missing {path}", flush=True)
                continue
            with path.open("r", encoding="utf-8") as handle:
                samples = [json.loads(line) for line in handle if line.strip()]

            for position, sample in enumerate(samples[: args.max_samples]):
                if (task, int(length), position) in done:
                    continue
                if budget_seconds > 0 and (
                        time.perf_counter() - started) > budget_seconds:
                    stopped_early = True
                    print(
                        f"session budget reached after {len(report['rows'])} "
                        "samples", flush=True)
                    break

                row = {
                    "task": task,
                    "length": int(length),
                    "sample": position,
                    "arms": {},
                }
                for beam in sorted(plan):
                    if not plan[beam]:
                        continue
                    result = run_case(
                        sample, task, tokenizer,
                        tails=(int(args.tail),),
                        feature_dims=(int(args.feature_dim),),
                        budgets=plan[beam],
                        block_size=int(args.block_size),
                        block_radius=int(args.block_radius),
                        boundary=args.boundary,
                        beam=int(beam),
                        unigram_fraction=float(args.unigram_fraction),
                        idf_power=float(args.idf_power),
                        radix=int(args.radix),
                        max_occurrences=int(args.max_occurrences),
                        prefix_validation=prefix_validation[beam],
                    )
                    row["prompt_ceiling"] = result["prompt_ceiling"]
                    row["original_prompt_tokens"] = result[
                        "original_prompt_tokens"]
                    for name, values in result["arms"].items():
                        _, _, budget = parse_factorial_arm(name)
                        row["arms"][arm_key(beam, budget)] = {
                            "ceiling": values["ceiling"],
                            "compression_fraction": values[
                                "compression_fraction"],
                            "compiled_prompt_tokens": values[
                                "compiled_prompt_tokens"],
                            "verdicts": summarize_verdicts(
                                values["per_reference"]),
                        }

                report["rows"].append(row)
                done.add((task, int(length), position))
                report["aggregate"] = aggregate_rows(
                    report["rows"], arms, in_scope_tasks=in_scope,
                    target=args.target_accuracy, margin=args.margin)
                atomic_json(args.out, report)
                print(
                    f"{task}@{length}#{position} "
                    + " ".join(
                        f"{name}={row['arms'][name]['ceiling']:.2f}"
                        for name in sorted(row["arms"])),
                    flush=True)

    report["aggregate"] = aggregate_rows(
        report["rows"], arms, in_scope_tasks=in_scope,
        target=args.target_accuracy, margin=args.margin)
    report["status"] = "running" if stopped_early else "completed"
    atomic_json(args.out, report)
    if args.summary:
        write_summary(report, args.summary)
    if args.figures_dir:
        write_figures(report, args.figures_dir)

    aggregate = report["aggregate"]
    print(json.dumps({
        "threshold": aggregate["decision_rule"]["threshold"],
        "passing_arms": aggregate["passing_arms"],
        "recommended_arm": aggregate["recommended_arm"],
    }, indent=2), flush=True)
    print(
        f"wrote {args.out} ({report['status']}, "
        f"{len(report['rows'])} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
