"""Combine natural YaRN length reports into paper tables and figures."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics


COMBINED_KIND = "spruce_pre_qwen_natural_yarn_suite_v1"
OUTCOMES = (
    "both_correct", "compiled_only", "dense_only", "neither_correct")
COLORS = {
    "dense": "#263238",
    "compiled": "#00897B",
    "compiled_only": "#43A047",
    "dense_only": "#E53935",
    "both_correct": "#90A4AE",
    "neither_correct": "#F9A825",
}


def _atomic_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)


def _wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _exact_binomial_two_sided(left, right):
    discordant = int(left) + int(right)
    if discordant == 0:
        return 1.0
    smaller = min(int(left), int(right))
    tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_delta(cases, iterations=10_000, seed=20260728):
    by_case = {}
    for case in cases:
        by_case.setdefault(case["source_case_id"], []).append(case)
    case_ids = sorted(by_case)
    rng = random.Random(int(seed))
    deltas = []
    for _iteration in range(int(iterations)):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        dense = 0
        compiled = 0
        count = 0
        for case_id in sampled:
            for row in by_case[case_id]:
                dense += int(row["dense"]["exact"])
                compiled += int(row["compiled"]["exact"])
                count += 1
        deltas.append((compiled - dense) / max(1, count))
    return {
        "iterations": int(iterations),
        "clusters": len(case_ids),
        "seed": int(seed),
        "lower_95": _percentile(deltas, 0.025),
        "median": _percentile(deltas, 0.5),
        "upper_95": _percentile(deltas, 0.975),
    }


def _accuracy(rows, mode):
    exact = sum(int(row[mode]["exact"]) for row in rows)
    lower, upper = _wilson(exact, len(rows))
    return {
        "cases": len(rows),
        "exact_count": exact,
        "exact_rate": exact / max(1, len(rows)),
        "wilson_95_low": lower,
        "wilson_95_high": upper,
    }


def _paired(rows):
    counts = {
        outcome: sum(row["paired_outcome"] == outcome for row in rows)
        for outcome in OUTCOMES
    }
    counts["mcnemar_exact_p"] = _exact_binomial_two_sided(
        counts["compiled_only"], counts["dense_only"])
    counts["accuracy_delta"] = (
        _accuracy(rows, "compiled")["exact_rate"]
        - _accuracy(rows, "dense")["exact_rate"])
    return counts


def _mean(rows, mode, metric):
    return statistics.mean(float(row[mode][metric]) for row in rows)


def _median(rows, mode, metric):
    return statistics.median(float(row[mode][metric]) for row in rows)


def _quartiles(values):
    materialized = [float(value) for value in values]
    return {
        "q25": _percentile(materialized, 0.25),
        "median": _percentile(materialized, 0.5),
        "q75": _percentile(materialized, 0.75),
    }


def summarize_group(rows):
    dense_sum = sum(float(row["dense"]["request_seconds"]) for row in rows)
    compiled_sum = sum(
        float(row["compiled"]["request_seconds"]) for row in rows)
    speedup_distribution = _quartiles(
        float(row["dense"]["request_seconds"])
        / float(row["compiled"]["request_seconds"])
        for row in rows)
    return {
        "cases": len(rows),
        "dense": _accuracy(rows, "dense"),
        "compiled": _accuracy(rows, "compiled"),
        "paired": _paired(rows),
        "sum_dense_request_seconds": dense_sum,
        "sum_compiled_request_seconds": compiled_sum,
        "sum_weighted_speedup": (
            dense_sum / compiled_sum if compiled_sum else float("inf")),
        "median_dense_request_seconds": _median(
            rows, "dense", "request_seconds"),
        "median_compiled_request_seconds": _median(
            rows, "compiled", "request_seconds"),
        "median_dense_prefill_seconds": _median(
            rows, "dense", "prefill_seconds"),
        "median_compiled_prefill_seconds": _median(
            rows, "compiled", "prefill_seconds"),
        "median_dense_input_tokens": _median(rows, "dense", "input_tokens"),
        "median_compiled_input_tokens": _median(
            rows, "compiled", "input_tokens"),
        "median_prompt_build_seconds": statistics.median(
            float(row.get("prompt_build_seconds", 0.0)) for row in rows),
        "median_compression_fraction": statistics.median(
            float(row["compiled"]["compression_fraction"]) for row in rows),
        "case_speedup_distribution": speedup_distribution,
        "median_dense_peak_memory_allocated_gb": _median(
            rows, "dense", "peak_memory_allocated_gb"),
        "median_compiled_peak_memory_allocated_gb": _median(
            rows, "compiled", "peak_memory_allocated_gb"),
        "median_dense_peak_memory_reserved_gb": _median(
            rows, "dense", "peak_memory_reserved_gb"),
        "median_compiled_peak_memory_reserved_gb": _median(
            rows, "compiled", "peak_memory_reserved_gb"),
        "median_compiled_visited_nodes": _median(
            rows, "compiled", "visited_nodes"),
        "direct_evidence_recall": statistics.mean(
            float(row["compiled"]["selected_contains_needle"])
            for row in rows),
        "expanded_evidence_recall": statistics.mean(
            float(row["compiled"]["expanded_contains_needle"])
            for row in rows),
        "mean_compiled_layout_seconds": _mean(
            rows, "compiled", "layout_tokenize_seconds"),
        "mean_compiled_index_seconds": _mean(
            rows, "compiled", "index_seconds"),
        "mean_compiled_selection_seconds": _mean(
            rows, "compiled", "selection_seconds"),
        "mean_compiled_stitch_seconds": _mean(
            rows, "compiled", "compile_seconds"),
        "mean_compiled_compact_tokenize_seconds": _mean(
            rows, "compiled", "compact_tokenize_seconds"),
        "mean_compiled_transfer_seconds": _mean(
            rows, "compiled", "input_transfer_seconds"),
        "mean_compiled_prefill_seconds": _mean(
            rows, "compiled", "prefill_seconds"),
        "mean_compiled_decode_seconds": _mean(
            rows, "compiled", "decode_seconds"),
    }


def _group(cases, key):
    grouped = {}
    for case in cases:
        grouped.setdefault(key(case), []).append(case)
    return grouped


def build_summary(cases, bootstrap_iterations=10_000):
    by_length_rows = _group(cases, lambda row: int(row["requested_length"]))
    by_depth_rows = _group(cases, lambda row: float(row["depth"]))
    by_case_rows = _group(cases, lambda row: row["source_case_id"])
    by_length_depth_rows = _group(
        cases,
        lambda row: (
            int(row["requested_length"]), float(row["depth"])))
    overall = summarize_group(cases)
    overall["cluster_bootstrap_accuracy_delta"] = _cluster_bootstrap_delta(
        cases, iterations=bootstrap_iterations)
    return {
        "overall": overall,
        "by_length": {
            str(key): summarize_group(rows)
            for key, rows in sorted(by_length_rows.items())
        },
        "by_depth": {
            str(key): summarize_group(rows)
            for key, rows in sorted(by_depth_rows.items())
        },
        "by_semantic_case": {
            key: summarize_group(rows)
            for key, rows in sorted(by_case_rows.items())
        },
        "by_length_depth": {
            f"{length}:{depth}": summarize_group(rows)
            for (length, depth), rows in sorted(
                by_length_depth_rows.items())
        },
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(cases, summary, tables_dir):
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    raw = []
    for row in cases:
        raw.append({
            "candidate_id": row["candidate_id"],
            "source_case_id": row["source_case_id"],
            "genre": row.get("genre"),
            "requested_length": row["requested_length"],
            "seq_len": row["seq_len"],
            "depth": row["depth"],
            "needle_block": row["needle_block"],
            "paired_outcome": row["paired_outcome"],
            "prompt_build_seconds": row.get("prompt_build_seconds", 0.0),
            "dense_exact": int(row["dense"]["exact"]),
            "compiled_exact": int(row["compiled"]["exact"]),
            "dense_answer": row["dense"]["answer"].strip(),
            "compiled_answer": row["compiled"]["answer"].strip(),
            "dense_request_seconds": row["dense"]["request_seconds"],
            "compiled_request_seconds": row["compiled"]["request_seconds"],
            "case_speedup": (
                row["dense"]["request_seconds"]
                / row["compiled"]["request_seconds"]),
            "dense_prefill_seconds": row["dense"]["prefill_seconds"],
            "compiled_prefill_seconds": row["compiled"]["prefill_seconds"],
            "dense_peak_memory_allocated_gb": (
                row["dense"]["peak_memory_allocated_gb"]),
            "compiled_peak_memory_allocated_gb": (
                row["compiled"]["peak_memory_allocated_gb"]),
            "dense_peak_memory_reserved_gb": (
                row["dense"]["peak_memory_reserved_gb"]),
            "compiled_peak_memory_reserved_gb": (
                row["compiled"]["peak_memory_reserved_gb"]),
            "compiled_input_tokens": row["compiled"]["input_tokens"],
            "compression_fraction": row["compiled"]["compression_fraction"],
            "selected_contains_needle": int(
                row["compiled"]["selected_contains_needle"]),
            "expanded_contains_needle": int(
                row["compiled"]["expanded_contains_needle"]),
            "visited_nodes": row["compiled"]["visited_nodes"],
            "prompt_sha256": row["prompt_sha256"],
        })
    _write_csv(tables / "raw_paired_cases.csv", raw)

    length_rows = []
    for length, item in summary["by_length"].items():
        length_rows.append({
            "requested_length": int(length),
            "cases": item["cases"],
            "dense_exact_count": item["dense"]["exact_count"],
            "dense_exact_rate": item["dense"]["exact_rate"],
            "dense_wilson_low": item["dense"]["wilson_95_low"],
            "dense_wilson_high": item["dense"]["wilson_95_high"],
            "compiled_exact_count": item["compiled"]["exact_count"],
            "compiled_exact_rate": item["compiled"]["exact_rate"],
            "compiled_wilson_low": item["compiled"]["wilson_95_low"],
            "compiled_wilson_high": item["compiled"]["wilson_95_high"],
            "accuracy_delta": item["paired"]["accuracy_delta"],
            "compiled_only": item["paired"]["compiled_only"],
            "dense_only": item["paired"]["dense_only"],
            "both_correct": item["paired"]["both_correct"],
            "neither_correct": item["paired"]["neither_correct"],
            "mcnemar_exact_p": item["paired"]["mcnemar_exact_p"],
            "sum_weighted_speedup": item["sum_weighted_speedup"],
            "median_dense_request_seconds": (
                item["median_dense_request_seconds"]),
            "median_compiled_request_seconds": (
                item["median_compiled_request_seconds"]),
            "median_compiled_input_tokens": (
                item["median_compiled_input_tokens"]),
            "median_prompt_build_seconds": (
                item["median_prompt_build_seconds"]),
            "median_compression_fraction": (
                item["median_compression_fraction"]),
            "case_speedup_q25": (
                item["case_speedup_distribution"]["q25"]),
            "case_speedup_median": (
                item["case_speedup_distribution"]["median"]),
            "case_speedup_q75": (
                item["case_speedup_distribution"]["q75"]),
            "median_dense_peak_memory_allocated_gb": (
                item["median_dense_peak_memory_allocated_gb"]),
            "median_compiled_peak_memory_allocated_gb": (
                item["median_compiled_peak_memory_allocated_gb"]),
            "median_dense_peak_memory_reserved_gb": (
                item["median_dense_peak_memory_reserved_gb"]),
            "median_compiled_peak_memory_reserved_gb": (
                item["median_compiled_peak_memory_reserved_gb"]),
            "median_compiled_visited_nodes": (
                item["median_compiled_visited_nodes"]),
            "direct_evidence_recall": item["direct_evidence_recall"],
            "expanded_evidence_recall": item["expanded_evidence_recall"],
        })
    _write_csv(tables / "by_length.csv", length_rows)

    depth_rows = []
    for depth, item in summary["by_depth"].items():
        depth_rows.append({
            "depth": float(depth),
            "cases": item["cases"],
            "dense_exact_rate": item["dense"]["exact_rate"],
            "compiled_exact_rate": item["compiled"]["exact_rate"],
            "accuracy_delta": item["paired"]["accuracy_delta"],
            "compiled_only": item["paired"]["compiled_only"],
            "dense_only": item["paired"]["dense_only"],
            "sum_weighted_speedup": item["sum_weighted_speedup"],
            "expanded_evidence_recall": item["expanded_evidence_recall"],
        })
    _write_csv(tables / "by_depth.csv", depth_rows)

    case_rows = []
    for case_id, item in summary["by_semantic_case"].items():
        case_rows.append({
            "source_case_id": case_id,
            "cases": item["cases"],
            "dense_exact_rate": item["dense"]["exact_rate"],
            "compiled_exact_rate": item["compiled"]["exact_rate"],
            "accuracy_delta": item["paired"]["accuracy_delta"],
            "compiled_only": item["paired"]["compiled_only"],
            "dense_only": item["paired"]["dense_only"],
            "sum_weighted_speedup": item["sum_weighted_speedup"],
            "expanded_evidence_recall": item["expanded_evidence_recall"],
        })
    _write_csv(tables / "by_semantic_case.csv", case_rows)

    grid_rows = []
    for key, item in summary["by_length_depth"].items():
        length, depth = key.split(":")
        grid_rows.append({
            "requested_length": int(length),
            "depth": float(depth),
            "cases": item["cases"],
            "dense_exact_rate": item["dense"]["exact_rate"],
            "compiled_exact_rate": item["compiled"]["exact_rate"],
            "accuracy_delta": item["paired"]["accuracy_delta"],
            "compiled_only": item["paired"]["compiled_only"],
            "dense_only": item["paired"]["dense_only"],
            "sum_weighted_speedup": item["sum_weighted_speedup"],
            "expanded_evidence_recall": item["expanded_evidence_recall"],
        })
    _write_csv(tables / "by_length_depth.csv", grid_rows)
    return {
        "raw_paired_cases": str(tables / "raw_paired_cases.csv"),
        "by_length": str(tables / "by_length.csv"),
        "by_depth": str(tables / "by_depth.csv"),
        "by_semantic_case": str(tables / "by_semantic_case.csv"),
        "by_length_depth": str(tables / "by_length_depth.csv"),
    }


def _save_figure(fig, figures_dir, stem):
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return str(png), str(pdf)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.22,
        "savefig.facecolor": "white",
    })
    return plt


def _length_data(summary):
    return [
        (int(length), item)
        for length, item in sorted(
            summary["by_length"].items(),
            key=lambda pair: int(pair[0]))
    ]


def make_figures(summary, figures_dir):
    plt = _style()
    outputs = {}
    length_data = _length_data(summary)
    lengths = [length / 1024 for length, _item in length_data]

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for mode, label in (
            ("dense", "Dense YaRN teacher"),
            ("compiled", "SPRUCE beam-16 compiler")):
        rates = [item[mode]["exact_rate"] for _length, item in length_data]
        lower = [
            rate - item[mode]["wilson_95_low"]
            for rate, (_length, item) in zip(rates, length_data)]
        upper = [
            item[mode]["wilson_95_high"] - rate
            for rate, (_length, item) in zip(rates, length_data)]
        axis.errorbar(
            lengths, rates, yerr=[lower, upper], marker="o",
            linewidth=2, capsize=3, label=label, color=COLORS[mode])
    axis.set(
        title="Unscreened natural retrieval accuracy vs context length",
        xlabel="Requested context (Ki tokens)", ylabel="Exact-answer rate",
        ylim=(-0.03, 1.05), xticks=lengths)
    axis.grid(True)
    axis.legend()
    outputs["01_accuracy_by_length"] = _save_figure(
        fig, figures_dir, "01_accuracy_by_length")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    bottom = [0] * len(lengths)
    labels = {
        "both_correct": "Both correct",
        "compiled_only": "Compiler only",
        "dense_only": "Dense only",
        "neither_correct": "Neither",
    }
    for outcome in OUTCOMES:
        values = [
            item["paired"][outcome] for _length, item in length_data]
        axis.bar(
            lengths, values, bottom=bottom, width=10,
            label=labels[outcome], color=COLORS[outcome])
        bottom = [a + b for a, b in zip(bottom, values)]
    axis.set(
        title="Paired correctness outcomes by context length",
        xlabel="Requested context (Ki tokens)", ylabel="Prompt count",
        xticks=lengths)
    axis.legend(ncol=2)
    axis.grid(True, axis="y")
    outputs["02_paired_outcomes"] = _save_figure(
        fig, figures_dir, "02_paired_outcomes")
    plt.close(fig)

    paper_style = {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 220,
    }
    with plt.rc_context(paper_style):
        positions = list(range(len(length_data)))
        figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.10))
        axes[0].plot(
            positions,
            [item["median_dense_request_seconds"]
             for _length, item in length_data],
            "o-", color="#2b2b2b", label="Dense")
        axes[0].plot(
            positions,
            [item["median_compiled_request_seconds"]
             for _length, item in length_data],
            "o-", color="#009e73", label="SPRUCE")
        axes[0].legend(frameon=False, loc="upper left")
        axes[0].set_title("(a) Fully charged request latency")
        axes[0].set_ylabel("Median request time (s)")
        axes[0].set_ylim(bottom=0)

        axes[1].axhline(
            1.0, color="#666666", linestyle="--", linewidth=0.8)
        axes[1].plot(
            positions,
            [item["sum_weighted_speedup"]
             for _length, item in length_data],
            "o-", color="#5B4BD6")
        axes[1].text(
            0.12, 1.18, "parity", color="#666666", fontsize=7.5)
        axes[1].set_title("(b) Sum-weighted dense / SPRUCE speedup")
        axes[1].set_ylabel("Speedup (×)")
        axes[1].set_ylim(bottom=0)

        for axis in axes:
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [int(length / 1024) for length, _item in length_data])
            axis.set_xlabel("Requested context (Ki tokens)")
            axis.grid(alpha=0.3, linewidth=0.4)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        figure.tight_layout(pad=0.3, w_pad=1.0)
        figures = Path(figures_dir)
        figures.mkdir(parents=True, exist_ok=True)
        png = figures / "03_latency_and_speedup.png"
        pdf = figures / "03_latency_and_speedup.pdf"
        figure.savefig(png, dpi=220)
        figure.savefig(pdf)
        outputs["03_latency_and_speedup"] = (str(png), str(pdf))
        plt.close(figure)

    fig, axis = plt.subplots(figsize=(8.6, 5.0))
    component_keys = [
        ("mean_compiled_layout_seconds", "Offset tokenization"),
        ("mean_compiled_index_seconds", "Index construction"),
        ("mean_compiled_selection_seconds", "Question + traversal"),
        ("mean_compiled_stitch_seconds", "Span stitching"),
        ("mean_compiled_compact_tokenize_seconds", "Compact tokenization"),
        ("mean_compiled_transfer_seconds", "Transfer"),
        ("mean_compiled_prefill_seconds", "Compact prefill"),
        ("mean_compiled_decode_seconds", "Decode"),
    ]
    bottoms = [0.0] * len(lengths)
    palette = plt.get_cmap("viridis")
    for index, (key, label) in enumerate(component_keys):
        values = [item[key] for _length, item in length_data]
        axis.bar(
            lengths, values, bottom=bottoms, width=10, label=label,
            color=palette((index + 1) / (len(component_keys) + 1)))
        bottoms = [a + b for a, b in zip(bottoms, values)]
    axis.set(
        title="SPRUCE fully charged request components",
        xlabel="Requested context (Ki tokens)", ylabel="Mean seconds",
        xticks=lengths)
    axis.legend(ncol=2)
    axis.grid(True, axis="y")
    outputs["04_compiled_latency_components"] = _save_figure(
        fig, figures_dir, "04_compiled_latency_components")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(
        lengths,
        [item["median_dense_input_tokens"] / 1024
         for _length, item in length_data],
        marker="o", color=COLORS["dense"], label="Original")
    axes[0].plot(
        lengths,
        [item["median_compiled_input_tokens"] / 1024
         for _length, item in length_data],
        marker="o", color=COLORS["compiled"], label="Compiled")
    axes[0].set(
        title="Tokens read by Qwen", xlabel="Requested context (Ki tokens)",
        ylabel="Median input (Ki tokens)", xticks=lengths)
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(
        lengths,
        [100 * item["median_compression_fraction"]
         for _length, item in length_data],
        marker="o", color="#FB8C00")
    axes[1].set(
        title="Compiled fraction of original",
        xlabel="Requested context (Ki tokens)", ylabel="Median percent",
        xticks=lengths)
    axes[1].grid(True)
    outputs["05_context_compression"] = _save_figure(
        fig, figures_dir, "05_context_compression")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        lengths,
        [item["direct_evidence_recall"] for _length, item in length_data],
        marker="o", linewidth=2, label="Direct M=4 block recall",
        color="#F4511E")
    axis.plot(
        lengths,
        [item["expanded_evidence_recall"] for _length, item in length_data],
        marker="o", linewidth=2, label="Radius-1 expanded recall",
        color="#3949AB")
    axis.set(
        title="Evidence-location recall vs context length",
        xlabel="Requested context (Ki tokens)", ylabel="Recall",
        ylim=(-0.03, 1.05), xticks=lengths)
    axis.grid(True)
    axis.legend()
    outputs["06_evidence_recall"] = _save_figure(
        fig, figures_dir, "06_evidence_recall")
    plt.close(fig)

    heatmap_style = {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 220,
    }
    with plt.rc_context(heatmap_style):
        depths = sorted(
            (float(depth) for depth in summary["by_depth"]), reverse=True)
        matrices = []
        for mode in ("dense", "compiled"):
            matrices.append([
                [
                    summary["by_length_depth"][
                        f"{int(length * 1024)}:{depth}"
                    ][mode]["exact_rate"]
                    for length in lengths
                ]
                for depth in depths
            ])
        matrices.append([
            [
                summary["by_length_depth"][
                    f"{int(length * 1024)}:{depth}"
                ]["paired"]["accuracy_delta"]
                for length in lengths
            ]
            for depth in depths
        ])
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.74))
        for axis, matrix, cmap, title, (vmin, vmax) in (
                (axes[0], matrices[0], "Greys", "(a) Dense exact", (0, 1)),
                (axes[1], matrices[1], "Blues", "(b) SPRUCE exact", (0, 1)),
                (axes[2], matrices[2], "RdYlGn",
                 "(c) SPRUCE $-$ dense", (-1, 1))):
            image = axis.imshow(
                matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axis.set_xticks(range(len(lengths)))
            axis.set_xticklabels([int(value) for value in lengths])
            axis.set_yticks(range(len(depths)))
            axis.set_yticklabels([f"{depth:.1f}" for depth in depths])
            axis.set_xlabel("Context (Ki tokens)")
            if axis is axes[0]:
                axis.set_ylabel("Evidence depth")
            axis.set_title(title)
            for row_index, row in enumerate(matrix):
                for column_index, value in enumerate(row):
                    dark = (
                        (value - vmin) / (vmax - vmin) > 0.6
                        if cmap != "RdYlGn"
                        else False
                    )
                    axis.text(
                        column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=6.2,
                        color="white" if dark else "black")
            colorbar = fig.colorbar(
                image, ax=axis, fraction=0.045, pad=0.03)
            colorbar.ax.tick_params(labelsize=6)
            colorbar.outline.set_linewidth(0.5)
        fig.tight_layout(pad=0.3, w_pad=1.0)
        figures = Path(figures_dir)
        figures.mkdir(parents=True, exist_ok=True)
        png = figures / "07_length_depth_heatmaps.png"
        pdf = figures / "07_length_depth_heatmaps.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        outputs["07_length_depth_heatmaps"] = (str(png), str(pdf))
        plt.close(fig)

    case_items = sorted(summary["by_semantic_case"].items())
    case_labels = [case_id.replace("paper_", "") for case_id, _ in case_items]
    positions = list(range(len(case_items)))
    fig, axis = plt.subplots(
        figsize=(9.0, max(5.2, len(case_items) * 0.42)))
    axis.barh(
        [position + 0.18 for position in positions],
        [item["dense"]["exact_rate"] for _case, item in case_items],
        height=0.34, color=COLORS["dense"], label="Dense")
    axis.barh(
        [position - 0.18 for position in positions],
        [item["compiled"]["exact_rate"] for _case, item in case_items],
        height=0.34, color=COLORS["compiled"], label="Compiler")
    axis.set(
        title="Accuracy by sealed semantic case", xlabel="Exact-answer rate",
        yticks=positions, yticklabels=case_labels, xlim=(0, 1.03))
    axis.grid(True, axis="x")
    axis.legend()
    outputs["08_semantic_case_accuracy"] = _save_figure(
        fig, figures_dir, "08_semantic_case_accuracy")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    axes[0, 0].plot(
        lengths,
        [item["dense"]["exact_rate"] for _length, item in length_data],
        marker="o", color=COLORS["dense"], label="Dense")
    axes[0, 0].plot(
        lengths,
        [item["compiled"]["exact_rate"] for _length, item in length_data],
        marker="o", color=COLORS["compiled"], label="Compiler")
    axes[0, 0].set(title="Exact accuracy", ylabel="Rate", ylim=(-0.03, 1.03))
    axes[0, 0].legend()
    axes[0, 1].plot(
        lengths,
        [item["sum_weighted_speedup"] for _length, item in length_data],
        marker="o", color="#5E35B1")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Fully charged speedup", ylabel="Dense / compiler (×)")
    axes[1, 0].plot(
        lengths,
        [item["expanded_evidence_recall"] for _length, item in length_data],
        marker="o", color="#3949AB")
    axes[1, 0].set(title="Expanded evidence recall", ylabel="Recall",
                   ylim=(-0.03, 1.03))
    axes[1, 1].plot(
        lengths,
        [100 * item["median_compression_fraction"]
         for _length, item in length_data],
        marker="o", color="#FB8C00")
    axes[1, 1].set(title="Context retained", ylabel="Percent")
    for axis in axes.flatten():
        axis.set_xlabel("Requested context (Ki tokens)")
        axis.set_xticks(lengths)
        axis.grid(True)
    fig.suptitle("SPRUCE beam-16 unscreened natural YaRN scaling")
    outputs["09_paper_overview"] = _save_figure(
        fig, figures_dir, "09_paper_overview")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        lengths,
        [item["median_dense_peak_memory_allocated_gb"]
         for _length, item in length_data],
        marker="o", color=COLORS["dense"], label="Dense")
    axis.plot(
        lengths,
        [item["median_compiled_peak_memory_allocated_gb"]
         for _length, item in length_data],
        marker="o", color=COLORS["compiled"], label="Compiler")
    axis.set(
        title="Peak allocated GPU memory", ylabel="GiB",
        xlabel="Requested context (Ki tokens)", xticks=lengths)
    axis.grid(True)
    axis.legend()
    outputs["10_peak_gpu_memory"] = _save_figure(
        fig, figures_dir, "10_peak_gpu_memory")
    plt.close(fig)

    speedup_medians = [
        item["case_speedup_distribution"]["median"]
        for _length, item in length_data]
    speedup_lower = [
        median - item["case_speedup_distribution"]["q25"]
        for median, (_length, item) in zip(speedup_medians, length_data)]
    speedup_upper = [
        item["case_speedup_distribution"]["q75"] - median
        for median, (_length, item) in zip(speedup_medians, length_data)]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.errorbar(
        lengths, speedup_medians, yerr=[speedup_lower, speedup_upper],
        marker="o", linewidth=2, capsize=4, color="#5E35B1",
        label="Median and interquartile range")
    axis.set(
        title="Per-prompt fully charged speedup distribution",
        xlabel="Requested context (Ki tokens)",
        ylabel="Dense / compiler request time (×)", xticks=lengths)
    axis.grid(True)
    axis.legend()
    outputs["11_case_speedup_distribution"] = _save_figure(
        fig, figures_dir, "11_case_speedup_distribution")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        lengths,
        [item["median_prompt_build_seconds"]
         for _length, item in length_data],
        marker="o", linewidth=2, color="#6D4C41")
    axis.set(
        title="Synthetic prompt construction cost (excluded from requests)",
        xlabel="Requested context (Ki tokens)",
        ylabel="Median harness seconds", xticks=lengths)
    axis.grid(True)
    outputs["12_prompt_build_time_excluded"] = _save_figure(
        fig, figures_dir, "12_prompt_build_time_excluded")
    plt.close(fig)
    return outputs


def write_markdown_summary(summary, path):
    overall = summary["overall"]
    bootstrap = overall["cluster_bootstrap_accuracy_delta"]
    lines = [
        "# SPRUCE unscreened natural YaRN sweep",
        "",
        "## Overall paired result",
        "",
        f"- Paired prompts: {overall['cases']}",
        (
            f"- Dense exact: {overall['dense']['exact_count']}/"
            f"{overall['dense']['cases']} "
            f"({overall['dense']['exact_rate']:.3%})"),
        (
            f"- Compiler exact: {overall['compiled']['exact_count']}/"
            f"{overall['compiled']['cases']} "
            f"({overall['compiled']['exact_rate']:.3%})"),
        (
            f"- Accuracy delta: "
            f"{overall['paired']['accuracy_delta']:+.3%}"),
        (
            f"- Compiler-only / dense-only: "
            f"{overall['paired']['compiled_only']} / "
            f"{overall['paired']['dense_only']}"),
        f"- Exact McNemar p: {overall['paired']['mcnemar_exact_p']:.6g}",
        (
            f"- Semantic-case cluster bootstrap delta 95% interval: "
            f"[{bootstrap['lower_95']:+.3%}, "
            f"{bootstrap['upper_95']:+.3%}]"),
        (
            f"- Fully charged sum-weighted speedup: "
            f"{overall['sum_weighted_speedup']:.3f}x"),
        (
            f"- Expanded evidence recall: "
            f"{overall['expanded_evidence_recall']:.3%}"),
        "",
        "## By requested context length",
        "",
        "| Ki tokens | N | Dense | Compiler | Delta | Compiler-only | Dense-only | Speedup | Expanded recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length, item in _length_data(summary):
        lines.append(
            f"| {length / 1024:.0f} | {item['cases']} | "
            f"{item['dense']['exact_rate']:.1%} | "
            f"{item['compiled']['exact_rate']:.1%} | "
            f"{item['paired']['accuracy_delta']:+.1%} | "
            f"{item['paired']['compiled_only']} | "
            f"{item['paired']['dense_only']} | "
            f"{item['sum_weighted_speedup']:.2f}x | "
            f"{item['expanded_evidence_recall']:.1%} |")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- Prompts were generated from the sealed paper bank without dense screening.",
        "- Prompt synthesis and model loading are harness setup, not request latency.",
        "- Both modes use the same static YaRN configuration and generated prompt.",
        "- Raw paired McNemar treats prompt rows independently; the case-clustered bootstrap is the more conservative semantic-diversity check.",
        "- A positive compiler-minus-dense result supports this controlled natural-retrieval distribution only; it is not a general quality claim.",
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output)


def combine_reports(
        report_paths, output_dir, bootstrap_iterations=10_000):
    reports = []
    for path in report_paths:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") != "completed":
            raise ValueError(f"incomplete length report: {path}")
        reports.append(report)
    if not reports:
        raise ValueError("at least one report is required")
    lengths = [
        int(report["config"]["requested_length"]) for report in reports]
    if lengths != sorted(lengths) or len(lengths) != len(set(lengths)):
        raise ValueError(
            "length reports must be strictly increasing and duplicate-free")
    reference = {
        key: value
        for key, value in reports[0]["config"].items()
        if key != "requested_length"
    }
    for report, length in zip(reports, lengths):
        comparable = {
            key: value
            for key, value in report["config"].items()
            if key != "requested_length"
        }
        if comparable != reference:
            raise ValueError(
                f"configuration differs in length report {length}")
        expected = int(report["suite"]["paired_cases_expected"])
        if len(report["cases"]) != expected:
            raise ValueError(
                f"length report {length} has {len(report['cases'])} cases; "
                f"expected {expected}")
        if any(
                int(case["requested_length"]) != length
                for case in report["cases"]):
            raise ValueError(
                f"length report {length} contains a mismatched case")

    cases = [case for report in reports for case in report["cases"]]
    candidate_ids = [case["candidate_id"] for case in cases]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("combined reports contain duplicate candidate IDs")
    summary = build_summary(
        cases, bootstrap_iterations=bootstrap_iterations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = write_tables(cases, summary, output / "tables")
    figures = make_figures(summary, output / "figures")
    markdown = write_markdown_summary(summary, output / "SUMMARY.md")

    combined = {
        "kind": COMBINED_KIND,
        "model": reports[0]["model"],
        "source_reports": [os.path.abspath(path) for path in report_paths],
        "suite": {
            **reports[0]["suite"],
            "lengths": [
                int(report["config"]["requested_length"])
                for report in reports],
            "depths": reports[0]["config"]["depths"],
            "paired_cases": len(cases),
            "semantic_cases": len({
                case["source_case_id"] for case in cases}),
            "screened_prompts": False,
            "teacher_definition": (
                "same frozen Qwen backbone with full dense YaRN attention"),
            "selector_definition": (
                "tokenizer-only D=512 radix-2 hierarchy, beam=16, M=4"),
        },
        "config_by_length": [
            report["config"] for report in reports],
        "runtime_by_length": [
            report["runtime"] for report in reports],
        "summary": summary,
        "tables": tables,
        "figures": figures,
        "markdown_summary": markdown,
        "cases": cases,
    }
    _atomic_json(output / "combined_report.json", combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        raise SystemExit("--bootstrap-iterations must be >= 1")
    combined = combine_reports(
        args.reports, args.out_dir,
        bootstrap_iterations=args.bootstrap_iterations)
    print(json.dumps(combined["summary"]["overall"], indent=2), flush=True)
    print(f"combined report -> {Path(args.out_dir) / 'combined_report.json'}")
    print(f"paper figures   -> {Path(args.out_dir) / 'figures'}")
    print(f"paper tables    -> {Path(args.out_dir) / 'tables'}")


if __name__ == "__main__":
    raise SystemExit(main())
