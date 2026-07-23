"""Plot SPRUCE efficiency and retrieval accuracy versus context length."""
import argparse
import csv
import json
import os
import statistics


def _mean(cases, section, metric):
    return statistics.mean(float(case[section][metric]) for case in cases)


def build_scaling_series(report):
    """Aggregate live-tree benchmark cases that share a sequence length."""
    grouped = {}
    for case in report["cases"]:
        grouped.setdefault(int(case["seq_len"]), []).append(case)

    series = []
    for seq_len, cases in sorted(grouped.items()):
        dense_prefill = _mean(cases, "dense", "prefill_seconds")
        sparse_prefill = _mean(cases, "sparse", "prefill_seconds")
        live_prefill = _mean(cases, "sparse", "live_prefill_seconds")
        series.append({
            "seq_len": seq_len,
            "targets": len(cases),
            "dense_prefill_seconds": dense_prefill,
            "sparse_kernel_prefill_seconds": sparse_prefill,
            "sparse_live_prefill_seconds": live_prefill,
            "kernel_prefill_speedup": dense_prefill / sparse_prefill,
            "live_prefill_speedup": dense_prefill / live_prefill,
            "sparse_exact_rate": statistics.mean(
                float(case["sparse"]["exact"]) for case in cases),
            "dense_exact_rate": statistics.mean(
                float(case["dense"]["exact"]) for case in cases),
            "sparse_fuzzy": _mean(cases, "sparse", "fuzzy"),
            "answers_match_rate": statistics.mean(
                float(case["answers_match"]) for case in cases),
            "tree_build_seconds": _mean(cases, "sparse", "tree_build_seconds"),
            "tree_traversal_seconds": _mean(
                cases, "sparse", "tree_traversal_seconds"),
            "route_pack_seconds": _mean(cases, "sparse", "route_pack_seconds"),
        })
    return series


def write_scaling_csv(series, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    return path


def save_efficiency_accuracy_plot(report, path, csv_path=None):
    """Write a four-panel scaling plot and its aggregated CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = build_scaling_series(report)
    if not series:
        raise ValueError("benchmark report contains no cases")
    x = [row["seq_len"] / 1024 for row in series]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    latency, speedup, accuracy, selector = axes.flatten()

    latency.plot(x, [row["dense_prefill_seconds"] for row in series],
                 marker="o", label="Dense SDPA")
    latency.plot(x, [row["sparse_kernel_prefill_seconds"] for row in series],
                 marker="o", label="Sparse kernel")
    latency.plot(x, [row["sparse_live_prefill_seconds"] for row in series],
                 marker="o", label="Live tree + sparse")
    latency.set(title="Prefill latency", ylabel="Seconds")
    latency.legend()

    speedup.axhline(1.0, color="black", linewidth=1, linestyle="--")
    speedup.plot(x, [row["kernel_prefill_speedup"] for row in series],
                 marker="o", label="Kernel-only")
    speedup.plot(x, [row["live_prefill_speedup"] for row in series],
                 marker="o", label="Tree-inclusive")
    speedup.set(title="Dense / SPRUCE prefill speedup", ylabel="Speedup (×)")
    speedup.legend()

    accuracy.plot(x, [row["sparse_exact_rate"] for row in series],
                  marker="o", label="Sparse exact")
    accuracy.plot(x, [row["dense_exact_rate"] for row in series],
                  marker="o", label="Dense exact")
    accuracy.plot(x, [row["sparse_fuzzy"] for row in series],
                  marker="o", label="Sparse fuzzy")
    accuracy.plot(x, [row["answers_match_rate"] for row in series],
                  marker="o", label="Answers match")
    accuracy.set(title="Retrieval accuracy", ylabel="Score", ylim=(-0.03, 1.03))
    accuracy.legend()

    selector.plot(x, [row["tree_build_seconds"] for row in series],
                  marker="o", label="Tree build")
    selector.plot(x, [row["tree_traversal_seconds"] for row in series],
                  marker="o", label="Traversal")
    selector.plot(x, [row["route_pack_seconds"] for row in series],
                  marker="o", label="Route packing")
    selector.set(title="Live selector overhead", ylabel="Seconds")
    selector.legend()

    for axis in axes[-1]:
        axis.set_xlabel("Context length (K tokens)")
    for axis in axes.flatten():
        axis.grid(alpha=0.2)

    selector_config = report.get("selector", {})
    fig.suptitle(
        f"SPRUCE held-out scaling — {report.get('model', 'model')}  "
        f"beam={selector_config.get('beam', '?')} "
        f"K={selector_config.get('k_selected', '?')} "
        f"selector={selector_config.get('compute_dtype', 'float32')}",
        fontsize=13,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)

    if csv_path is None:
        csv_path = os.path.splitext(path)[0] + ".csv"
    write_scaling_csv(series, csv_path)
    return path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="live-tree benchmark JSON")
    parser.add_argument("--out", required=True, help="PNG output path")
    parser.add_argument("--csv", help="optional aggregated CSV output path")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    plot_path, csv_path = save_efficiency_accuracy_plot(
        report, args.out, csv_path=args.csv)
    print(f"scaling plot -> {plot_path}")
    print(f"scaling data -> {csv_path}")


if __name__ == "__main__":
    main()
