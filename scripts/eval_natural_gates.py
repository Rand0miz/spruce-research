"""Compare selector gates on full held-out natural teacher targets.

This is a structural selector evaluation, not a generation benchmark. Each
gate runs the same live recursive traversal and K-width route packing used by
SPRUCE inference. The resulting routes are compared with dense teacher block
mass, with results aggregated by requested context length.

Use only natural held-out targets that were never included in selector
training. Feature-only artifacts are insufficient because they do not contain
the dense teacher target needed for recall and coverage.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.compare_dense_sparse_live_tree import live_route
from scripts.eval_tree_traversal import load_gate
from selector.targets import load_teacher


REPORT_KIND = "spruce_natural_gate_eval_v1"


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = (
            sorted(glob.glob(pattern))
            if any(character in pattern for character in "*?[")
            else [pattern]
        )
        paths.extend(matches)
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def parse_gate_specs(specifications):
    """Parse ``NAME=PATH`` or bare checkpoint paths into unique labels."""
    gates = []
    labels = set()
    for specification in specifications:
        if "=" in specification:
            label, path = specification.split("=", 1)
        else:
            path = specification
            label = os.path.splitext(os.path.basename(path))[0]
        label = label.strip()
        path = os.path.abspath(path.strip())
        if not label or not path:
            raise ValueError(f"invalid gate specification {specification!r}")
        if label in labels:
            raise ValueError(f"duplicate gate label {label!r}")
        labels.add(label)
        gates.append((label, path))
    return gates


def selected_blocks_to_mask(selected_blocks, key_blocks):
    """Convert ``[1,L,G,Q,K]`` packed IDs to ``[L,G,Q,KB]`` mask."""
    selected = selected_blocks[0].long()
    valid = (selected >= 0) & (selected < key_blocks)
    safe = selected.clamp(min=0, max=key_blocks)
    safe = safe.masked_fill(~valid, key_blocks)
    with_pad = torch.zeros(
        (*selected.shape[:-1], key_blocks + 1),
        dtype=torch.bool,
        device=selected.device,
    )
    with_pad.scatter_(-1, safe, True)
    return with_pad[..., :key_blocks]


def safe_ratio(value, ceiling):
    if ceiling <= 0:
        return float("nan")
    return min(float(value) / float(ceiling), 1.0)


@torch.no_grad()
def route_metrics(
        selected, target, cmask, budgets, needle_block, selection_budget):
    """Measure a packed live route against one dense teacher target."""
    layers, groups, query_blocks, key_blocks = target.shape
    selected = selected & cmask[None, None]
    valid_rows = target.sum(dim=-1) > 0.5
    query_ids = torch.arange(query_blocks, device=target.device)
    metrics = {}

    selected_count = selected.sum(dim=-1).float()
    metrics["avg_selected"] = float(
        (selected_count * valid_rows).sum()
        / valid_rows.sum().clamp_min(1)
    )
    coverage = (target * selected).sum(dim=-1)
    metrics["coverage"] = float(
        (coverage * valid_rows).sum()
        / valid_rows.sum().clamp_min(1)
    )

    route_width = min(int(selection_budget), key_blocks)
    teacher_route = target.topk(route_width, dim=-1).indices
    oracle_mask = torch.zeros_like(target, dtype=torch.bool)
    oracle_mask.scatter_(-1, teacher_route, True)
    oracle_coverage = (target * oracle_mask).sum(dim=-1)
    metrics["oracle_coverage"] = float(
        (oracle_coverage * valid_rows).sum()
        / valid_rows.sum().clamp_min(1)
    )
    metrics["coverage_ratio"] = safe_ratio(
        metrics["coverage"], metrics["oracle_coverage"])

    for budget in budgets:
        width = min(int(budget), key_blocks)
        teacher_top = target.topk(width, dim=-1).indices
        overlap = selected.gather(-1, teacher_top).sum(dim=-1).float()
        enough = (
            query_ids + 1 >= width
        )[None, None].expand(layers, groups, query_blocks)
        rows = valid_rows & enough
        metrics[f"recall@{budget}"] = float(
            ((overlap / width) * rows).sum() / rows.sum().clamp_min(1)
        )

    if 0 <= int(needle_block) < key_blocks:
        needle_block = int(needle_block)
        reader_selected = selected[:, :, -1, :]
        student_hit = reader_selected[..., needle_block]
        metrics["needle_hit"] = float(student_hit.float().mean())
        metrics["needle_union"] = float(
            student_hit.any(dim=1).float().mean())

        teacher_reader = target[:, :, -1, :]
        teacher_top = teacher_reader.topk(route_width, dim=-1).indices
        teacher_hit = (teacher_top == needle_block).any(dim=-1)
        metrics["teacher_needle"] = float(teacher_hit.float().mean())
        metrics["teacher_needle_union"] = float(
            teacher_hit.any(dim=1).float().mean())
        metrics["needle_ratio"] = safe_ratio(
            metrics["needle_hit"], metrics["teacher_needle"])
        metrics["needle_union_ratio"] = safe_ratio(
            metrics["needle_union"], metrics["teacher_needle_union"])
    return metrics


def mean_metrics(rows):
    keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    means = {}
    for key in keys:
        values = [
            float(row[key]) for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            means[key] = sum(values) / len(values)
    return means


def aggregate_cases(cases):
    by_length = {}
    for case in cases:
        requested = str(case["requested_length"])
        by_length.setdefault(requested, []).append(case["metrics"])
    return {
        "cases": len(cases),
        "overall": mean_metrics([case["metrics"] for case in cases]),
        "by_length": {
            length: {
                "cases": len(rows),
                **mean_metrics(rows),
            }
            for length, rows in sorted(
                by_length.items(), key=lambda item: int(item[0]))
        },
    }


def save_plot(report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True)
    metric_specs = (
        (f"recall@{report['config']['k_selected']}", "Teacher block recall"),
        ("coverage_ratio", "Teacher mass / oracle"),
        ("needle_union_ratio", "Evidence preservation / teacher"),
    )
    for label, gate_result in report["gates"].items():
        buckets = gate_result["aggregate"]["by_length"]
        lengths = sorted(int(length) for length in buckets)
        for axis, (metric, title) in zip(axes, metric_specs):
            axis.plot(
                lengths,
                [buckets[str(length)].get(metric, float("nan"))
                 for length in lengths],
                marker="o",
                label=label,
            )
            axis.set(
                title=title,
                xlabel="Requested context tokens",
                ylabel=metric,
                ylim=(0, 1.02),
            )
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gates", nargs="+", required=True,
        help="gate checkpoints as NAME=PATH or bare paths")
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="FULL held-out natural teacher .pt paths or globs")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[8, 10, 16])
    parser.add_argument(
        "--selector-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot")
    args = parser.parse_args()

    if args.beam < 1 or args.radix < 2 or args.k_selected < 1:
        raise SystemExit("beam, radix, and k-selected must be positive")
    if args.local_window < 0:
        raise SystemExit("local-window must be non-negative")
    if args.k_selected < args.local_window + 1:
        raise SystemExit("k-selected must fit the diagonal and local window")
    if any(budget < 1 for budget in args.budgets):
        raise SystemExit("budgets must be positive")

    try:
        gate_specs = parse_gate_specs(args.gates)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    target_paths = expand_paths(args.targets)
    if not target_paths:
        raise SystemExit(f"no targets matched {args.targets}")
    for label, path in gate_specs:
        if not os.path.isfile(path):
            raise SystemExit(f"gate {label!r} not found: {path}")
    for path in target_paths:
        if not os.path.isfile(path):
            raise SystemExit(f"target not found: {path}")

    device = torch.device(args.device)
    gates = {}
    for label, path in gate_specs:
        gate, config = load_gate(path, device)
        gates[label] = {"path": path, "gate": gate, "config": config}

    results = {label: [] for label in gates}
    seen_targets = set()
    evaluated_targets = []
    skipped_duplicates = []
    started = time.perf_counter()
    for index, target_path in enumerate(target_paths, start=1):
        doc = load_teacher(target_path, device=device)
        meta = doc["meta"]
        if (meta.get("prompt_format") != "qwen_chat_v1"
                or not meta.get("dense_screen_accepted")):
            raise SystemExit(
                f"{target_path} is not a dense-accepted chat-formatted "
                "natural held-out target; regenerate it through "
                "screen_natural_prompts.py and --verified-manifest")
        identity = (
            meta.get("case_id"),
            meta["requested_length"],
            meta["seq_len"],
            meta.get("depth"),
            meta["needle_block"],
        )
        if identity in seen_targets:
            skipped_duplicates.append(target_path)
            print(f"skip duplicate target: {target_path}", flush=True)
            del doc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        seen_targets.add(identity)
        evaluated_targets.append(target_path)
        features = {
            "q_feat": doc["q_feat"],
            "k_feat": doc["k_feat"],
            "meta": meta,
        }
        print(
            f"target {index}/{len(target_paths)} "
            f"{os.path.basename(target_path)} "
            f"requested={meta['requested_length']} seq={meta['seq_len']}",
            flush=True,
        )
        for label, entry in gates.items():
            config = entry["config"]
            if (meta["num_layers"] != config["num_layers"]
                    or meta["head_dim"] != config["head_dim"]):
                raise SystemExit(
                    f"{target_path} shape {meta} mismatches gate "
                    f"{label} config {config}")
            selected, timing = live_route(
                entry["gate"],
                config,
                target_path,
                beam=args.beam,
                radix=args.radix,
                k_selected=args.k_selected,
                local_window=args.local_window,
                selector_device=device,
                selector_dtype=args.selector_dtype,
                selector_layer_chunk=args.selector_layer_chunk,
                features=features,
            )
            selected_mask = selected_blocks_to_mask(
                selected, meta["kb"])
            metrics = route_metrics(
                selected_mask,
                doc["target"],
                doc["cmask"],
                tuple(args.budgets),
                meta["needle_block"],
                args.k_selected,
            )
            metrics["selector_seconds"] = timing["selector_seconds"]
            case = {
                "target": target_path,
                "requested_length": meta["requested_length"],
                "seq_len": meta["seq_len"],
                "needle_block": meta["needle_block"],
                "metrics": metrics,
            }
            results[label].append(case)
            print(
                f"  {label}: recall@{args.k_selected}="
                f"{metrics[f'recall@{args.k_selected}']:.3f} "
                f"coverage_ratio={metrics['coverage_ratio']:.3f} "
                f"needle_union_ratio="
                f"{metrics.get('needle_union_ratio', float('nan')):.3f}",
                flush=True,
            )
            del selected, selected_mask
        del doc, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "kind": REPORT_KIND,
        "config": {
            "gates": {label: entry["path"] for label, entry in gates.items()},
            "targets": evaluated_targets,
            "skipped_duplicate_targets": skipped_duplicates,
            "beam": args.beam,
            "radix": args.radix,
            "k_selected": args.k_selected,
            "local_window": args.local_window,
            "budgets": args.budgets,
            "selector_dtype": args.selector_dtype,
            "selector_layer_chunk": args.selector_layer_chunk,
            "device": str(device),
            "evaluation_population": (
                "dense-correct chat-formatted natural heldout prompts"),
        },
        "seconds": time.perf_counter() - started,
        "gates": {
            label: {
                "cases": cases,
                "aggregate": aggregate_cases(cases),
            }
            for label, cases in results.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"natural gate report -> {os.path.abspath(args.out)}")
    if args.plot:
        if save_plot(report, args.plot):
            print(f"natural gate plot   -> {os.path.abspath(args.plot)}")
        else:
            print("matplotlib not installed; skipped natural gate plot")


if __name__ == "__main__":
    main()
