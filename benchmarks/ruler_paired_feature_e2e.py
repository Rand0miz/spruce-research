"""Paired cached-route RULER generation for D=1024 versus D=4096.

This is an accuracy decision run, not an independent-arm latency benchmark.
It reuses selector routes produced by ``ruler_cached_factorial.py`` when they
are available, compiles the two evidence packets, and evaluates both packets
in one greedy GPU batch.  Byte-identical packets are generated only once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_dense_sparse import _free_model, _load_model
from benchmarks.ruler_cached_factorial import (
    DEFAULT_LENGTHS,
    DEFAULT_TASKS,
    extract_query,
    score_for,
)
from configs.long_context import configure_tokenizer, context_limit
from interfaces.evidence_compiler import (
    compile_evidence_packet_from_layout,
    locate_prompt_layout,
)
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


ARM_DIMS = (1024, 4096)


def arm_name(feature_dim: int) -> str:
    return f"d{int(feature_dim)}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def config_fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def generated_path(data_dir: Path, task: str, length: int) -> Path:
    return data_dir / str(int(length)) / task / "validation.jsonl"


def read_samples(
        data_dir: Path, task: str, length: int, num_samples: int) -> list[dict]:
    path = generated_path(data_dir, task, length)
    if not path.is_file():
        raise FileNotFoundError(f"missing generated RULER data: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:int(num_samples)]


def deduplicate_prompts(prompts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Return stable unique prompts and an index for each original prompt."""
    unique: list[str] = []
    positions: dict[str, int] = {}
    mapping: list[int] = []
    for prompt in prompts:
        if prompt not in positions:
            positions[prompt] = len(unique)
            unique.append(prompt)
        mapping.append(positions[prompt])
    return unique, mapping


def validate_route_config(report: dict, args: argparse.Namespace) -> None:
    config = report.get("config", {})
    required = {
        "model": args.model,
        "beam": args.beam,
        "block_size": args.block_size,
        "block_radius": args.block_radius,
        "boundary": args.boundary,
        "unigram_fraction": args.unigram_fraction,
        "idf_power": args.idf_power,
        "radix": args.radix,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    for key, expected in (
        ("tails", args.query_tail_lines),
        ("feature_dims", ARM_DIMS[0]),
        ("feature_dims", ARM_DIMS[1]),
        ("budgets", args.candidate_blocks),
    ):
        if expected not in config.get(key, []):
            mismatches[f"{key}:{expected}"] = (config.get(key), expected)
    if mismatches:
        raise ValueError(f"factorial route-cache config mismatch: {mismatches}")


def load_route_report(
        route_cache_dir: Path | None, task: str, length: int,
        args: argparse.Namespace) -> dict | None:
    if route_cache_dir is None:
        return None
    path = route_cache_dir / f"{task}_{int(length)}.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"route cache is not complete: {path}")
    validate_route_config(report, args)
    return report


def cached_selections(
        report: dict | None, position: int, references: Sequence[str]) -> dict[int, list[int]] | None:
    if report is None:
        return None
    cases = report.get("cases", [])
    if position >= len(cases):
        raise IndexError(f"route cache has no case at position {position}")
    case = cases[position]
    if [str(value) for value in case.get("references", [])] != list(references):
        raise ValueError(f"route-cache reference mismatch at position {position}")
    selected = {}
    for feature_dim in ARM_DIMS:
        key = f"t3_d{feature_dim}_m9"
        if key not in case.get("arms", {}):
            raise KeyError(f"route cache is missing arm {key}")
        selected[feature_dim] = [
            int(block) for block in case["arms"][key]["selected_blocks"]
        ]
    return selected


def compile_pair(
        tokenizer, sample: dict, position: int, route_report: dict | None,
        args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    prompt = str(sample["input"])
    references = [str(value) for value in sample.get("outputs", [])]
    question = extract_query(prompt, args.query_tail_lines)

    started = time.perf_counter()
    layout = locate_prompt_layout(tokenizer, prompt, prompt, question)
    layout_seconds = time.perf_counter() - started

    selections = cached_selections(route_report, position, references)
    route_source = "factorial_cache" if selections is not None else "live"
    arms: dict[str, dict] = {}
    prompts: dict[str, str] = {}
    for feature_dim in ARM_DIMS:
        index_seconds = 0.0
        selection_seconds = 0.0
        if selections is None:
            mark = time.perf_counter()
            index = build_pre_qwen_index(
                layout, args.block_size, feature_dim=feature_dim,
                unigram_fraction=args.unigram_fraction, radix=args.radix)
            index_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            weights = question_feature_weights(
                tokenizer, question, index, idf_power=args.idf_power)
            route = select_pre_qwen_blocks(
                index, weights, top_m=args.candidate_blocks,
                beam=args.beam, radix=args.radix)
            selection_seconds = time.perf_counter() - mark
            blocks = [int(block) for block in route.blocks]
        else:
            blocks = selections[feature_dim]

        mark = time.perf_counter()
        packet = compile_evidence_packet_from_layout(
            tokenizer, prompt, layout, question, blocks, args.block_size,
            block_radius=args.block_radius, boundary=args.boundary)
        compile_seconds = time.perf_counter() - mark
        key = arm_name(feature_dim)
        prompts[key] = packet.prompt
        arms[key] = {
            "feature_dim": feature_dim,
            "route_source": route_source,
            "selected_blocks": list(packet.selected_blocks),
            "expanded_blocks": list(packet.expanded_blocks),
            "span_count": len(packet.spans),
            "input_tokens": int(packet.compiled_prompt_tokens),
            "original_input_tokens": int(packet.original_prompt_tokens),
            "compression_fraction": float(packet.compression_fraction),
            "evidence_score": float(score_for(
                str(sample.get("task", "")) or args.current_task,
                packet.prompt, references)),
            "index_seconds": float(index_seconds),
            "selection_seconds": float(selection_seconds),
            "compile_seconds": float(compile_seconds),
        }

    metadata = {
        "index": int(position),
        "source_index": sample.get("index", position),
        "task": args.current_task,
        "length": int(args.current_length),
        "references": references,
        "query_chars": len(question),
        "layout_seconds": float(layout_seconds),
        "original_prompt_tokens": len(layout.input_ids),
        "route_source": route_source,
        "identical_packet": prompts[arm_name(1024)] == prompts[arm_name(4096)],
        "arms": arms,
    }
    return metadata, prompts


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_unique_prompts(
        model, tokenizer, prompts: Sequence[str], max_new_tokens: int) -> tuple[list[str], dict]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    started = time.perf_counter()
    mark = time.perf_counter()
    encoded = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        add_special_tokens=True)
    tokenize_seconds = time.perf_counter() - mark
    input_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
    device = next(model.parameters()).device
    mark = time.perf_counter()
    inputs = {key: value.to(device) for key, value in encoded.items()}
    _sync()
    transfer_seconds = time.perf_counter() - mark
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mark = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    _sync()
    model_seconds = time.perf_counter() - mark
    prompt_width = int(inputs["input_ids"].shape[1])
    generated = output[:, prompt_width:].detach().cpu()
    answers = [
        tokenizer.decode(row, skip_special_tokens=True) for row in generated
    ]
    generated_tokens = [int(row.shape[0]) for row in generated]
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda" else 0.0
    )
    return answers, {
        "unique_prompts": len(prompts),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "tokenize_seconds": float(tokenize_seconds),
        "transfer_seconds": float(transfer_seconds),
        "model_seconds": float(model_seconds),
        "batch_seconds": float(time.perf_counter() - started),
        "peak_memory_allocated_gb": float(peak_gb),
        "latency_scope": "shared paired batch; not independent per-arm latency",
    }


def score_generated_pair(
        case: dict, prompts: dict[str, str], answers: Sequence[str],
        mapping: Sequence[int], task: str) -> dict:
    references = case["references"]
    for arm_position, feature_dim in enumerate(ARM_DIMS):
        key = arm_name(feature_dim)
        answer = answers[mapping[arm_position]]
        score = float(score_for(task, answer, references))
        case["arms"][key].update({
            "answer": answer,
            "score": score,
            "perfect": bool(math.isclose(score, 1.0)),
        })
    case["score_delta_d4096_minus_d1024"] = (
        case["arms"]["d4096"]["score"] - case["arms"]["d1024"]["score"]
    )
    return case


def group(rows: Sequence[dict], key_fn) -> dict:
    result: dict = {}
    for row in rows:
        result.setdefault(key_fn(row), []).append(row)
    return result


def paired_bootstrap_ci(deltas: Sequence[float], repeats: int = 10000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    import numpy as np
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(20260802)
    means = np.empty(repeats, dtype=np.float64)
    chunk = 500
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def aggregate_block(rows: Sequence[dict]) -> dict:
    deltas = [float(row["score_delta_d4096_minus_d1024"]) for row in rows]
    scores = {
        key: [float(row["arms"][key]["score"]) for row in rows]
        for key in ("d1024", "d4096")
    }
    return {
        "samples": len(rows),
        "score_d1024": statistics.fmean(scores["d1024"]),
        "score_d4096": statistics.fmean(scores["d4096"]),
        "delta_d4096_minus_d1024": statistics.fmean(deltas),
        "delta_bootstrap_95ci": paired_bootstrap_ci(deltas),
        "d4096_wins": sum(delta > 1e-12 for delta in deltas),
        "d1024_wins": sum(delta < -1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "perfect_d1024": statistics.fmean(
            float(row["arms"]["d1024"]["perfect"]) for row in rows),
        "perfect_d4096": statistics.fmean(
            float(row["arms"]["d4096"]["perfect"]) for row in rows),
        "evidence_d1024": statistics.fmean(
            float(row["arms"]["d1024"]["evidence_score"]) for row in rows),
        "evidence_d4096": statistics.fmean(
            float(row["arms"]["d4096"]["evidence_score"]) for row in rows),
        "median_tokens_d1024": statistics.median(
            int(row["arms"]["d1024"]["input_tokens"]) for row in rows),
        "median_tokens_d4096": statistics.median(
            int(row["arms"]["d4096"]["input_tokens"]) for row in rows),
        "median_compression_d1024": statistics.median(
            float(row["arms"]["d1024"]["compression_fraction"]) for row in rows),
        "median_compression_d4096": statistics.median(
            float(row["arms"]["d4096"]["compression_fraction"]) for row in rows),
        "identical_packet_fraction": statistics.fmean(
            float(row["identical_packet"]) for row in rows),
        "median_paired_batch_seconds": statistics.median(
            float(row["paired_batch"]["batch_seconds"]) for row in rows),
    }


def collect_reports(
        task_dir: Path, tasks: Sequence[str], lengths: Sequence[int]) -> tuple[list[dict], list[dict], set[tuple[str, int]]]:
    rows: list[dict] = []
    errors: list[dict] = []
    completed: set[tuple[str, int]] = set()
    for path in sorted(task_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        pair = (report.get("task"), int(report.get("length", -1)))
        if pair[0] not in tasks or pair[1] not in lengths:
            continue
        rows.extend(report.get("cases", []))
        errors.extend(report.get("errors", []))
        if report.get("status") == "completed":
            completed.add(pair)
    return rows, errors, completed


def make_figures(summary: dict, figure_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def save(fig, name: str) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(
                figure_dir / f"{name}.{suffix}",
                dpi=200 if suffix == "png" else None,
                bbox_inches="tight")
        plt.close(fig)
        made.append(f"{name}.png")

    overall = summary["overall"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([0, 1], [overall["score_d1024"], overall["score_d4096"]],
           color=["#4472c4", "#d65f5f"])
    ax.set_xticks([0, 1], ["D=1024", "D=4096"])
    ax.set(ylabel="mean RULER score", ylim=(0, 1),
           title="Paired end-to-end RULER accuracy")
    ax.grid(alpha=0.25, axis="y")
    save(fig, "01_overall_accuracy")

    lengths = sorted(int(value) for value in summary["by_length"])
    x = [value // 1024 for value in lengths]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, color in (("d1024", "#4472c4"), ("d4096", "#d65f5f")):
        ax.plot(x, [summary["by_length"][str(length)][f"score_{key}"]
                    for length in lengths], marker="o", linewidth=2,
                color=color, label=key.upper())
    ax.set(xlabel="context length (Ki tokens)", ylabel="mean RULER score",
           ylim=(0, 1), title="Accuracy by context length")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "02_accuracy_by_length")

    tasks = list(summary["by_task"])
    deltas = [summary["by_task"][task]["delta_d4096_minus_d1024"]
              for task in tasks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(tasks, deltas,
            color=["#2a9d8f" if value >= 0 else "#d65f5f" for value in deltas])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="score delta: D4096 - D1024",
           title="Paired feature-width effect by task")
    ax.grid(alpha=0.25, axis="x")
    save(fig, "03_delta_by_task")

    wins = [summary["by_length"][str(length)]["d4096_wins"] for length in lengths]
    losses = [summary["by_length"][str(length)]["d1024_wins"] for length in lengths]
    ties = [summary["by_length"][str(length)]["ties"] for length in lengths]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, wins, width=6, label="D4096 wins", color="#2a9d8f")
    ax.bar(x, ties, width=6, bottom=wins, label="ties", color="#bbbbbb")
    ax.bar(x, losses, width=6,
           bottom=[a + b for a, b in zip(wins, ties)],
           label="D1024 wins", color="#d65f5f")
    ax.set(xlabel="context length (Ki tokens)", ylabel="paired samples",
           title="Per-sample paired outcomes")
    ax.legend(fontsize=8)
    save(fig, "04_paired_outcomes_by_length")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, [summary["by_length"][str(length)]["evidence_d1024"]
                for length in lengths], marker="o", label="D1024")
    ax.plot(x, [summary["by_length"][str(length)]["evidence_d4096"]
                for length in lengths], marker="o", label="D4096")
    ax.set(xlabel="context length (Ki tokens)", ylabel="packet evidence score",
           ylim=(0, 1), title="Evidence available to the model")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "05_evidence_by_length")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, [summary["by_length"][str(length)]["median_tokens_d1024"]
                for length in lengths], marker="o", label="D1024")
    ax.plot(x, [summary["by_length"][str(length)]["median_tokens_d4096"]
                for length in lengths], marker="o", label="D4096")
    ax.set(xlabel="context length (Ki tokens)", ylabel="median packet tokens",
           title="Compiled packet size")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "06_packet_tokens_by_length")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, [summary["by_length"][str(length)]["identical_packet_fraction"]
                for length in lengths], marker="o", color="#6a4c93")
    ax.set(xlabel="context length (Ki tokens)", ylabel="identical packet fraction",
           ylim=(0, 1), title="Exact generation reuse across D arms")
    ax.grid(alpha=0.25)
    save(fig, "07_identical_packet_cache")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, [summary["by_length"][str(length)]["median_paired_batch_seconds"]
                for length in lengths], marker="o", color="#444444")
    ax.set(xlabel="context length (Ki tokens)", ylabel="median paired batch seconds",
           yscale="log", title="Cached-route paired L4 throughput")
    ax.grid(alpha=0.25, which="both")
    save(fig, "08_paired_batch_time")
    return made


def write_summary_markdown(summary: dict, path: Path) -> None:
    overall = summary["overall"]
    low, high = overall["delta_bootstrap_95ci"]
    lines = [
        "# Paired RULER end-to-end: D=1024 vs D=4096",
        "",
        f"- generated: {summary['created_utc']}",
        f"- GPU: {summary['gpu_name']}",
        f"- complete: {summary['run_complete']}",
        f"- samples: {overall['samples']}",
        "- fixed configuration: tail=3, M=9, beam=16, B=64",
        "- routes: cached factorial routes where available",
        "",
        "## Overall",
        "",
        f"- D=1024: {overall['score_d1024']:.4f}",
        f"- D=4096: {overall['score_d4096']:.4f}",
        f"- paired delta: {overall['delta_d4096_minus_d1024']:+.4f} "
        f"(paired bootstrap 95% CI {low:+.4f} to {high:+.4f})",
        f"- wins/ties/losses for D=4096: {overall['d4096_wins']} / "
        f"{overall['ties']} / {overall['d1024_wins']}",
        "",
        "## By length",
        "",
        "| length | D1024 | D4096 | delta | wins/ties/losses | n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length, block in summary["by_length"].items():
        lines.append(
            f"| {length} | {block['score_d1024']:.3f} | "
            f"{block['score_d4096']:.3f} | "
            f"{block['delta_d4096_minus_d1024']:+.3f} | "
            f"{block['d4096_wins']}/{block['ties']}/{block['d1024_wins']} | "
            f"{block['samples']} |")
    lines += [
        "",
        "## By task",
        "",
        "| task | D1024 | D4096 | delta | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, block in summary["by_task"].items():
        lines.append(
            f"| {task} | {block['score_d1024']:.3f} | "
            f"{block['score_d4096']:.3f} | "
            f"{block['delta_d4096_minus_d1024']:+.3f} | "
            f"{block['samples']} |")
    if not summary["run_complete"]:
        lines += [
            "",
            f"> PARTIAL: {summary['completed_pairs']}/{summary['expected_pairs']} "
            "pairs complete. Do not quote as the final result.",
        ]
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Both arms use tail=3 and M=9; only feature width changes.",
        "- Cached routes make this an accuracy/throughput run, not cold-request latency.",
        "- Non-identical arms are generated together; paired batch time is not per-arm latency.",
        "- RULER data generation is NVIDIA's; the serving loop is SPRUCE's.",
        "- Record complete results in LOG.md before quoting them elsewhere.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-cache-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--query-tail-lines", type=int, default=3)
    parser.add_argument("--candidate-blocks", type=int, default=9)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument("--boundary", default="paragraph")
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument("--original-context", type=int, default=32768)
    parser.add_argument("--session-budget-minutes", type=float, default=300.0)
    parser.add_argument("--archive-sha256", default="")
    parser.add_argument("--require-gpu-substring", default="L4")
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if tuple(sorted(ARM_DIMS)) != ARM_DIMS:
        raise AssertionError("ARM_DIMS must be stable and sorted")
    if args.query_tail_lines != 3 or args.candidate_blocks != 9:
        raise SystemExit("decision test is frozen at tail=3 and M=9")
    if args.num_samples < 1 or args.max_new_tokens < 1:
        raise SystemExit("num-samples and max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the paired end-to-end run")
    properties = torch.cuda.get_device_properties(0)
    gpu_name = properties.name
    if args.require_gpu_substring.lower() not in gpu_name.lower():
        raise SystemExit(
            f"GPU {gpu_name!r} does not contain required label "
            f"{args.require_gpu_substring!r}; select an NVIDIA L4 runtime")

    config = {
        "model": args.model,
        "tasks": list(args.tasks),
        "lengths": [int(value) for value in args.lengths],
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "query_tail_lines": args.query_tail_lines,
        "feature_dims": list(ARM_DIMS),
        "candidate_blocks": args.candidate_blocks,
        "beam": args.beam,
        "block_size": args.block_size,
        "block_radius": args.block_radius,
        "boundary": args.boundary,
        "unigram_fraction": args.unigram_fraction,
        "idf_power": args.idf_power,
        "radix": args.radix,
        "yarn_factor": args.yarn_factor,
        "original_context": args.original_context,
        "generation": "greedy paired batch; identical packets generated once",
    }
    fingerprint = config_fingerprint(config)
    output_dir = args.output_dir
    task_dir = output_dir / "task_reports"
    figure_dir = output_dir / "figures"
    task_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    samples_by_pair: dict[tuple[str, int], list[dict]] = {}
    for length in args.lengths:
        for task in args.tasks:
            samples_by_pair[(task, int(length))] = read_samples(
                args.data_dir, task, int(length), args.num_samples)
    expected_pairs = set(samples_by_pair)

    pending = False
    for (task, length), samples in samples_by_pair.items():
        path = task_dir / f"{task}_{length}.json"
        if not path.is_file():
            pending = True
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("config_fingerprint") != fingerprint:
            raise SystemExit(f"refusing mixed resume config in {path}")
        done = {int(case["index"]) for case in report.get("cases", [])}
        errors = {int(error["index"]) for error in report.get("errors", [])}
        if args.rerun_errors:
            errors.clear()
        if len(done | errors) < len(samples):
            pending = True

    session_started = time.perf_counter()
    model = None
    tokenizer = None
    try:
        if pending:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            configure_tokenizer(
                tokenizer, yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_context)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            maximum = context_limit(
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_context)
            print(f"GPU: {gpu_name} | VRAM {properties.total_memory / 2**30:.1f} GiB")
            print(f"loading {args.model} once; configured context {maximum}")
            model = _load_model(
                args.model, "sdpa", torch.float16,
                str(Path(tempfile.gettempdir()) / "spruce_ruler_paired_offload"),
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_context)
            generate_unique_prompts(
                model, tokenizer, ["Answer briefly."], max_new_tokens=1)

        budget_hit = False
        for length in args.lengths:
            if budget_hit:
                break
            print(f"=== {int(length)} tokens ===", flush=True)
            for task in args.tasks:
                if budget_hit:
                    break
                pair = (task, int(length))
                samples = samples_by_pair[pair]
                path = task_dir / f"{task}_{int(length)}.json"
                if path.is_file():
                    report = json.loads(path.read_text(encoding="utf-8"))
                else:
                    report = {
                        "kind": "spruce_ruler_paired_feature_e2e_task_v1",
                        "task": task,
                        "length": int(length),
                        "gpu_name": gpu_name,
                        "config": config,
                        "config_fingerprint": fingerprint,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "cases": [],
                        "errors": [],
                        "status": "running",
                    }
                if report.get("config_fingerprint") != fingerprint:
                    raise SystemExit(f"refusing mixed resume config in {path}")
                if args.rerun_errors and report.get("errors"):
                    report["errors"] = []
                done = {int(case["index"]) for case in report.get("cases", [])}
                failed = {int(error["index"]) for error in report.get("errors", [])}
                route_report = load_route_report(
                    args.route_cache_dir, task, int(length), args)
                for position, sample in enumerate(samples):
                    if position in done or position in failed:
                        continue
                    if args.session_budget_minutes and (
                            time.perf_counter() - session_started
                            >= args.session_budget_minutes * 60):
                        budget_hit = True
                        break
                    args.current_task = task
                    args.current_length = int(length)
                    try:
                        case, prompt_by_arm = compile_pair(
                            tokenizer, sample, position, route_report, args)
                        ordered = [prompt_by_arm[arm_name(value)] for value in ARM_DIMS]
                        unique, mapping = deduplicate_prompts(ordered)
                        answers, batch = generate_unique_prompts(
                            model, tokenizer, unique, args.max_new_tokens)
                        case["paired_batch"] = batch
                        case = score_generated_pair(
                            case, prompt_by_arm, answers, mapping, task)
                        report["cases"].append(case)
                        done.add(position)
                        atomic_json(path, report)
                        print(
                            "  %-18s #%02d D1024 %.2f D4096 %.2f delta %+.2f %s"
                            % (task, position,
                               case["arms"]["d1024"]["score"],
                               case["arms"]["d4096"]["score"],
                               case["score_delta_d4096_minus_d1024"],
                               "shared" if case["identical_packet"] else "paired"),
                            flush=True)
                    except Exception as error:  # noqa: BLE001
                        report["errors"].append({
                            "index": int(position),
                            "source_index": sample.get("index", position),
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        })
                        atomic_json(path, report)
                        print(f"  ERROR {task} {length} #{position}: {error!r}")
                        if isinstance(error, torch.cuda.OutOfMemoryError):
                            gc.collect()
                            torch.cuda.empty_cache()
                attempted = done | {
                    int(error["index"]) for error in report.get("errors", [])
                }
                if len(attempted) == len(samples) and not report.get("errors"):
                    report["status"] = "completed"
                    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
                elif budget_hit:
                    report["status"] = "running"
                else:
                    report["status"] = "completed_with_errors"
                atomic_json(path, report)
                if report["cases"]:
                    block = aggregate_block(report["cases"])
                    print(
                        "  %-18s aggregate %.3f -> %.3f (%+.3f), n=%d"
                        % (task, block["score_d1024"], block["score_d4096"],
                           block["delta_d4096_minus_d1024"], block["samples"]),
                        flush=True)
    finally:
        if model is not None:
            model = _free_model(model)

    rows, errors, completed_pairs = collect_reports(
        task_dir, args.tasks, args.lengths)
    if not rows:
        raise SystemExit("no completed samples exist to aggregate")
    run_complete = completed_pairs >= expected_pairs and not errors
    by_task = {
        str(key): aggregate_block(value)
        for key, value in sorted(group(rows, lambda row: row["task"]).items())
    }
    by_length = {
        str(key): aggregate_block(value)
        for key, value in sorted(group(rows, lambda row: int(row["length"])).items())
    }
    summary = {
        "kind": "spruce_ruler_paired_feature_e2e_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": args.archive_sha256,
        "gpu_name": gpu_name,
        "gpu_vram_gib": properties.total_memory / 2**30,
        "config": config,
        "config_fingerprint": fingerprint,
        "run_complete": bool(run_complete),
        "completed_pairs": len(completed_pairs),
        "expected_pairs": len(expected_pairs),
        "sample_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "session_wall_clock_seconds": time.perf_counter() - session_started,
        "overall": aggregate_block(rows),
        "by_task": by_task,
        "by_length": by_length,
    }
    summary["figures"] = make_figures(summary, figure_dir)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "run_record.json", {
        key: summary[key] for key in (
            "kind", "created_utc", "archive_sha256", "gpu_name",
            "config", "config_fingerprint", "run_complete",
            "completed_pairs", "expected_pairs", "sample_count",
            "error_count", "session_wall_clock_seconds", "overall",
            "by_task", "by_length")
    })
    write_summary_markdown(summary, output_dir / "SUMMARY.md")
    overall = summary["overall"]
    print(
        "OVERALL D1024 %.4f D4096 %.4f delta %+.4f, wins/ties/losses "
        "%d/%d/%d, pairs %d/%d, errors %d"
        % (overall["score_d1024"], overall["score_d4096"],
           overall["delta_d4096_minus_d1024"], overall["d4096_wins"],
           overall["ties"], overall["d1024_wins"], len(completed_pairs),
           len(expected_pairs), len(errors)))
    if not run_complete:
        print("PARTIAL RUN: rerun the notebook to resume; do not quote yet.")


if __name__ == "__main__":
    main()
