"""Held-out paper RULER accuracy: dense Qwen versus SPRUCE compilation.

The dataset must be generated separately with NVIDIA RULER and accompanied by
a frozen SHA-256 manifest.  This runner never reuses development routes: it
builds the validation-selected D=4096/beam=64/M=32 selector live, then compares
generation from the complete official input (including ``answer_prefix``)
against the compiled packet.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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


TASK_MAX_NEW_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}

PAPER_CANDIDATE = {
    "feature_dim": 4096,
    "candidate_blocks": 32,
    "beam": 64,
    "block_size": 64,
    "block_radius": 1,
    "boundary": "paragraph",
    "unigram_fraction": 0.5,
    "idf_power": 2.0,
    "radix": 2,
}
PROMPT_BOUNDARY_PARSER = "nvidia_ruler_legacy_templates_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def config_fingerprint(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dataset_path(data_dir: Path, task: str, length: int, subset: str) -> Path:
    return data_dir / str(int(length)) / task / f"{subset}.jsonl"


def verify_dataset_manifest(
        data_dir: Path, manifest_path: Path, tasks: Sequence[str],
        lengths: Sequence[int], subset: str, num_samples: int) -> tuple[dict, str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("subset")) != str(subset):
        raise ValueError("dataset manifest subset does not match the run")
    if int(manifest.get("num_samples", -1)) != int(num_samples):
        raise ValueError("dataset manifest sample count does not match the run")
    if set(manifest.get("tasks", [])) != set(tasks):
        raise ValueError("dataset manifest task list does not match the run")
    if {int(value) for value in manifest.get("lengths", [])} != {
            int(value) for value in lengths}:
        raise ValueError("dataset manifest length list does not match the run")
    expected = {
        f"{int(length)}/{task}/{subset}.jsonl"
        for length in lengths for task in tasks
    }
    entries = manifest.get("files", {})
    if set(entries) != expected:
        raise ValueError(
            f"dataset manifest grid mismatch: expected {len(expected)}, "
            f"found {len(entries)}")
    for relative in sorted(expected):
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest dataset file is missing: {path}")
        lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip())
        if lines != int(num_samples):
            raise ValueError(
                f"{relative} has {lines} rows, expected exactly {num_samples}")
        observed = file_sha256(path)
        recorded = str(entries[relative]["sha256"]).upper()
        if observed != recorded:
            raise ValueError(
                f"dataset hash mismatch for {relative}: {observed} != {recorded}")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(canonical).hexdigest().upper()


def read_samples(
        data_dir: Path, task: str, length: int, subset: str) -> list[dict]:
    path = dataset_path(data_dir, task, length, subset)
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _required_suffix(text: str, marker: str, task: str, role: str) -> str:
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(
            f"official {task} template is missing its {role} marker {marker!r}")
    suffix = text[start:]
    if not suffix.strip() or suffix == text:
        raise ValueError(f"official {task} {role} boundary leaves no document")
    return suffix


def official_prompt_parts(sample: dict, task: str) -> tuple[str, str, str]:
    """Return full input, lexical query, and exact document-layout suffix.

    These boundaries follow the NVIDIA RULER legacy synthetic templates rather
    than a trailing-line heuristic.  CWe and FWe serialize their whole context
    on one line, so a generic last-N-lines rule can accidentally classify the
    source data as part of the question and defeat the compression comparison.
    Unknown or changed templates fail closed instead of silently leaking data.
    """
    base = str(sample["input"])
    prefix = str(sample.get("answer_prefix", ""))
    if task.startswith("niah_"):
        selector_query = _required_suffix(base, "\nWhat ", task, "query")
        reader_suffix = selector_query
    elif task in {"vt", "cwe", "fwe"}:
        selector_query = _required_suffix(base, "\nQuestion:", task, "query")
        reader_suffix = selector_query
    elif task in {"qa_1", "qa_2"}:
        selector_query = _required_suffix(base, "\nQuestion:", task, "query")
        reader_suffix = _required_suffix(
            base, "\n\nAnswer the question based on the given documents.",
            task, "reader-instruction")
        if not reader_suffix.endswith(selector_query):
            raise ValueError(f"official {task} query is outside its reader suffix")
    else:
        raise ValueError(f"unsupported RULER task: {task!r}")
    full_prompt = base + prefix
    layout_suffix = reader_suffix + prefix
    if not full_prompt.endswith(layout_suffix):
        raise ValueError(
            "official RULER prompt does not end with reader suffix + answer_prefix")
    return full_prompt, selector_query, layout_suffix


def append_official_answer_prefix(compiled_chat_prompt: str, sample: dict) -> str:
    """Place RULER's prefix immediately before the model's generated text."""
    return compiled_chat_prompt + str(sample.get("answer_prefix", ""))


def compile_spruce_prompt(
        tokenizer, sample: dict, task: str,
        args: argparse.Namespace) -> tuple[str, dict]:
    full_prompt, selector_query, layout_suffix = official_prompt_parts(
        sample, task)
    started = time.perf_counter()
    layout = locate_prompt_layout(
        tokenizer, full_prompt, full_prompt, layout_suffix)
    layout_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index = build_pre_qwen_index(
        layout, args.block_size, feature_dim=args.feature_dim,
        unigram_fraction=args.unigram_fraction, radix=args.radix)
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    weights = question_feature_weights(
        tokenizer, selector_query, index, idf_power=args.idf_power)
    selection = select_pre_qwen_blocks(
        index, weights, top_m=args.candidate_blocks,
        beam=args.beam, radix=args.radix)
    selection_seconds = time.perf_counter() - started
    started = time.perf_counter()
    packet = compile_evidence_packet_from_layout(
        tokenizer, full_prompt, layout, selector_query, selection.blocks,
        args.block_size, block_radius=args.block_radius,
        boundary=args.boundary)
    compile_seconds = time.perf_counter() - started
    # RULER's caller places answer_prefix immediately before generated text.
    # The evidence compiler creates a chat generation prompt, so append the
    # prefix after that wrapper rather than burying it inside the user message.
    compiled_prompt = append_official_answer_prefix(packet.prompt, sample)
    compiled_tokens = len(tokenizer(
        compiled_prompt, add_special_tokens=True)["input_ids"])
    return compiled_prompt, {
        "selected_blocks": [int(value) for value in selection.blocks],
        "expanded_blocks": [int(value) for value in packet.expanded_blocks],
        "visited_nodes": int(selection.visited_nodes),
        "original_input_tokens": int(packet.original_prompt_tokens),
        "input_tokens": int(compiled_tokens),
        "compression_fraction": float(
            compiled_tokens / max(1, packet.original_prompt_tokens)),
        "layout_seconds": float(layout_seconds),
        "index_seconds": float(index_seconds),
        "selection_seconds": float(selection_seconds),
        "compile_seconds": float(compile_seconds),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FirstTokenTimer:
    """Record synchronized time to the first post-prefill token scores."""

    def __init__(self) -> None:
        self.started: float | None = None
        self.first_token_seconds: float | None = None

    def start(self) -> None:
        self.started = time.perf_counter()

    def __call__(self, input_ids, scores):
        if self.first_token_seconds is None:
            _sync()
            if self.started is None:
                raise RuntimeError("first-token timer was not started")
            self.first_token_seconds = time.perf_counter() - self.started
        return scores


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    started = time.perf_counter()
    encoded = tokenizer(prompt, return_tensors="pt")
    tokenize_seconds = time.perf_counter() - started
    input_tokens = int(encoded["input_ids"].shape[1])
    device = next(model.parameters()).device
    transfer_started = time.perf_counter()
    inputs = {key: value.to(device) for key, value in encoded.items()}
    _sync()
    transfer_seconds = time.perf_counter() - transfer_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_token_timer = FirstTokenTimer()
    model_started = time.perf_counter()
    first_token_timer.start()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[first_token_timer],
        )
    _sync()
    model_seconds = time.perf_counter() - model_started
    generated = output[0, inputs["input_ids"].shape[1]:].detach().cpu()
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tokens = int(generated.shape[0])
    ttft_seconds = float(first_token_timer.first_token_seconds or model_seconds)
    decode_seconds = max(0.0, float(model_seconds - ttft_seconds))
    decode_tokens = max(0, generated_tokens - 1)
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda" else 0.0)
    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "tokenize_seconds": float(tokenize_seconds),
        "transfer_seconds": float(transfer_seconds),
        "model_seconds": float(model_seconds),
        "ttft_seconds": ttft_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": float(
            decode_tokens / decode_seconds if decode_seconds > 0 else 0.0),
        "request_seconds": float(time.perf_counter() - started),
        "peak_memory_allocated_gb": float(peak_gb),
    }


def run_case(
        model, tokenizer, sample: dict, task: str, args: argparse.Namespace,
        generation_order: str = "dense_first") -> dict:
    references = [str(value) for value in sample.get("outputs", [])]
    dense_prompt, selector_query, layout_suffix = official_prompt_parts(sample, task)
    spruce_prompt, compiler = compile_spruce_prompt(
        tokenizer, sample, task, args)
    limit = TASK_MAX_NEW_TOKENS[task]
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_context)
    dense_preflight_started = time.perf_counter()
    dense_tokens = len(tokenizer(
        dense_prompt, add_special_tokens=True)["input_ids"])
    dense_preflight_seconds = time.perf_counter() - dense_preflight_started
    if dense_tokens + limit > maximum_context:
        raise ValueError(
            f"dense request {dense_tokens}+{limit} exceeds context {maximum_context}")

    if generation_order == "dense_first":
        dense = generate_one(model, tokenizer, dense_prompt, limit)
        spruce = generate_one(model, tokenizer, spruce_prompt, limit)
    elif generation_order == "spruce_first":
        spruce = generate_one(model, tokenizer, spruce_prompt, limit)
        dense = generate_one(model, tokenizer, dense_prompt, limit)
    else:
        raise ValueError(f"unknown generation order: {generation_order}")
    dense_score = float(score_for(task, dense["answer"], references))
    spruce_score = float(score_for(task, spruce["answer"], references))
    evidence_score = float(score_for(task, spruce_prompt, references))
    spruce.update(compiler)
    compiler_seconds = sum(float(compiler[name]) for name in (
        "layout_seconds", "index_seconds", "selection_seconds",
        "compile_seconds"))
    dense["preflight_tokenize_seconds"] = float(dense_preflight_seconds)
    dense["fully_charged_request_seconds"] = float(
        dense_preflight_seconds + dense["request_seconds"])
    dense["fully_charged_ttft_seconds"] = float(
        dense_preflight_seconds + dense["ttft_seconds"])
    spruce["compiler_seconds"] = compiler_seconds
    spruce["fully_charged_request_seconds"] = float(
        compiler_seconds + spruce["request_seconds"])
    spruce["fully_charged_ttft_seconds"] = float(
        compiler_seconds + spruce["ttft_seconds"])
    spruce["evidence_score"] = evidence_score
    spruce["score"] = spruce_score
    dense["score"] = dense_score
    return {
        "references": references,
        "answer_prefix": str(sample.get("answer_prefix", "")),
        "selector_query_chars": len(selector_query),
        "layout_suffix_chars": len(layout_suffix),
        "generation_order": generation_order,
        "max_new_tokens": limit,
        "dense": dense,
        "spruce": spruce,
        "score_delta_spruce_minus_dense": spruce_score - dense_score,
    }


def grouped(rows: Sequence[dict], key_fn) -> dict:
    result: dict = defaultdict(list)
    for row in rows:
        result[key_fn(row)].append(row)
    return dict(result)


def stratified_bootstrap(
        rows: Sequence[dict], value_fn, repeats: int = 10000) -> list[float]:
    import numpy as np
    strata = list(grouped(rows, lambda row: row["task"]).values())
    rng = np.random.default_rng(314159)
    estimates = np.zeros(repeats, dtype=np.float64)
    for stratum in strata:
        values = np.asarray([value_fn(row) for row in stratum], dtype=np.float64)
        indices = rng.integers(
            0, len(values), size=(int(repeats), len(values)))
        estimates += values[indices].mean(axis=1) / len(strata)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def macro_block(rows: Sequence[dict]) -> dict:
    by_task = grouped(rows, lambda row: row["task"])
    dense_task = [statistics.fmean(r["dense"]["score"] for r in values)
                  for values in by_task.values()]
    spruce_task = [statistics.fmean(r["spruce"]["score"] for r in values)
                   for values in by_task.values()]
    deltas = [row["score_delta_spruce_minus_dense"] for row in rows]
    dense_request = [row["dense"]["fully_charged_request_seconds"] for row in rows]
    spruce_request = [
        row["spruce"]["fully_charged_request_seconds"] for row in rows]
    dense_ttft = [row["dense"]["fully_charged_ttft_seconds"] for row in rows]
    spruce_ttft = [row["spruce"]["fully_charged_ttft_seconds"] for row in rows]
    compiler_components = {
        name: [row["spruce"][name] for row in rows]
        for name in ("layout_seconds", "index_seconds", "selection_seconds",
                     "compile_seconds", "compiler_seconds")
    }
    dense_total = sum(dense_request)
    spruce_total = sum(spruce_request)
    dense_ttft_total = sum(dense_ttft)
    spruce_ttft_total = sum(spruce_ttft)
    return {
        "samples": len(rows),
        "tasks": len(by_task),
        "score_dense_macro": statistics.fmean(dense_task),
        "score_spruce_macro": statistics.fmean(spruce_task),
        "delta_macro": statistics.fmean(spruce_task) - statistics.fmean(dense_task),
        "dense_bootstrap_95ci": stratified_bootstrap(
            rows, lambda row: row["dense"]["score"]),
        "spruce_bootstrap_95ci": stratified_bootstrap(
            rows, lambda row: row["spruce"]["score"]),
        "delta_bootstrap_95ci": stratified_bootstrap(
            rows, lambda row: row["score_delta_spruce_minus_dense"]),
        "spruce_wins": sum(value > 1e-12 for value in deltas),
        "dense_wins": sum(value < -1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "median_dense_tokens": statistics.median(
            row["dense"]["input_tokens"] for row in rows),
        "median_spruce_tokens": statistics.median(
            row["spruce"]["input_tokens"] for row in rows),
        "median_compression": statistics.median(
            row["spruce"]["compression_fraction"] for row in rows),
        "mean_evidence_score": statistics.fmean(
            row["spruce"]["evidence_score"] for row in rows),
        "dense_total_request_seconds": dense_total,
        "spruce_total_request_seconds_fully_charged": spruce_total,
        "spruce_total_reader_request_seconds": sum(
            row["spruce"]["request_seconds"] for row in rows),
        "spruce_total_compiler_seconds": sum(
            row["spruce"]["compiler_seconds"] for row in rows),
        "fully_charged_speedup_ratio_of_sums": float(
            dense_total / spruce_total if spruce_total > 0 else 0.0),
        "fully_charged_time_reduction_fraction": float(
            1.0 - spruce_total / dense_total if dense_total > 0 else 0.0),
        "median_dense_request_seconds": statistics.median(dense_request),
        "median_spruce_request_seconds_fully_charged": statistics.median(
            spruce_request),
        "median_pair_speedup": statistics.median(
            dense / spruce for dense, spruce in zip(dense_request, spruce_request)
            if spruce > 0),
        "p10_pair_speedup": float(sorted(
            dense / spruce for dense, spruce in zip(dense_request, spruce_request)
            if spruce > 0)[max(0, math.ceil(0.10 * len(rows)) - 1)]),
        "p90_pair_speedup": float(sorted(
            dense / spruce for dense, spruce in zip(dense_request, spruce_request)
            if spruce > 0)[min(len(rows) - 1, math.ceil(0.90 * len(rows)) - 1)]),
        "dense_total_ttft_seconds": dense_ttft_total,
        "spruce_total_ttft_seconds_fully_charged": spruce_ttft_total,
        "fully_charged_ttft_speedup_ratio_of_sums": float(
            dense_ttft_total / spruce_ttft_total
            if spruce_ttft_total > 0 else 0.0),
        "median_dense_ttft_seconds": statistics.median(dense_ttft),
        "median_spruce_ttft_seconds_fully_charged": statistics.median(
            spruce_ttft),
        "dense_generated_tokens": sum(
            row["dense"]["generated_tokens"] for row in rows),
        "spruce_generated_tokens": sum(
            row["spruce"]["generated_tokens"] for row in rows),
        "dense_decode_tokens_per_second": float(
            sum(row["dense"]["generated_tokens"] - 1 for row in rows)
            / max(1e-12, sum(row["dense"]["decode_seconds"] for row in rows))),
        "spruce_decode_tokens_per_second": float(
            sum(row["spruce"]["generated_tokens"] - 1 for row in rows)
            / max(1e-12, sum(row["spruce"]["decode_seconds"] for row in rows))),
        "median_dense_peak_memory_allocated_gb": statistics.median(
            row["dense"]["peak_memory_allocated_gb"] for row in rows),
        "median_spruce_peak_memory_allocated_gb": statistics.median(
            row["spruce"]["peak_memory_allocated_gb"] for row in rows),
        "mean_dense_request_seconds": statistics.fmean(dense_request),
        "mean_dense_preflight_tokenize_seconds": statistics.fmean(
            row["dense"].get("preflight_tokenize_seconds", 0.0) for row in rows),
        "mean_spruce_reader_request_seconds": statistics.fmean(
            row["spruce"]["request_seconds"] for row in rows),
        **{
            f"mean_spruce_{name}": statistics.fmean(values)
            for name, values in compiler_components.items()
        },
    }


def collect_reports(task_dir: Path, tasks: Sequence[str], lengths: Sequence[int]):
    rows, errors, completed = [], [], set()
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
    import numpy as np

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_files: list[str] = []
    prefix = "fig" if summary["run_complete"] else "PARTIAL"
    lengths = sorted(int(value) for value in summary["by_length"])
    x = np.asarray([value // 1024 for value in lengths])
    blocks = [summary["by_length"][str(value)] for value in lengths]
    dense = np.asarray([block["score_dense_macro"] * 100 for block in blocks])
    spruce = np.asarray([block["score_spruce_macro"] * 100 for block in blocks])
    dense_ci = np.asarray([block["dense_bootstrap_95ci"] for block in blocks]) * 100
    spruce_ci = np.asarray([block["spruce_bootstrap_95ci"] for block in blocks]) * 100
    delta = np.asarray([block["delta_macro"] * 100 for block in blocks])
    delta_ci = np.asarray([block["delta_bootstrap_95ci"] for block in blocks]) * 100

    def save(fig, stem: str) -> None:
        if not summary["run_complete"]:
            fig.text(0.5, 0.5, "PARTIAL — NOT FOR PAPER",
                     ha="center", va="center", fontsize=28,
                     color="#b00020", alpha=0.22, rotation=25)
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"{stem}.{suffix}",
                        dpi=220 if suffix == "png" else None,
                        bbox_inches="tight")
        figure_files.append(f"{stem}.png")
        plt.close(fig)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    top.plot(x, dense, marker="s", linewidth=2.2, color="#4a4a4a",
             label="Dense Qwen (without SPRUCE)")
    top.fill_between(x, dense_ci[:, 0], dense_ci[:, 1],
                     color="#4a4a4a", alpha=0.14)
    top.plot(x, spruce, marker="o", linewidth=2.2, color="#16866f",
             label="Qwen + SPRUCE (D=4096, beam=64, M=32)")
    top.fill_between(x, spruce_ci[:, 0], spruce_ci[:, 1],
                     color="#16866f", alpha=0.16)
    top.set(ylabel="Macro RULER score (%)",
            title="Held-out NVIDIA RULER: accuracy with and without SPRUCE")
    top.set_ylim(0, 100)
    top.grid(alpha=0.22)
    top.legend(fontsize=9)
    errors = np.maximum(
        0.0, np.vstack([delta - delta_ci[:, 0], delta_ci[:, 1] - delta]))
    bottom.bar(x, delta, width=6, color=[
        "#16866f" if value >= 0 else "#c44e52" for value in delta])
    bottom.errorbar(x, delta, yerr=errors, fmt="none", ecolor="black",
                    elinewidth=1, capsize=3)
    bottom.axhline(0, color="black", linewidth=0.9)
    bottom.set(xlabel="Context length (Ki tokens)",
               ylabel="SPRUCE - dense\n(points)")
    bottom.set_xticks(x)
    bottom.grid(alpha=0.2, axis="y")
    main_name = f"{prefix}_ruler_dense_vs_spruce"
    save(fig, main_name)

    tasks = list(summary["by_task"])
    matrix = np.asarray([
        [summary["by_task_length"].get(
            f"{task}_{length}", {}).get("delta_macro", float("nan")) * 100
         for length in lengths]
        for task in tasks])
    finite = np.abs(matrix[np.isfinite(matrix)])
    bound = max(1.0, float(np.max(finite)) if finite.size else 1.0)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                      vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(lengths)), [f"{value // 1024}K" for value in lengths])
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
    ax.set(xlabel="Context length", title="SPRUCE - dense by RULER task (points)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(column, row, f"{matrix[row, column]:+.0f}",
                        ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="accuracy delta")
    heat_name = f"{prefix}_ruler_task_delta_heatmap"
    save(fig, heat_name)

    task_x = np.arange(len(tasks))
    task_dense = np.asarray([
        summary["by_task"][task]["score_dense_macro"] * 100 for task in tasks])
    task_spruce = np.asarray([
        summary["by_task"][task]["score_spruce_macro"] * 100 for task in tasks])
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    width = 0.38
    ax.bar(task_x - width / 2, task_dense, width, color="#4a4a4a",
           label="Dense Qwen")
    ax.bar(task_x + width / 2, task_spruce, width, color="#16866f",
           label="Qwen + SPRUCE")
    ax.set_xticks(task_x, tasks, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Macro score (%)", title="RULER accuracy by task family",
           ylim=(0, 100))
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    save(fig, f"{prefix}_ruler_accuracy_by_task")

    dense_seconds = np.asarray([
        block["mean_dense_request_seconds"] for block in blocks])
    spruce_seconds = np.asarray([
        block["spruce_total_request_seconds_fully_charged"] / block["samples"]
        for block in blocks])
    speedup = np.asarray([
        block["fully_charged_speedup_ratio_of_sums"] for block in blocks])
    ttft_speedup = np.asarray([
        block["fully_charged_ttft_speedup_ratio_of_sums"] for block in blocks])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.08})
    top.plot(x, dense_seconds, marker="s", linewidth=2.2, color="#4a4a4a",
             label="Dense full request")
    top.plot(x, spruce_seconds, marker="o", linewidth=2.2, color="#16866f",
             label="SPRUCE fully charged")
    top.set_yscale("log")
    top.set(ylabel="Mean seconds / request (log)",
            title="Same-run L4 speed: cold compiler + reader included")
    top.grid(alpha=0.22, which="both")
    top.legend(fontsize=9)
    bottom.plot(x, speedup, marker="o", linewidth=2.2, color="#16866f",
                label="End-to-end request speedup")
    bottom.plot(x, ttft_speedup, marker="^", linewidth=1.8, color="#4c72b0",
                label="Fully charged TTFT speedup")
    bottom.axhline(1.0, color="black", linewidth=0.9)
    bottom.set(xlabel="Context length (Ki tokens)", ylabel="Dense / SPRUCE (x)")
    bottom.set_xticks(x)
    bottom.grid(alpha=0.2)
    bottom.legend(fontsize=8)
    save(fig, f"{prefix}_ruler_speed_by_length")

    positions = np.arange(len(lengths))
    components = [
        ("Layout", "mean_spruce_layout_seconds", "#8dd3c7"),
        ("Index build", "mean_spruce_index_seconds", "#80b1d3"),
        ("Selection", "mean_spruce_selection_seconds", "#bebada"),
        ("Packet compile", "mean_spruce_compile_seconds", "#fdb462"),
        ("Compact reader request", "mean_spruce_reader_request_seconds", "#16866f"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    bottoms = np.zeros(len(lengths))
    for label, key, color in components:
        values = np.asarray([block[key] for block in blocks])
        ax.bar(positions + 0.18, values, width=0.36, bottom=bottoms,
               color=color, label=label)
        bottoms += values
    ax.bar(positions - 0.18, dense_seconds, width=0.36, color="#4a4a4a",
           label="Dense full request")
    ax.set_xticks(positions, [f"{value // 1024}K" for value in lengths])
    ax.set(xlabel="Context length", ylabel="Mean seconds / request",
           title="Fully charged time decomposition on one NVIDIA L4")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=8, ncol=2)
    save(fig, f"{prefix}_ruler_time_breakdown")

    dense_tokens = np.asarray([block["median_dense_tokens"] for block in blocks])
    spruce_tokens = np.asarray([block["median_spruce_tokens"] for block in blocks])
    compression = np.asarray([block["median_compression"] * 100 for block in blocks])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08})
    top.plot(x, dense_tokens, marker="s", color="#4a4a4a", label="Dense")
    top.plot(x, spruce_tokens, marker="o", color="#16866f", label="SPRUCE")
    top.set(ylabel="Median input tokens", title="Prompt size and compression")
    top.grid(alpha=0.2)
    top.legend()
    bottom.plot(x, compression, marker="o", color="#d17c18")
    bottom.set(xlabel="Context length (Ki tokens)",
               ylabel="SPRUCE / dense tokens (%)")
    bottom.set_xticks(x)
    bottom.grid(alpha=0.2)
    save(fig, f"{prefix}_ruler_prompt_compression")

    dense_memory = np.asarray([
        block["median_dense_peak_memory_allocated_gb"] for block in blocks])
    spruce_memory = np.asarray([
        block["median_spruce_peak_memory_allocated_gb"] for block in blocks])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, dense_memory, marker="s", linewidth=2.2, color="#4a4a4a",
            label="Dense reader")
    ax.plot(x, spruce_memory, marker="o", linewidth=2.2, color="#16866f",
            label="SPRUCE compact reader")
    ax.set(xlabel="Context length (Ki tokens)",
           ylabel="Median peak allocated GPU memory (GB)",
           title="Reader-side allocated GPU memory")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, f"{prefix}_ruler_memory_by_length")

    speed_matrix = np.asarray([[
        summary["by_task_length"].get(f"{task}_{length}", {}).get(
            "fully_charged_speedup_ratio_of_sums", float("nan"))
        for length in lengths] for task in tasks])
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    finite_speed = speed_matrix[np.isfinite(speed_matrix)]
    vmax = max(1.0, float(np.max(finite_speed)) if finite_speed.size else 1.0)
    image = ax.imshow(speed_matrix, aspect="auto", cmap="viridis", vmin=0,
                      vmax=vmax)
    ax.set_xticks(range(len(lengths)), [f"{value // 1024}K" for value in lengths])
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
    ax.set(xlabel="Context length",
           title="Fully charged end-to-end speedup by task and length")
    for row in range(speed_matrix.shape[0]):
        for column in range(speed_matrix.shape[1]):
            if np.isfinite(speed_matrix[row, column]):
                ax.text(column, row, f"{speed_matrix[row, column]:.1f}x",
                        ha="center", va="center", fontsize=5.5, color="white")
    fig.colorbar(image, ax=ax, label="dense time / SPRUCE time")
    save(fig, f"{prefix}_ruler_speedup_heatmap")

    dense_decode = np.asarray([
        block["dense_decode_tokens_per_second"] for block in blocks])
    spruce_decode = np.asarray([
        block["spruce_decode_tokens_per_second"] for block in blocks])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, dense_decode, marker="s", color="#4a4a4a", label="Dense")
    ax.plot(x, spruce_decode, marker="o", color="#16866f", label="SPRUCE")
    ax.set(xlabel="Context length (Ki tokens)",
           ylabel="Decode tokens / second",
           title="Dense-decode throughput after the first generated token")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, f"{prefix}_ruler_decode_throughput")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter = ax.scatter(speedup, delta, c=x, s=80, cmap="viridis")
    for index, length in enumerate(lengths):
        ax.annotate(f"{length // 1024}K", (speedup[index], delta[index]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(1, color="black", linewidth=0.9)
    ax.set(xlabel="Fully charged end-to-end speedup (x)",
           ylabel="SPRUCE - dense accuracy (points)",
           title="Accuracy-speed tradeoff by context length")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Context length (Ki tokens)")
    save(fig, f"{prefix}_ruler_accuracy_speed_tradeoff")
    return figure_files


TABLE_METRICS = (
    "samples", "score_dense_macro", "score_spruce_macro", "delta_macro",
    "dense_bootstrap_95ci", "spruce_bootstrap_95ci", "delta_bootstrap_95ci",
    "spruce_wins", "dense_wins", "ties", "mean_evidence_score",
    "median_dense_tokens", "median_spruce_tokens", "median_compression",
    "dense_total_request_seconds", "spruce_total_request_seconds_fully_charged",
    "spruce_total_reader_request_seconds", "spruce_total_compiler_seconds",
    "fully_charged_speedup_ratio_of_sums",
    "fully_charged_time_reduction_fraction", "median_pair_speedup",
    "p10_pair_speedup", "p90_pair_speedup",
    "fully_charged_ttft_speedup_ratio_of_sums",
    "median_dense_request_seconds", "median_spruce_request_seconds_fully_charged",
    "median_dense_ttft_seconds", "median_spruce_ttft_seconds_fully_charged",
    "dense_decode_tokens_per_second", "spruce_decode_tokens_per_second",
    "median_dense_peak_memory_allocated_gb",
    "median_spruce_peak_memory_allocated_gb", "mean_spruce_layout_seconds",
    "mean_dense_preflight_tokenize_seconds",
    "mean_spruce_index_seconds", "mean_spruce_selection_seconds",
    "mean_spruce_compile_seconds", "mean_spruce_reader_request_seconds",
)


def write_metric_tables(summary: dict, output_dir: Path) -> list[str]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("metrics_by_length.csv", "length", summary["by_length"]),
        ("metrics_by_task.csv", "task", summary["by_task"]),
        ("metrics_by_task_length.csv", "task_length", summary["by_task_length"]),
    )
    written = []
    for filename, group_name, groups in specifications:
        path = table_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=(group_name,) + TABLE_METRICS)
            writer.writeheader()
            for group, block in groups.items():
                row = {group_name: group}
                for metric in TABLE_METRICS:
                    value = block.get(metric)
                    row[metric] = (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict)) else value)
                writer.writerow(row)
        written.append(str(path.relative_to(output_dir)).replace("\\", "/"))
    return written


def write_case_table(rows: Sequence[dict], output_dir: Path) -> str:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "paired_cases.csv"
    fields = (
        "task", "length", "position", "source_index", "generation_order",
        "dense_score", "spruce_score", "score_delta", "evidence_score",
        "dense_input_tokens", "spruce_input_tokens", "compression_fraction",
        "dense_generated_tokens", "spruce_generated_tokens",
        "dense_fully_charged_seconds", "spruce_fully_charged_seconds",
        "dense_ttft_seconds", "spruce_fully_charged_ttft_seconds",
        "dense_decode_seconds", "spruce_decode_seconds",
        "dense_decode_tokens_per_second", "spruce_decode_tokens_per_second",
        "dense_peak_memory_allocated_gb", "spruce_peak_memory_allocated_gb",
        "layout_seconds", "index_seconds", "selection_seconds",
        "compile_seconds", "compiler_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (
                int(value["length"]), value["task"], int(value["position"]))):
            dense, spruce = row["dense"], row["spruce"]
            writer.writerow({
                "task": row["task"], "length": row["length"],
                "position": row["position"],
                "source_index": row.get("source_index"),
                "generation_order": row.get("generation_order"),
                "dense_score": dense["score"], "spruce_score": spruce["score"],
                "score_delta": row["score_delta_spruce_minus_dense"],
                "evidence_score": spruce["evidence_score"],
                "dense_input_tokens": dense["input_tokens"],
                "spruce_input_tokens": spruce["input_tokens"],
                "compression_fraction": spruce["compression_fraction"],
                "dense_generated_tokens": dense["generated_tokens"],
                "spruce_generated_tokens": spruce["generated_tokens"],
                "dense_fully_charged_seconds": dense["fully_charged_request_seconds"],
                "spruce_fully_charged_seconds": spruce["fully_charged_request_seconds"],
                "dense_ttft_seconds": dense["fully_charged_ttft_seconds"],
                "spruce_fully_charged_ttft_seconds": spruce[
                    "fully_charged_ttft_seconds"],
                "dense_decode_seconds": dense["decode_seconds"],
                "spruce_decode_seconds": spruce["decode_seconds"],
                "dense_decode_tokens_per_second": dense[
                    "decode_tokens_per_second"],
                "spruce_decode_tokens_per_second": spruce[
                    "decode_tokens_per_second"],
                "dense_peak_memory_allocated_gb": dense[
                    "peak_memory_allocated_gb"],
                "spruce_peak_memory_allocated_gb": spruce[
                    "peak_memory_allocated_gb"],
                "layout_seconds": spruce["layout_seconds"],
                "index_seconds": spruce["index_seconds"],
                "selection_seconds": spruce["selection_seconds"],
                "compile_seconds": spruce["compile_seconds"],
                "compiler_seconds": spruce["compiler_seconds"],
            })
    return str(path.relative_to(output_dir)).replace("\\", "/")


def write_summary(summary: dict, path: Path) -> None:
    overall = summary["overall"]
    lines = [
        "# Held-out NVIDIA RULER: dense versus SPRUCE",
        "",
        f"- complete: {summary['run_complete']}",
        f"- GPU: {summary['gpu_name']}",
        f"- samples: {summary['sample_count']}",
        f"- dataset manifest SHA-256: `{summary['dataset_manifest_sha256']}`",
        f"- dense macro score: {overall['score_dense_macro']:.4f}",
        f"- SPRUCE macro score: {overall['score_spruce_macro']:.4f}",
        f"- delta: {overall['delta_macro']:+.4f}",
        f"- fully charged end-to-end speedup: "
        f"{overall['fully_charged_speedup_ratio_of_sums']:.3f}x",
        f"- fully charged time reduction: "
        f"{overall['fully_charged_time_reduction_fraction'] * 100:.2f}%",
        f"- fully charged TTFT speedup: "
        f"{overall['fully_charged_ttft_speedup_ratio_of_sums']:.3f}x",
        f"- median pair speedup [P10, P90]: "
        f"{overall['median_pair_speedup']:.3f}x "
        f"[{overall['p10_pair_speedup']:.3f}x, "
        f"{overall['p90_pair_speedup']:.3f}x]",
        "",
        "## By length",
        "",
        "| length | dense | SPRUCE | delta | 95% CI delta | speedup | TTFT speedup | dense/SPRUCE tokens | dense/SPRUCE GB | n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length, block in summary["by_length"].items():
        low, high = block["delta_bootstrap_95ci"]
        lines.append(
            f"| {length} | {block['score_dense_macro']:.3f} | "
            f"{block['score_spruce_macro']:.3f} | {block['delta_macro']:+.3f} | "
            f"[{low:+.3f},{high:+.3f}] | "
            f"{block['fully_charged_speedup_ratio_of_sums']:.2f}x | "
            f"{block['fully_charged_ttft_speedup_ratio_of_sums']:.2f}x | "
            f"{block['median_dense_tokens']:.0f}/{block['median_spruce_tokens']:.0f} | "
            f"{block['median_dense_peak_memory_allocated_gb']:.2f}/"
            f"{block['median_spruce_peak_memory_allocated_gb']:.2f} | "
            f"{block['samples']} |")
    lines += [
        "",
        "## By task",
        "",
        "| task | dense | SPRUCE | delta | speedup | evidence score | n |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, block in summary["by_task"].items():
        lines.append(
            f"| {task} | {block['score_dense_macro']:.3f} | "
            f"{block['score_spruce_macro']:.3f} | {block['delta_macro']:+.3f} | "
            f"{block['fully_charged_speedup_ratio_of_sums']:.2f}x | "
            f"{block['mean_evidence_score']:.3f} | {block['samples']} |")
    if not summary["run_complete"]:
        lines += ["", "> PARTIAL — NOT FOR PAPER. Resume until all 78 cells complete with zero errors."]
    lines += [
        "",
        "## Protocol",
        "",
        "- Fresh NVIDIA RULER test split and frozen per-file hashes.",
        "- All 13 tasks; task-macro average at each context length.",
        "- Official answer_prefix and task-specific generation budgets.",
        "- Exact task-template boundaries; no trailing-line query heuristic.",
        "- SPRUCE candidate fixed from validation: D=4096, beam=64, M=32.",
        "- Same held-out cases, Qwen model, YaRN, dtype, decoding, and GPU.",
        "- Arm order is counterbalanced by sample-position parity within every cell.",
        "- Dense reads the official full prompt; SPRUCE reads its compiled evidence prompt.",
        "- SPRUCE is the evidence compiler plus dense reader, not Stage-3 sparse attention.",
        "- Fully charged SPRUCE time includes layout, cold per-sample index build, selector, packet compilation, compact tokenization, transfer, prefill, and decode.",
        "- Fully charged dense time includes its context-limit preflight tokenization, request tokenization, transfer, prefill, and decode.",
        "- TTFT is synchronized at the first generation logits-processor callback; SPRUCE TTFT also includes every compiler stage.",
        "- Speed is secondary same-run telemetry with one accuracy generation per prompt, not a dedicated repeated latency benchmark.",
        "- Peak memory is PyTorch reader-side allocated GPU memory; CPU compiler memory and CUDA reserved memory are not represented.",
        "- Ratio-of-sums is the primary speed aggregate; per-pair P10/P50/P90 are descriptive and have no repeated-run confidence interval.",
        f"- Detailed CSV tables: {', '.join(summary.get('tables', []))}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--subset", default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--dataset-seed", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--candidate-blocks", type=int, default=32)
    parser.add_argument("--beam", type=int, default=64)
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
    observed_candidate = {
        key: getattr(args, key) for key in PAPER_CANDIDATE
    }
    if observed_candidate != PAPER_CANDIDATE:
        raise SystemExit(
            "paper candidate is frozen at "
            "D=4096, beam=64, M=32, B=64, radius=1, paragraph boundary, "
            "unigram_fraction=0.5, idf_power=2.0, radix=2; observed "
            f"{observed_candidate}")
    if set(args.tasks) != set(TASK_MAX_NEW_TOKENS):
        raise SystemExit("paper run requires exactly all 13 RULER tasks")
    if set(args.lengths) != set(DEFAULT_LENGTHS):
        raise SystemExit("paper run requires the full 4K/8K/16K/32K/64K/128K grid")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    gpu_name = properties.name
    if args.require_gpu_substring.lower() not in gpu_name.lower():
        raise SystemExit(f"paper run requires L4; current GPU is {gpu_name}")

    manifest, manifest_hash = verify_dataset_manifest(
        args.data_dir, args.dataset_manifest, args.tasks, args.lengths,
        args.subset, args.num_samples)
    if int(manifest.get("random_seed", -1)) != args.dataset_seed:
        raise SystemExit("dataset manifest seed does not match --dataset-seed")
    manifest_model = str(manifest.get("tokenizer_path", manifest.get("model", "")))
    if manifest_model != args.model:
        raise SystemExit(
            "dataset manifest tokenizer/model does not match --model: "
            f"{manifest_model!r} != {args.model!r}")
    config = {
        "model": args.model,
        "tasks": list(args.tasks),
        "lengths": list(args.lengths),
        "subset": args.subset,
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "dataset_manifest_sha256": manifest_hash,
        "feature_dim": args.feature_dim,
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
        "dtype": "float16",
        "decoding": "greedy",
        "task_max_new_tokens": TASK_MAX_NEW_TOKENS,
        "official_answer_prefix": True,
        "prompt_boundary_parser": PROMPT_BOUNDARY_PARSER,
        "candidate_selection_source": "ruler_ceiling_knee_validation_v1",
        "telemetry_schema": "synchronized_ttft_fully_charged_v1",
        "generation_order": "position_parity_counterbalanced",
    }
    fingerprint = config_fingerprint(config)
    output_dir = args.output_dir
    task_dir = output_dir / "task_reports"
    figure_dir = output_dir / "figures"
    task_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    samples_by_pair = {
        (task, int(length)): read_samples(
            args.data_dir, task, int(length), args.subset)
        for length in args.lengths for task in args.tasks
    }
    if any(len(rows) != args.num_samples for rows in samples_by_pair.values()):
        raise SystemExit("dataset row count changed after manifest verification")
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
        done = {int(case["position"]) for case in report.get("cases", [])}
        failed = {int(error["position"]) for error in report.get("errors", [])}
        if args.rerun_errors:
            failed.clear()
        if len(done | failed) < len(samples):
            pending = True

    session_started = time.perf_counter()
    model = tokenizer = None
    try:
        if pending:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            configure_tokenizer(
                tokenizer, yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_context)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"GPU: {gpu_name} | VRAM {properties.total_memory / 2**30:.1f} GiB")
            print(f"dataset manifest: {manifest_hash}")
            model = _load_model(
                args.model, "sdpa", torch.float16,
                str(Path(tempfile.gettempdir()) / "spruce_ruler_paper_offload"),
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_context)
            generate_one(model, tokenizer, "Answer briefly.", 1)

        budget_hit = False
        execution_lengths = sorted(args.lengths, reverse=True)
        execution_tasks = ["cwe"] + [task for task in args.tasks if task != "cwe"]
        for length in execution_lengths:
            if budget_hit:
                break
            print(f"=== {length} tokens ===", flush=True)
            for task in execution_tasks:
                if budget_hit:
                    break
                path = task_dir / f"{task}_{length}.json"
                if path.is_file():
                    report = json.loads(path.read_text(encoding="utf-8"))
                else:
                    report = {
                        "kind": "spruce_ruler_paper_accuracy_task_v3",
                        "task": task, "length": length,
                        "gpu_name": gpu_name,
                        "config": config,
                        "config_fingerprint": fingerprint,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "cases": [], "errors": [], "status": "running",
                    }
                if report.get("config_fingerprint") != fingerprint:
                    raise SystemExit(f"refusing mixed resume config in {path}")
                if args.rerun_errors:
                    report["errors"] = []
                done = {int(case["position"]) for case in report.get("cases", [])}
                failed = {int(error["position"]) for error in report.get("errors", [])}
                for position, sample in enumerate(samples_by_pair[(task, length)]):
                    if position in done or position in failed:
                        continue
                    if args.session_budget_minutes and (
                            time.perf_counter() - session_started
                            >= args.session_budget_minutes * 60):
                        budget_hit = True
                        break
                    try:
                        generation_order = (
                            "dense_first" if position % 2 == 0 else "spruce_first")
                        result = run_case(
                            model, tokenizer, sample, task, args,
                            generation_order=generation_order)
                        result.update({
                            "position": position,
                            "source_index": sample.get("index", position),
                            "task": task,
                            "length": length,
                        })
                        report["cases"].append(result)
                        done.add(position)
                        atomic_json(path, report)
                        print(
                            "  %-18s #%03d dense %.2f spruce %.2f delta %+.2f"
                            % (task, position, result["dense"]["score"],
                               result["spruce"]["score"],
                               result["score_delta_spruce_minus_dense"]),
                            flush=True)
                    except Exception as error:  # noqa: BLE001
                        if isinstance(error, torch.cuda.OutOfMemoryError):
                            gc.collect()
                            torch.cuda.empty_cache()
                        report["errors"].append({
                            "position": position,
                            "source_index": sample.get("index", position),
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        })
                        atomic_json(path, report)
                        print(f"  ERROR {task} {length} #{position}: {error!r}")
                attempted = done | {
                    int(error["position"]) for error in report.get("errors", [])}
                if len(attempted) == args.num_samples and not report.get("errors"):
                    report["status"] = "completed"
                    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
                elif budget_hit:
                    report["status"] = "running"
                else:
                    report["status"] = "completed_with_errors"
                atomic_json(path, report)
                if report["cases"]:
                    dense_mean = statistics.fmean(
                        row["dense"]["score"] for row in report["cases"])
                    spruce_mean = statistics.fmean(
                        row["spruce"]["score"] for row in report["cases"])
                    print(f"  {task} aggregate {dense_mean:.3f} -> {spruce_mean:.3f}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if model is not None:
            model = _free_model(model)

    rows, errors, completed = collect_reports(task_dir, args.tasks, args.lengths)
    if not rows:
        raise SystemExit("no completed samples exist to aggregate")
    run_complete = completed >= expected_pairs and not errors and len(rows) == (
        len(expected_pairs) * args.num_samples)
    by_task = {
        str(key): macro_block(value)
        for key, value in sorted(grouped(rows, lambda row: row["task"]).items())
    }
    by_length = {
        str(key): macro_block(value)
        for key, value in sorted(grouped(rows, lambda row: int(row["length"])).items())
    }
    by_task_length = {
        f"{task}_{length}": macro_block(values)
        for (task, length), values in sorted(grouped(
            rows, lambda row: (row["task"], int(row["length"]))).items())
    }
    summary = {
        "kind": "spruce_ruler_paper_accuracy_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": args.archive_sha256,
        "dataset_manifest_sha256": manifest_hash,
        "gpu_name": gpu_name,
        "gpu_vram_gib": properties.total_memory / 2**30,
        "config": config,
        "config_fingerprint": fingerprint,
        "run_complete": run_complete,
        "completed_pairs": len(completed),
        "expected_pairs": len(expected_pairs),
        "sample_count": len(rows),
        "expected_samples": len(expected_pairs) * args.num_samples,
        "error_count": len(errors),
        "errors": errors,
        "session_wall_clock_seconds": time.perf_counter() - session_started,
        "overall": macro_block(rows),
        "by_task": by_task,
        "by_length": by_length,
        "by_task_length": by_task_length,
    }
    summary["tables"] = write_metric_tables(summary, output_dir)
    summary["tables"].append(write_case_table(rows, output_dir))
    summary["figures"] = make_figures(summary, figure_dir)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "run_record.json", {
        key: summary[key] for key in (
            "kind", "created_utc", "archive_sha256",
            "dataset_manifest_sha256", "gpu_name", "config",
            "config_fingerprint", "run_complete", "completed_pairs",
            "expected_pairs", "sample_count", "expected_samples",
            "error_count", "session_wall_clock_seconds", "overall",
            "by_task", "by_length")
    })
    write_summary(summary, output_dir / "SUMMARY.md")
    overall = summary["overall"]
    print(
        "OVERALL dense %.4f SPRUCE %.4f delta %+.4f; pairs %d/%d; "
        "samples %d/%d; errors %d"
        % (overall["score_dense_macro"], overall["score_spruce_macro"],
           overall["delta_macro"], len(completed), len(expected_pairs),
           len(rows), summary["expected_samples"], len(errors)))
    if not run_complete:
        print("PARTIAL — final figures are watermarked; resume before paper use.")


if __name__ == "__main__":
    main()
