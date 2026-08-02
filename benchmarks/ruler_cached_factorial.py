"""Cached compiler-only RULER factorial ablation.

This runner isolates three factors without loading model weights:

* query boundary (the number of trailing non-empty lines),
* lexical sketch width, and
* selected-block budget.

The expensive, query-independent work is cached within each sample.  The
prompt is tokenized once, each (tail, feature_dim) index is built once, and one
beam-ranked traversal supplies every requested M by prefix.  Reports are
checkpointed after every sample and can be resumed safely.
"""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Iterable, Sequence

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.long_context import configure_tokenizer
from interfaces.evidence_compiler import (
    PromptLayout,
    compile_evidence_packet_from_layout,
    document_block_ids,
    locate_prompt_layout,
)
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


TASK_TYPES = {
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa",
    "qa_2": "qa",
}
DEFAULT_TASKS = tuple(TASK_TYPES)
DEFAULT_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
VERDICTS = ("ok", "data", "selector", "stitch")


def atomic_json(path: Path, payload: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def string_match_all(prediction: str, references: Sequence[str]) -> float:
    prediction = (prediction or "").lower()
    if not references:
        return 0.0
    return sum(str(ref).lower() in prediction for ref in references) / len(
        references
    )


def string_match_part(prediction: str, references: Sequence[str]) -> float:
    prediction = (prediction or "").lower()
    return float(any(str(ref).lower() in prediction for ref in references))


def score_for(task: str, prediction: str, references: Sequence[str]) -> float:
    if TASK_TYPES[task] == "qa":
        return string_match_part(prediction, references)
    return string_match_all(prediction, references)


def extract_query(prompt_text: str, tail_lines: int) -> str:
    """Return an exact suffix containing ``tail_lines`` non-empty lines."""
    lines = prompt_text.splitlines(keepends=True)
    taken = 0
    nonempty = 0
    for line in reversed(lines):
        taken += 1
        if line.strip():
            nonempty += 1
        if nonempty >= int(tail_lines):
            break
    question = "".join(lines[len(lines) - taken :]) if taken else ""
    if not question.strip() or len(question) >= len(prompt_text):
        question = prompt_text[-256:]
    if not prompt_text.endswith(question):
        raise AssertionError("query is not an exact prompt suffix")
    return question


def _flat_ints(values) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one tokenized prompt")
        values = values[0]
    return tuple(int(value) for value in values)


def _flat_offsets(values) -> tuple[tuple[int, int], ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(
        values[0][0], (list, tuple)
    ):
        if len(values) != 1:
            raise ValueError("expected one offset-mapped prompt")
        values = values[0]
    return tuple((int(start), int(end)) for start, end in values)


class CharToToken:
    """Fast character-to-token lookup over non-empty tokenizer offsets."""

    def __init__(self, offsets: Sequence[tuple[int, int]]) -> None:
        real = sorted(
            (
                (int(start), int(end), index)
                for index, (start, end) in enumerate(offsets)
                if int(end) > int(start)
            ),
            key=lambda item: (item[0], item[1]),
        )
        self.starts = [item[0] for item in real]
        self.ends = [item[1] for item in real]
        self.indices = [item[2] for item in real]

    def token_range(self, char_start: int, char_end: int) -> tuple[int, int] | None:
        if not self.starts or int(char_end) <= int(char_start):
            return None
        first = bisect.bisect_right(self.ends, int(char_start))
        last = bisect.bisect_left(self.starts, int(char_end)) - 1
        if first > last:
            return None
        return self.indices[first], self.indices[last] + 1


def tokenize_layouts(tokenizer, prompt_text: str, tails: Sequence[int]):
    """Tokenize once, then derive an exact production-equivalent layout/tail."""
    encoded = tokenizer(prompt_text, return_offsets_mapping=True)
    input_ids = _flat_ints(encoded["input_ids"])
    offsets = _flat_offsets(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned different id and offset counts")
    mapper = CharToToken(offsets)
    questions = {tail: extract_query(prompt_text, tail) for tail in tails}
    layouts = {}
    for tail, question in questions.items():
        document_end = len(prompt_text) - len(question)
        token_span = mapper.token_range(0, document_end)
        if token_span is None:
            raise ValueError("source document did not map to any tokens")
        layouts[tail] = PromptLayout(
            input_ids=input_ids,
            offsets=offsets,
            document_char_start=0,
            document_char_end=document_end,
            document_token_start=token_span[0],
            document_token_end=token_span[1],
        )
    return questions, layouts, mapper


def find_occurrences(
    haystack_lower: str,
    needle_lower: str,
    lo: int,
    hi: int,
    cap: int,
) -> tuple[list[int], bool]:
    if not needle_lower:
        return [], False
    hits: list[int] = []
    cursor = int(lo)
    while True:
        found = haystack_lower.find(needle_lower, cursor, int(hi))
        if found < 0:
            return hits, False
        hits.append(found)
        if len(hits) >= int(cap):
            return hits, haystack_lower.find(
                needle_lower, found + 1, int(hi)
            ) >= 0
        cursor = found + 1


def blocks_for_char_span(
    mapper: CharToToken, char_start: int, char_end: int, block_size: int
) -> set[int]:
    span = mapper.token_range(char_start, char_end)
    if span is None:
        return set()
    token_start, token_end = span
    return set(
        range(
            token_start // int(block_size),
            (token_end - 1) // int(block_size) + 1,
        )
    )


def reference_gold(
    prompt_text: str,
    layout: PromptLayout,
    mapper: CharToToken,
    references: Sequence[str],
    block_size: int,
    max_occurrences: int,
) -> list[dict]:
    prompt_lower = prompt_text.lower()
    rows = []
    for reference in references:
        needle = reference.lower()
        occurrences, capped = find_occurrences(
            prompt_lower,
            needle,
            layout.document_char_start,
            layout.document_char_end,
            max_occurrences,
        )
        gold_blocks: set[int] = set()
        for start in occurrences:
            gold_blocks |= blocks_for_char_span(
                mapper, start, start + len(reference), block_size
            )
        rows.append(
            {
                "reference": reference,
                "in_document": bool(occurrences),
                "occurrences": len(occurrences),
                "occurrences_capped": bool(capped),
                "gold_blocks": sorted(gold_blocks),
            }
        )
    return rows


def index_bytes(index) -> int:
    feature_bytes = sum(
        int(level.features.numel() * level.features.element_size())
        for level in index.levels
    )
    frequency_bytes = int(
        index.document_frequencies.numel()
        * index.document_frequencies.element_size()
    )
    return feature_bytes + frequency_bytes


def arm_name(tail: int, feature_dim: int, budget: int) -> str:
    return f"t{int(tail)}_d{int(feature_dim)}_m{int(budget)}"


def parse_arm(name: str) -> tuple[int, int, int]:
    tail, dim, budget = name.split("_")
    return int(tail[1:]), int(dim[1:]), int(budget[1:])


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def load_baseline_cases(path: Path | None) -> dict[tuple[str, int, str], dict]:
    rows: dict[tuple[str, int, str], dict] = {}
    if path is None or not path.is_dir():
        return rows
    for report_path in sorted(path.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            rows[(case["task"], int(case["length"]), str(case["index"]))] = case
    return rows


def load_e2e_cases(path: Path | None) -> dict[tuple[str, int, str], dict]:
    rows: dict[tuple[str, int, str], dict] = {}
    if path is None or not path.is_dir():
        return rows
    for report_path in sorted(path.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            rows[(case["task"], int(case["length"]), str(case["index"]))] = case
    return rows


def validate_cached_layouts(tokenizer, sample: dict, tails: Sequence[int]) -> None:
    prompt_text = sample["input"]
    questions, layouts, _mapper = tokenize_layouts(tokenizer, prompt_text, tails)
    for tail in tails:
        reference = locate_prompt_layout(
            tokenizer, prompt_text, prompt_text, questions[tail]
        )
        cached = layouts[tail]
        if cached != reference:
            raise AssertionError(
                f"cached layout differs from production layout at tail={tail}"
            )


def run_case(
    sample: dict,
    task: str,
    tokenizer,
    *,
    tails: Sequence[int],
    feature_dims: Sequence[int],
    budgets: Sequence[int],
    block_size: int,
    block_radius: int,
    boundary: str,
    beam: int,
    unigram_fraction: float,
    idf_power: float,
    radix: int,
    max_occurrences: int,
    prefix_validation: set[tuple[int, int]],
) -> dict:
    sample_started = time.perf_counter()
    prompt_text = sample["input"]
    references = [
        str(value) for value in sample.get("outputs", []) if str(value).strip()
    ]

    mark = time.perf_counter()
    questions, layouts, mapper = tokenize_layouts(tokenizer, prompt_text, tails)
    tokenize_layout_seconds = time.perf_counter() - mark

    gold_by_tail = {
        tail: reference_gold(
            prompt_text,
            layouts[tail],
            mapper,
            references,
            block_size,
            max_occurrences,
        )
        for tail in tails
    }
    arms: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    max_budget = max(int(value) for value in budgets)
    if max_budget > int(beam):
        raise ValueError("every block budget must be <= beam")

    for tail in tails:
        question = questions[tail]
        layout = layouts[tail]
        for feature_dim in feature_dims:
            key = f"t{tail}_d{feature_dim}"
            mark = time.perf_counter()
            index = build_pre_qwen_index(
                layout,
                block_size,
                feature_dim=feature_dim,
                unigram_fraction=unigram_fraction,
                radix=radix,
            )
            build_seconds = time.perf_counter() - mark

            mark = time.perf_counter()
            weights = question_feature_weights(
                tokenizer, question, index, idf_power=idf_power
            )
            ranked = select_pre_qwen_blocks(
                index,
                weights,
                top_m=max_budget,
                beam=beam,
                radix=radix,
            )
            selection_seconds = time.perf_counter() - mark

            validation_key = (int(tail), int(feature_dim))
            if validation_key not in prefix_validation:
                direct = select_pre_qwen_blocks(
                    index,
                    weights,
                    top_m=min(budgets),
                    beam=beam,
                    radix=radix,
                )
                expected = tuple(ranked.blocks[: min(budgets)])
                if direct.blocks != expected:
                    raise AssertionError(
                        "M-prefix cache changed selection for "
                        f"tail={tail}, D={feature_dim}"
                    )
                prefix_validation.add(validation_key)

            cache[key] = {
                "index_seconds": build_seconds,
                "selection_seconds": selection_seconds,
                "index_bytes": index_bytes(index),
                "tree_levels": len(index.levels),
                "visited_nodes": ranked.visited_nodes,
                "document_blocks": len(document_block_ids(layout, block_size)),
            }

            for budget in budgets:
                chosen = tuple(ranked.blocks[: int(budget)])
                mark = time.perf_counter()
                packet = compile_evidence_packet_from_layout(
                    tokenizer,
                    prompt_text,
                    layout,
                    question,
                    chosen,
                    block_size,
                    block_radius=block_radius,
                    boundary=boundary,
                )
                compile_seconds = time.perf_counter() - mark

                selected = set(chosen)
                expanded = set(packet.expanded_blocks)
                content_lower = packet.content.lower()
                ranked_blocks = list(ranked.blocks)
                per_reference = []
                for gold in gold_by_tail[tail]:
                    gold_blocks = set(gold["gold_blocks"])
                    in_packet = gold["reference"].lower() in content_lower
                    selected_hit = bool(gold_blocks & selected)
                    expanded_hit = bool(gold_blocks & expanded)
                    if not gold["in_document"]:
                        verdict = "data"
                    elif in_packet:
                        verdict = "ok"
                    elif not expanded_hit:
                        verdict = "selector"
                    else:
                        verdict = "stitch"
                    rank = next(
                        (
                            position
                            for position, block in enumerate(ranked_blocks)
                            if block in gold_blocks
                        ),
                        None,
                    )
                    per_reference.append(
                        {
                            "in_document": gold["in_document"],
                            "occurrences": gold["occurrences"],
                            "occurrences_capped": gold["occurrences_capped"],
                            "gold_block_count": len(gold_blocks),
                            "selected_hit": selected_hit,
                            "expanded_hit": expanded_hit,
                            "in_packet": in_packet,
                            "verdict": verdict,
                            "rank_in_beam": rank,
                        }
                    )

                name = arm_name(tail, feature_dim, budget)
                arms[name] = {
                    "ceiling": score_for(task, packet.content, references),
                    "per_reference": per_reference,
                    "selected_blocks": list(chosen),
                    "expanded_blocks": list(packet.expanded_blocks),
                    "compiled_prompt_tokens": packet.compiled_prompt_tokens,
                    "compression_fraction": packet.compression_fraction,
                    "span_count": len(packet.spans),
                    "compile_seconds": compile_seconds,
                }

    return {
        "task": task,
        "references": references,
        "prompt_ceiling": score_for(task, prompt_text, references),
        "original_prompt_tokens": len(next(iter(layouts.values())).input_ids),
        "tokenize_layout_seconds": tokenize_layout_seconds,
        "cache": cache,
        "arms": arms,
        "pipeline_seconds": time.perf_counter() - sample_started,
    }


def group(rows: Iterable[dict], keyfn) -> dict:
    result: dict = {}
    for row in rows:
        result.setdefault(keyfn(row), []).append(row)
    return result


def aggregate_block(cases: Sequence[dict], arm: str) -> dict:
    results = [case["arms"][arm] for case in cases]
    references = [
        reference for result in results for reference in result["per_reference"]
    ]
    total = max(1, len(references))
    verdicts = {
        verdict: sum(1 for ref in references if ref["verdict"] == verdict)
        for verdict in VERDICTS
    }
    return {
        "samples": len(cases),
        "references": len(references),
        "ceiling": statistics.mean(result["ceiling"] for result in results),
        "prompt_ceiling": statistics.mean(case["prompt_ceiling"] for case in cases),
        "perfect_ceiling_fraction": sum(
            1 for result in results if result["ceiling"] >= 1.0
        )
        / len(results),
        "compiler_miss_samples": sum(
            1 for result in results if result["ceiling"] < 1.0
        ),
        "in_document": sum(1 for ref in references if ref["in_document"]) / total,
        "selected_hit": sum(1 for ref in references if ref["selected_hit"]) / total,
        "expanded_hit": sum(1 for ref in references if ref["expanded_hit"]) / total,
        "in_packet": sum(1 for ref in references if ref["in_packet"]) / total,
        "verdicts": verdicts,
        "verdict_fractions": {
            verdict: verdicts[verdict] / total for verdict in VERDICTS
        },
        "median_compiled_tokens": statistics.median(
            result["compiled_prompt_tokens"] for result in results
        ),
        "median_compression_fraction": statistics.median(
            result["compression_fraction"] for result in results
        ),
        "median_compile_seconds": statistics.median(
            result["compile_seconds"] for result in results
        ),
    }


def pairwise(cases: Sequence[dict], baseline: str, arm: str) -> dict:
    deltas = [
        case["arms"][arm]["ceiling"] - case["arms"][baseline]["ceiling"]
        for case in cases
    ]
    return {
        "samples": len(deltas),
        "mean_ceiling_delta": statistics.mean(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def cross_tab(
    cases: Sequence[dict], arms: Sequence[str], e2e_rows: dict
) -> dict[str, dict]:
    result = {}
    for arm in arms:
        counts = dict.fromkeys(
            ("ok", "model_side", "compiler_side", "answered_without_evidence"),
            0,
        )
        by_task: dict[str, dict] = {}
        joined = 0
        for case in cases:
            other = e2e_rows.get(
                (case["task"], int(case["length"]), str(case["index"]))
            )
            if other is None:
                continue
            joined += 1
            evidence = case["arms"][arm]["ceiling"] >= 1.0
            correct = other["score_spruce"] >= 1.0
            if evidence and correct:
                label = "ok"
            elif evidence and not correct:
                label = "model_side"
            elif not evidence and correct:
                label = "answered_without_evidence"
            else:
                label = "compiler_side"
            counts[label] += 1
            by_task.setdefault(case["task"], dict.fromkeys(counts, 0))[label] += 1
        wrong = counts["model_side"] + counts["compiler_side"]
        result[arm] = {
            "joined": joined,
            "counts": counts,
            "by_task": by_task,
            "compiler_side_share_of_failures": (
                counts["compiler_side"] / wrong if wrong else 0.0
            ),
        }
    return result


def make_figures(summary: dict, cases: Sequence[dict], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    arms = summary["arms"]
    labels = [arm.upper().replace("_", "/") for arm in arms]
    colors = plt.cm.tab10(np.linspace(0, 1, len(arms)))
    baseline = summary["baseline_arm"]
    lengths = [
        int(value) for value in summary["config"]["lengths"]
        if str(value) in summary["by_length"][baseline]
    ]
    tasks = [
        task for task in summary["config"]["tasks"]
        if task in summary["by_task"][baseline]
    ]
    made: list[str] = []

    def save(fig, name: str) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(
                figure_dir / f"{name}.{suffix}",
                dpi=200 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
        made.append(name)

    # 1. Overall ceiling for every arm.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    values = [summary["overall"][arm]["ceiling"] for arm in arms]
    bars = ax.bar(range(len(arms)), values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="compiler ceiling", ylim=(0, 1.03),
           title="RULER compiler ceiling: cached factorial ablation")
    ax.grid(alpha=0.25, axis="y")
    save(fig, "01_overall_ceiling")

    # 2. Compiler-side joined misses: the direct 376-miss target.
    if summary.get("crosstab"):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        misses = [
            summary["crosstab"][arm]["counts"]["compiler_side"] for arm in arms
        ]
        bars = ax.bar(range(len(arms)), misses, color=colors)
        for bar, value in zip(bars, misses):
            ax.text(bar.get_x() + bar.get_width() / 2, value, str(value),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel="joined wrong answers with evidence absent",
               title="Compiler-side end-to-end failures (lower is better)")
        ax.grid(alpha=0.25, axis="y")
        save(fig, "02_joined_compiler_misses")

    # 3. Ceiling by context length.
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, color in zip(arms, colors):
        ax.plot(
            [length // 1024 for length in lengths],
            [summary["by_length"][arm][str(length)]["ceiling"]
             for length in lengths],
            marker="o",
            linewidth=1.6,
            color=color,
            label=arm.upper().replace("_", "/"),
        )
    ax.set(xlabel="context length (Ki tokens)", ylabel="compiler ceiling",
           ylim=(0, 1.03), title="Compiler ceiling by length")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "03_ceiling_by_length")

    # 4. D=4096 minus D=1024, holding tail and M fixed.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tail in summary["config"]["tails"]:
        for budget in summary["config"]["budgets"]:
            low = arm_name(tail, 1024, budget)
            high = arm_name(tail, 4096, budget)
            if low not in arms or high not in arms:
                continue
            delta = [
                summary["by_length"][high][str(length)]["ceiling"]
                - summary["by_length"][low][str(length)]["ceiling"]
                for length in lengths
            ]
            ax.plot([length // 1024 for length in lengths], delta, marker="o",
                    label=f"tail={tail}, M={budget}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="context length (Ki tokens)",
           ylabel="ceiling delta: D4096 - D1024",
           title="Does the wider sketch help at matched query and budget?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "04_feature_dim_effect")

    # 5. Tail=2 minus tail=3, holding D and M fixed.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for dim in summary["config"]["feature_dims"]:
        for budget in summary["config"]["budgets"]:
            clean = arm_name(2, dim, budget)
            noisy = arm_name(3, dim, budget)
            if clean not in arms or noisy not in arms:
                continue
            delta = [
                summary["by_length"][clean][str(length)]["ceiling"]
                - summary["by_length"][noisy][str(length)]["ceiling"]
                for length in lengths
            ]
            ax.plot([length // 1024 for length in lengths], delta, marker="o",
                    label=f"D={dim}, M={budget}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="context length (Ki tokens)",
           ylabel="ceiling delta: tail2 - tail3",
           title="Effect of removing one random haystack query line")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "05_query_tail_effect")

    # 6. High-budget minus low-budget, holding tail and D fixed.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    low_budget = min(summary["config"]["budgets"])
    high_budget = max(summary["config"]["budgets"])
    for tail in summary["config"]["tails"]:
        for dim in summary["config"]["feature_dims"]:
            narrow = arm_name(tail, dim, low_budget)
            wide = arm_name(tail, dim, high_budget)
            if narrow not in arms or wide not in arms:
                continue
            delta = [
                summary["by_length"][wide][str(length)]["ceiling"]
                - summary["by_length"][narrow][str(length)]["ceiling"]
                for length in lengths
            ]
            ax.plot([length // 1024 for length in lengths], delta, marker="o",
                    label=f"tail={tail}, D={dim}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="context length (Ki tokens)",
           ylabel=f"ceiling delta: M{high_budget} - M{low_budget}",
           title="Effect of increasing the selected-block budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "06_block_budget_effect")

    # 7. Per-task delta against the authored baseline.
    grid = np.zeros((len(tasks), len(arms)))
    for row, task in enumerate(tasks):
        base = summary["by_task"][baseline][task]["ceiling"]
        for col, arm in enumerate(arms):
            grid[row, col] = summary["by_task"][arm][task]["ceiling"] - base
    limit = max(0.01, float(np.nanmax(np.abs(grid))))
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    for row in range(len(tasks)):
        for col in range(len(arms)):
            ax.text(col, row, f"{grid[row, col]:+.2f}", ha="center",
                    va="center", fontsize=6)
    ax.set(title=f"Ceiling delta by task versus {baseline.upper()}")
    fig.colorbar(image, ax=ax, shrink=0.8, label="ceiling delta")
    save(fig, "07_task_delta_heatmap")

    # 8. Median cached index size by D and length.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dim in summary["config"]["feature_dims"]:
        values = []
        for length in lengths:
            rows = [
                case["cache"][f"t{tail}_d{dim}"]["index_bytes"] / 2**20
                for case in cases
                if int(case["length"]) == length
                for tail in summary["config"]["tails"]
            ]
            values.append(statistics.median(rows))
        ax.plot([length // 1024 for length in lengths], values, marker="o",
                linewidth=2, label=f"D={dim}")
    ax.set(xlabel="context length (Ki tokens)", ylabel="median index MiB",
           title="Cached lexical-index storage")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "08_index_storage")

    # 9. Actual cached work versus naive per-arm recomputation.
    cache = summary["cache_accounting"]
    categories = ["tokenize/layout", "index build", "beam traversal"]
    actual = [cache["actual_tokenizations"], cache["actual_index_builds"],
              cache["actual_traversals"]]
    naive = [cache["naive_tokenizations"], cache["naive_index_builds"],
             cache["naive_traversals"]]
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.18, naive, width=0.36, color="#999999", label="naive")
    ax.bar(x + 0.18, actual, width=0.36, color="#d62728", label="cached")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set(ylabel="operations over the completed sample set",
           title="Work eliminated by exact query-independent caching")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    save(fig, "09_cache_work_saved")

    return made


def write_summary_markdown(summary: dict, output_dir: Path) -> None:
    baseline = summary["baseline_arm"]
    lines = [
        "# Cached RULER compiler factorial ablation",
        "",
        f"- generated: {summary['created_utc']}",
        f"- complete: {summary['run_complete']}",
        f"- samples: {summary['sample_count']}",
        f"- baseline: `{baseline}`",
        "- model weights: none (tokenizer and compiler only)",
        "",
        "## Overall",
        "",
        "| arm | ceiling | perfect samples | selector misses | packet % | vs baseline | wins/losses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        block = summary["overall"][arm]
        paired = summary["pairwise_vs_baseline"][arm]
        lines.append(
            "| `%s` | %.4f | %.1f%% | %d | %.2f%% | %+.4f | %d/%d |"
            % (
                arm,
                block["ceiling"],
                100 * block["perfect_ceiling_fraction"],
                block["verdicts"]["selector"],
                100 * block["median_compression_fraction"],
                paired["mean_ceiling_delta"],
                paired["wins"],
                paired["losses"],
            )
        )

    if summary.get("crosstab"):
        lines += [
            "",
            "## Joined end-to-end attribution",
            "",
            "| arm | joined | compiler-side misses | model-side misses | compiler share of wrong |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm in summary["arms"]:
            row = summary["crosstab"][arm]
            lines.append(
                "| `%s` | %d | %d | %d | %.1f%% |"
                % (
                    arm,
                    row["joined"],
                    row["counts"]["compiler_side"],
                    row["counts"]["model_side"],
                    100 * row["compiler_side_share_of_failures"],
                )
            )

    cache = summary["cache_accounting"]
    lines += [
        "",
        "## Exact caching",
        "",
        "- tokenizations/layouts: %d actual versus %d naive" % (
            cache["actual_tokenizations"], cache["naive_tokenizations"]),
        "- index builds: %d actual versus %d naive" % (
            cache["actual_index_builds"], cache["naive_index_builds"]),
        "- beam traversals: %d actual versus %d naive" % (
            cache["actual_traversals"], cache["naive_traversals"]),
        "- The low-M route is the asserted prefix of the cached high-M ranking "
        "for every (tail,D) configuration.",
        "- baseline comparisons against ruler_compiler_only_1024: %d checked, "
        "%d mismatches" % (
            summary["baseline_validation"]["checked"],
            summary["baseline_validation"]["mismatches"],
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- Compare D only while tail and M are fixed.",
        "- Compare M only while tail and D are fixed.",
        "- `cwe` and `fwe` common-word ceilings are sanity checks, not retrieval evidence.",
        "- This is compiler-only; it cannot clear end-to-end RULER or Stage-3 KS1.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--e2e-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--tails", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--feature-dims", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 9])
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument("--boundary", default="paragraph")
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--max-occurrences", type=int, default=256)
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument("--original-context", type=int, default=32768)
    parser.add_argument("--session-budget-minutes", type=float, default=90.0)
    parser.add_argument("--archive-sha256", default="unknown")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.tails)) != len(args.tails):
        raise SystemExit("--tails contains duplicates")
    if len(set(args.feature_dims)) != len(args.feature_dims):
        raise SystemExit("--feature-dims contains duplicates")
    if len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("--budgets contains duplicates")
    if max(args.budgets) > args.beam:
        raise SystemExit("max budget must be <= beam")
    if not set(args.tasks) <= set(TASK_TYPES):
        raise SystemExit(f"unknown tasks: {sorted(set(args.tasks) - set(TASK_TYPES))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = args.output_dir / "task_reports"
    task_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        arm_name(tail, dim, budget)
        for tail in args.tails
        for dim in args.feature_dims
        for budget in args.budgets
    ]
    baseline_arm = arm_name(3, 1024, 4)
    if baseline_arm not in arms:
        raise SystemExit("factorial must include the authored t3_d1024_m4 baseline")

    config = {
        "model": args.model,
        "tasks": args.tasks,
        "lengths": args.lengths,
        "num_samples": args.num_samples,
        "tails": args.tails,
        "feature_dims": args.feature_dims,
        "budgets": args.budgets,
        "beam": args.beam,
        "block_size": args.block_size,
        "block_radius": args.block_radius,
        "boundary": args.boundary,
        "unigram_fraction": args.unigram_fraction,
        "idf_power": args.idf_power,
        "radix": args.radix,
        "max_occurrences": args.max_occurrences,
        "yarn_factor": args.yarn_factor,
        "original_context": args.original_context,
    }
    fingerprint = config_fingerprint(config)

    generated = {
        (task, length): args.data_dir / str(length) / task / "validation.jsonl"
        for length in args.lengths
        for task in args.tasks
    }
    missing = [key for key, path in generated.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} generated RULER files: {missing[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer,
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_context,
    )
    probe_key = sorted(generated, key=lambda key: (key[1], key[0]))[0]
    probe = json.loads(
        next(
            line
            for line in generated[probe_key].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    validate_cached_layouts(tokenizer, probe, args.tails)
    print("cached single-tokenization layouts match production locate_prompt_layout")

    # Validate the complete cached execution path before touching a report.
    # This proves that taking the low-M prefix of one high-M traversal is exact
    # for every (tail,D) arm, even when a later session merely resumes outputs.
    prefix_validation: set[tuple[int, int]] = set()
    run_case(
        probe,
        probe_key[0],
        tokenizer,
        tails=args.tails,
        feature_dims=args.feature_dims,
        budgets=args.budgets,
        block_size=args.block_size,
        block_radius=args.block_radius,
        boundary=args.boundary,
        beam=args.beam,
        unigram_fraction=args.unigram_fraction,
        idf_power=args.idf_power,
        radix=args.radix,
        max_occurrences=args.max_occurrences,
        prefix_validation=prefix_validation,
    )
    expected_prefix_validations = len(args.tails) * len(args.feature_dims)
    if len(prefix_validation) != expected_prefix_validations:
        raise AssertionError("not every (tail,D) prefix cache was validated")
    print(f"validated {len(prefix_validation)} cached M-prefix configurations")

    baseline_rows = load_baseline_cases(args.baseline_dir)
    e2e_rows = load_e2e_cases(args.e2e_dir)
    print(f"baseline rows available: {len(baseline_rows)}")
    print(f"end-to-end rows available: {len(e2e_rows)}")

    started = time.perf_counter()
    budget_hit = False

    def budget_spent() -> bool:
        return bool(args.session_budget_minutes) and (
            (time.perf_counter() - started) / 60 >= args.session_budget_minutes
        )

    for length in args.lengths:
        if budget_hit:
            break
        print(f"=== {length} tokens ===", flush=True)
        for task in args.tasks:
            if budget_hit:
                break
            out_path = task_dir / f"{task}_{length}.json"
            report = None
            if out_path.is_file():
                report = json.loads(out_path.read_text(encoding="utf-8"))
                if report.get("config_fingerprint") != fingerprint:
                    raise RuntimeError(
                        f"refusing to mix configurations in {out_path}; use a new output dir"
                    )
                if report.get("status") == "completed" and not args.rerun_completed:
                    print(f"  resume complete: {task}")
                    continue
            if report is None or args.rerun_completed:
                report = {
                    "task": task,
                    "length": int(length),
                    "config": config,
                    "config_fingerprint": fingerprint,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "cases": [],
                    "errors": [],
                    "status": "running",
                }
            done = {str(case["index"]) for case in report["cases"]}
            samples = [
                json.loads(line)
                for line in generated[(task, length)]
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ][: args.num_samples]
            for position, sample in enumerate(samples):
                key = str(sample.get("index", position))
                if key in done:
                    continue
                if budget_spent():
                    budget_hit = True
                    break
                try:
                    case = run_case(
                        sample,
                        task,
                        tokenizer,
                        tails=args.tails,
                        feature_dims=args.feature_dims,
                        budgets=args.budgets,
                        block_size=args.block_size,
                        block_radius=args.block_radius,
                        boundary=args.boundary,
                        beam=args.beam,
                        unigram_fraction=args.unigram_fraction,
                        idf_power=args.idf_power,
                        radix=args.radix,
                        max_occurrences=args.max_occurrences,
                        prefix_validation=prefix_validation,
                    )
                    old = baseline_rows.get((task, int(length), key))
                    case["baseline_checked"] = old is not None
                    case["baseline_match"] = None
                    if old is not None:
                        current = case["arms"][baseline_arm]
                        matches = (
                            math.isclose(
                                current["ceiling"], old["ceiling"], abs_tol=1e-12
                            )
                            and current["selected_blocks"] == old["selected_blocks"]
                            and current["expanded_blocks"] == old["expanded_blocks"]
                        )
                        case["baseline_match"] = bool(matches)
                        if not matches:
                            raise AssertionError(
                                f"cached baseline differs from compiler-only v1 for {task}/{length}/{key}"
                            )
                except Exception as error:  # noqa: BLE001
                    report["errors"].append(
                        {
                            "task": task,
                            "length": int(length),
                            "index": key,
                            "error": repr(error),
                        }
                    )
                    atomic_json(out_path, report)
                    traceback.print_exc()
                    done.add(key)
                    continue
                case["index"] = key
                case["length"] = int(length)
                report["cases"].append(case)
                done.add(key)
                atomic_json(out_path, report)
            if budget_hit:
                atomic_json(out_path, report)
                break
            report["status"] = "completed"
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(out_path, report)
            if report["cases"]:
                base = statistics.mean(
                    case["arms"][baseline_arm]["ceiling"]
                    for case in report["cases"]
                )
                best = max(
                    statistics.mean(case["arms"][arm]["ceiling"] for case in report["cases"])
                    for arm in arms
                )
                print(f"  {task:16s} baseline {base:.3f}  best {best:.3f}", flush=True)

    all_cases = []
    errors = []
    completed_pairs = set()
    for report_path in sorted(task_dir.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("config_fingerprint") != fingerprint:
            continue
        pair = (report["task"], int(report["length"]))
        if pair not in generated:
            continue
        all_cases.extend(report.get("cases", []))
        errors.extend(report.get("errors", []))
        if report.get("status") == "completed":
            completed_pairs.add(pair)
    if not all_cases:
        raise RuntimeError("no completed cases to aggregate")
    run_complete = completed_pairs >= set(generated)
    baseline_checked = sum(
        bool(case.get("baseline_checked")) for case in all_cases
    )
    baseline_mismatches = sum(
        case.get("baseline_match") is False for case in all_cases
    )

    overall = {arm: aggregate_block(all_cases, arm) for arm in arms}
    by_length = {
        arm: {
            str(length): aggregate_block(rows, arm)
            for length, rows in sorted(group(all_cases, lambda row: int(row["length"])).items())
        }
        for arm in arms
    }
    by_task = {
        arm: {
            task: aggregate_block(rows, arm)
            for task, rows in sorted(group(all_cases, lambda row: row["task"]).items())
        }
        for arm in arms
    }
    paired = {arm: pairwise(all_cases, baseline_arm, arm) for arm in arms}
    crosstab = cross_tab(all_cases, arms, e2e_rows) if e2e_rows else {}
    sample_count = len(all_cases)
    actual_index_builds = sample_count * len(args.tails) * len(args.feature_dims)
    actual_traversals = actual_index_builds
    naive_ops = sample_count * len(arms)
    cache_accounting = {
        "actual_tokenizations": sample_count,
        "naive_tokenizations": naive_ops,
        "actual_index_builds": actual_index_builds,
        "naive_index_builds": naive_ops,
        "actual_traversals": actual_traversals,
        "naive_traversals": naive_ops,
        "actual_compiles": naive_ops,
        "median_pipeline_seconds": statistics.median(
            case["pipeline_seconds"] for case in all_cases
        ),
    }

    summary = {
        "kind": "spruce_ruler_cached_factorial_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": args.archive_sha256,
        "config": config,
        "config_fingerprint": fingerprint,
        "arms": arms,
        "baseline_arm": baseline_arm,
        "run_complete": run_complete,
        "completed_pairs": len(completed_pairs),
        "expected_pairs": len(generated),
        "sample_count": sample_count,
        "wall_clock_minutes": (time.perf_counter() - started) / 60,
        "error_count": len(errors),
        "errors": errors[:200],
        "baseline_validation": {
            "available_rows": len(baseline_rows),
            "checked": baseline_checked,
            "mismatches": baseline_mismatches,
            "prefix_configurations_validated": sorted(
                [list(value) for value in prefix_validation]
            ),
        },
        "cache_accounting": cache_accounting,
        "overall": overall,
        "by_length": by_length,
        "by_task": by_task,
        "pairwise_vs_baseline": paired,
        "crosstab": crosstab,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    figures = make_figures(summary, all_cases, args.output_dir)
    summary["figures"] = figures
    atomic_json(args.output_dir / "summary.json", summary)
    write_summary_markdown(summary, args.output_dir)
    run_record = {
        key: summary[key]
        for key in (
            "kind",
            "created_utc",
            "archive_sha256",
            "config",
            "config_fingerprint",
            "arms",
            "baseline_arm",
            "run_complete",
            "completed_pairs",
            "expected_pairs",
            "sample_count",
            "wall_clock_minutes",
            "error_count",
            "baseline_validation",
            "cache_accounting",
            "overall",
            "pairwise_vs_baseline",
            "crosstab",
            "figures",
        )
    }
    atomic_json(args.output_dir / "run_record.json", run_record)

    print("")
    print(f"{sample_count} paired samples | {len(completed_pairs)}/{len(generated)} pairs")
    print(f"baseline checks: {baseline_checked}, mismatches: {baseline_mismatches}")
    print(f"figures: {len(figures)}")
    print("")
    print(f"{'arm':18s} {'ceiling':>8s} {'delta':>8s} {'wins':>6s} {'loss':>6s} {'compiler':>9s}")
    for arm in arms:
        compiler = (
            crosstab[arm]["counts"]["compiler_side"] if crosstab else -1
        )
        print(
            f"{arm:18s} {overall[arm]['ceiling']:8.4f} "
            f"{paired[arm]['mean_ceiling_delta']:+8.4f} "
            f"{paired[arm]['wins']:6d} {paired[arm]['losses']:6d} {compiler:9d}"
        )
    if not run_complete:
        print("PARTIAL RUN: rerun the command to resume before quoting a result.")
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(f"markdown: {args.output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
