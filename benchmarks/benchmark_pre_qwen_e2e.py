"""Matched dense-vs-compiled Qwen benchmark with every live request cost.

Dense request:
  tokenize full prompt -> transfer -> dense prefill -> dense decode

Compiled request:
  tokenize full prompt with offsets -> build lexical hierarchy -> encode query
  -> top-down select -> stitch exact source text -> tokenize compact prompt
  -> transfer -> dense prefill -> dense decode

Model loading, manifest loading, and one backend warm-up are excluded from both
paths.  Everything needed after an in-memory request string arrives is charged.
The compiled path conservatively rebuilds its document index on every repeat;
no cross-request cache is assumed.
"""
import argparse
import gc
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

from benchmarks.compare_dense_sparse import (
    _free_model,
    _generate,
    _load_model,
    _warmup,
    runtime_metadata,
)
from configs.long_context import (
    QWEN_NATIVE_CONTEXT,
    configure_tokenizer,
    context_limit,
    yarn_metadata,
)
from eval.score import score_retrieval
from interfaces.evidence_compiler import (
    compile_evidence_packet_from_layout,
    locate_prompt_layout,
)
from scripts.extract_teacher_targets import load_verified_manifest
from selector.pre_qwen import (
    build_pre_qwen_index,
    question_feature_weights,
    select_pre_qwen_blocks,
)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _median(samples, key):
    return float(statistics.median(float(sample[key]) for sample in samples))


def _peak_memory(device):
    device = torch.device(device)
    if device.type != "cuda":
        return {
            "peak_memory_allocated_gb": 0.0,
            "peak_memory_reserved_gb": 0.0,
        }
    return {
        "peak_memory_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9),
        "peak_memory_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1e9),
    }


def _transfer_inputs(encoded, device):
    started = time.perf_counter()
    inputs = {
        name: value.to(device, non_blocking=False)
        for name, value in encoded.items()
    }
    _sync(device)
    return inputs, time.perf_counter() - started


def _model_read(
        model, tokenizer, encoded, target, *,
        max_new_tokens, request_started, preprocessing):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inputs, transfer_seconds = _transfer_inputs(encoded, device)
    token_ids, model_timing = _generate(
        model, inputs, max_new_tokens=max_new_tokens)
    _sync(device)
    request_seconds = time.perf_counter() - request_started
    answer = tokenizer.decode(token_ids, skip_special_tokens=True)
    score = score_retrieval(
        answer, target["needle"],
        reference_answers=target.get(
            "reference_answers", target.get("answers")))
    return {
        **preprocessing,
        "input_transfer_seconds": transfer_seconds,
        "prefill_seconds": float(model_timing["prefill_seconds"]),
        "decode_seconds": float(model_timing["decode_seconds"]),
        "model_seconds": float(model_timing["seconds"]),
        "request_seconds": float(request_seconds),
        "answer": answer,
        "exact": bool(score["exact"]),
        "fuzzy": float(score["fuzzy"]),
        **_peak_memory(device),
    }


def run_dense_request(
        model, tokenizer, target, *, max_new_tokens):
    """Measure the complete ordinary dense path for one in-memory prompt."""
    request_started = time.perf_counter()
    started = time.perf_counter()
    encoded = tokenizer(target["prompt_text"], return_tensors="pt")
    tokenize_seconds = time.perf_counter() - started
    return _model_read(
        model, tokenizer, encoded, target,
        max_new_tokens=max_new_tokens,
        request_started=request_started,
        preprocessing={
            "tokenize_seconds": tokenize_seconds,
            "input_tokens": int(encoded["input_ids"].shape[1]),
        },
    )


def run_compiled_request(
        model, tokenizer, target, *,
        candidate_blocks,
        block_radius,
        boundary,
        beam,
        feature_dim,
        unigram_fraction,
        idf_power,
        radix,
        max_new_tokens):
    """Measure the complete live lexical-select-and-dense-read path."""
    request_started = time.perf_counter()
    prompt = target["prompt_text"]
    user_prompt = target.get("user_prompt_text")
    if not isinstance(user_prompt, str) or not user_prompt:
        if target.get("prompt_format") == "qwen_chat_v1":
            raise ValueError("chat prompt is missing user_prompt_text")
        user_prompt = prompt
    question = target["question"]

    started = time.perf_counter()
    layout = locate_prompt_layout(
        tokenizer, prompt, user_prompt, question)
    layout_tokenize_seconds = time.perf_counter() - started

    started = time.perf_counter()
    index = build_pre_qwen_index(
        layout, int(target["block_size"]),
        feature_dim=feature_dim,
        unigram_fraction=unigram_fraction,
        radix=radix)
    index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    weights = question_feature_weights(
        tokenizer, question, index, idf_power=idf_power)
    selection = select_pre_qwen_blocks(
        index, weights, top_m=candidate_blocks,
        beam=beam, radix=radix)
    selection_seconds = time.perf_counter() - started

    started = time.perf_counter()
    packet = compile_evidence_packet_from_layout(
        tokenizer, prompt, layout, question, selection.blocks,
        int(target["block_size"]),
        block_radius=block_radius, boundary=boundary)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    encoded = tokenizer(packet.prompt, return_tensors="pt")
    compact_tokenize_seconds = time.perf_counter() - started
    needle_block = int(target["needle_block"])
    result = _model_read(
        model, tokenizer, encoded, target,
        max_new_tokens=max_new_tokens,
        request_started=request_started,
        preprocessing={
            "layout_tokenize_seconds": layout_tokenize_seconds,
            "index_seconds": index_seconds,
            "selection_seconds": selection_seconds,
            "compile_seconds": compile_seconds,
            "compact_tokenize_seconds": compact_tokenize_seconds,
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "original_input_tokens": len(layout.input_ids),
            "compression_fraction": packet.compression_fraction,
            "selected_blocks": list(selection.blocks),
            "expanded_blocks": list(packet.expanded_blocks),
            "selected_contains_needle": (
                needle_block in selection.blocks),
            "expanded_contains_needle": (
                needle_block in packet.expanded_blocks),
            "tree_nodes": index.node_count,
            "tree_levels": len(index.levels),
            "visited_nodes": selection.visited_nodes,
            "final_candidates": selection.final_candidates,
            "indexed_tokens": index.indexed_tokens,
        },
    )
    return result


def _summarize_samples(samples):
    timing_keys = [
        key for key, value in samples[0].items()
        if key.endswith("_seconds") and isinstance(value, (int, float))
    ]
    summary = {
        key: _median(samples, key) for key in timing_keys
    }
    summary.update({
        "input_tokens": int(statistics.median(
            int(sample["input_tokens"]) for sample in samples)),
        "answer": samples[0]["answer"],
        "answer_repeat_match": (
            len(set(sample["answer"].strip() for sample in samples)) == 1),
        "exact": bool(samples[0]["exact"]),
        "fuzzy": float(samples[0]["fuzzy"]),
        "peak_memory_allocated_gb": _median(
            samples, "peak_memory_allocated_gb"),
        "peak_memory_reserved_gb": _median(
            samples, "peak_memory_reserved_gb"),
        "samples": samples,
    })
    for key in (
            "original_input_tokens", "compression_fraction",
            "selected_blocks", "expanded_blocks",
            "selected_contains_needle", "expanded_contains_needle",
            "tree_nodes", "tree_levels", "visited_nodes",
            "final_candidates", "indexed_tokens"):
        if key in samples[0]:
            summary[key] = samples[0][key]
    return summary


def aggregate_cases(cases):
    """Compute accuracy and sum-weighted fully charged speed."""
    dense_seconds = sum(
        float(case["dense"]["request_seconds"]) for case in cases)
    compiled_seconds = sum(
        float(case["compiled"]["request_seconds"]) for case in cases)
    lengths = {}
    for case in cases:
        lengths.setdefault(str(case["requested_length"]), []).append(case)

    def accuracy(rows, mode):
        exact = sum(int(row[mode]["exact"]) for row in rows)
        return {
            "cases": len(rows),
            "exact_count": exact,
            "exact_rate": exact / max(1, len(rows)),
        }

    return {
        "cases": len(cases),
        "dense": {
            **accuracy(cases, "dense"),
            "sum_request_seconds": dense_seconds,
            "median_request_seconds": statistics.median(
                case["dense"]["request_seconds"] for case in cases),
            "median_prefill_seconds": statistics.median(
                case["dense"]["prefill_seconds"] for case in cases),
        },
        "compiled": {
            **accuracy(cases, "compiled"),
            "sum_request_seconds": compiled_seconds,
            "median_request_seconds": statistics.median(
                case["compiled"]["request_seconds"] for case in cases),
            "median_prefill_seconds": statistics.median(
                case["compiled"]["prefill_seconds"] for case in cases),
            "median_input_tokens": statistics.median(
                case["compiled"]["input_tokens"] for case in cases),
            "median_compression_fraction": statistics.median(
                case["compiled"]["compression_fraction"] for case in cases),
            "needle_block_recall": (
                sum(int(case["compiled"]["selected_contains_needle"])
                    for case in cases) / max(1, len(cases))),
            "expanded_needle_recall": (
                sum(int(case["compiled"]["expanded_contains_needle"])
                    for case in cases) / max(1, len(cases))),
        },
        "by_requested_length": {
            length: {
                "dense": accuracy(rows, "dense"),
                "compiled": accuracy(rows, "compiled"),
                "sum_dense_request_seconds": sum(
                    row["dense"]["request_seconds"] for row in rows),
                "sum_compiled_request_seconds": sum(
                    row["compiled"]["request_seconds"] for row in rows),
            }
            for length, rows in sorted(lengths.items())
        },
        "sum_weighted_speedup": (
            dense_seconds / compiled_seconds
            if compiled_seconds > 0 else float("inf")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"),
        default="paragraph")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--expected-cases", type=int, default=25)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"),
        default="auto")
    parser.add_argument("--yarn-factor", type=float, default=1.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_pre_qwen_e2e_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the matched Qwen benchmark")
    if args.candidate_blocks < 1 or args.block_radius < 0:
        raise SystemExit("invalid candidate-blocks/block-radius")
    if args.beam < args.candidate_blocks:
        raise SystemExit("--beam must be >= --candidate-blocks")
    if args.feature_dim < 16 or args.radix < 2:
        raise SystemExit("invalid feature-dim/radix")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise SystemExit("--repeats and --max-new-tokens must be >= 1")

    records = load_verified_manifest(args.manifest)
    if args.max_cases is not None:
        records = records[:args.max_cases]
    if args.expected_cases > 0 and len(records) != args.expected_cases:
        raise SystemExit(
            f"expected exactly {args.expected_cases} accepted prompts, "
            f"found {len(records)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
    for record in records:
        if int(record["seq_len"]) > maximum_context:
            raise SystemExit(
                f"{record['id']} has {record['seq_len']} tokens, exceeding "
                f"configured context {maximum_context}")

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = _load_model(
        args.model, "sdpa", dtype, args.load_offload_dir,
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
    _warmup(
        model, tokenizer(
            "Read the supplied document and answer briefly.",
            return_tensors="pt"))

    cases = []
    try:
        for case_index, target in enumerate(records, start=1):
            print(
                f"[{case_index}/{len(records)}] {target['id']}",
                flush=True)
            dense_samples = []
            compiled_samples = []
            for repeat in range(args.repeats):
                # Alternate order to limit systematic thermal/cache bias.
                modes = (
                    ("dense", "compiled")
                    if (case_index + repeat) % 2
                    else ("compiled", "dense")
                )
                for mode in modes:
                    if mode == "dense":
                        sample = run_dense_request(
                            model, tokenizer, target,
                            max_new_tokens=args.max_new_tokens)
                        dense_samples.append(sample)
                    else:
                        sample = run_compiled_request(
                            model, tokenizer, target,
                            candidate_blocks=args.candidate_blocks,
                            block_radius=args.block_radius,
                            boundary=args.boundary,
                            beam=args.beam,
                            feature_dim=args.feature_dim,
                            unigram_fraction=args.unigram_fraction,
                            idf_power=args.idf_power,
                            radix=args.radix,
                            max_new_tokens=args.max_new_tokens)
                        compiled_samples.append(sample)
            dense = _summarize_samples(dense_samples)
            compiled = _summarize_samples(compiled_samples)
            cases.append({
                "candidate_id": target["id"],
                "source_case_id": target.get("source_case_id"),
                "requested_length": int(target["requested_length"]),
                "seq_len": int(target["seq_len"]),
                "needle_block": int(target["needle_block"]),
                "dense": dense,
                "compiled": compiled,
            })
            print(
                f"  dense exact={int(dense['exact'])} "
                f"request={dense['request_seconds']:.4f}s; "
                f"compiled exact={int(compiled['exact'])} "
                f"request={compiled['request_seconds']:.4f}s "
                f"tokens={compiled['input_tokens']} "
                f"evidence={int(compiled['expanded_contains_needle'])}",
                flush=True)
            gc.collect()
    finally:
        model = _free_model(model)

    aggregate = aggregate_cases(cases)
    compiled_exact = aggregate["compiled"]["exact_count"]
    long_bucket = aggregate["by_requested_length"].get("32768", {})
    exact_32k = long_bucket.get("compiled", {}).get("exact_count", 0)
    cases_32k = long_bucket.get("compiled", {}).get("cases", 0)
    accuracy_gate = (
        len(cases) == 25
        and compiled_exact >= 24
        and cases_32k == 11
        and exact_32k == 11
    )
    speed_gate = aggregate["sum_weighted_speedup"] > 1.0
    report = {
        "kind": "spruce_pre_qwen_e2e_v1",
        "timing_scope": {
            "input_state": "prompt and question strings already in memory",
            "excluded": [
                "manifest file read",
                "model/tokenizer load",
                "one backend warm-up",
            ],
            "dense_charged": [
                "full-prompt tokenization",
                "host-to-device input transfer",
                "dense prefill",
                "dense decode",
            ],
            "compiled_charged": [
                "full-prompt offset tokenization",
                "lexical hierarchy construction",
                "question feature construction",
                "top-down tree traversal",
                "exact-text span expansion and stitching",
                "compact-prompt tokenization",
                "host-to-device input transfer",
                "dense prefill",
                "dense decode",
            ],
            "index_cache": "disabled; rebuilt on every measured repeat",
        },
        "model": args.model,
        "runtime": runtime_metadata(),
        "manifest": os.path.abspath(args.manifest),
        "config": {
            "candidate_blocks": args.candidate_blocks,
            "block_radius": args.block_radius,
            "boundary": args.boundary,
            "beam": args.beam,
            "feature_dim": args.feature_dim,
            "unigram_fraction": args.unigram_fraction,
            "idf_power": args.idf_power,
            "radix": args.radix,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
        },
        "rope": yarn_metadata(
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=(
                args.original_max_position_embeddings)),
        "cases": cases,
        "aggregate": aggregate,
        "accuracy_gate_passed": accuracy_gate,
        "speed_gate_passed": speed_gate,
        "deployability_gate_passed": accuracy_gate and speed_gate,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "aggregate": aggregate,
        "accuracy_gate_passed": accuracy_gate,
        "speed_gate_passed": speed_gate,
        "deployability_gate_passed": accuracy_gate and speed_gate,
    }, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
