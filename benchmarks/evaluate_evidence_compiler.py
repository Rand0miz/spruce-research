"""Evaluate select -> stitch original text -> fresh dense Qwen reading.

This diagnostic does not modify Qwen attention. It reconstructs each saved
prompt, turns evidence block IDs back into exact source text, repairs paragraph
boundaries, and runs an ordinary dense pass over the compact packet.

Modes:
  oracle  Known evidence block. Attribution ceiling; not deployable.
  flat    Gate scores every document leaf at the reader row. Selection ceiling
          with O(n) flat scanning; not SPRUCE's logarithmic selector.
  tree    Recursive SPRUCE traversal, followed by reranking only its leaf union.

Saved teacher features are used as selector inputs, so their one-time
extraction is not represented by feature-load timing. Do not treat this
diagnostic as an end-to-end deployment speed benchmark.
"""
import argparse
import gc
import glob
import json
import math
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
from eval.score import score_retrival
from interfaces.evidence_compiler import (
    compile_evidence_packet,
    document_block_ids,
    locate_prompt_layout,
)
from scripts.eval_tree_traversal import load_gate, traverse_to_leaf_ids
from selector.evidence import rank_reader_candidate_blocks
from selector.targets import load_selector_features
from selector.tree import build_key_tree
from teacher.prompt_replay import reconstruct_teacher_prompt


MODES = ("oracle", "flat", "tree")


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = (
            sorted(glob.glob(pattern))
            if any(character in pattern for character in "*?[")
            else [pattern]
        )
        paths.extend(matches)
    return list(dict.fromkeys(
        os.path.abspath(path) for path in paths if os.path.isfile(path)))


def _sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(samples, key):
    return float(statistics.median(sample[key] for sample in samples))


def _user_prompt_text(prompt, target):
    user_prompt = target.get("user_prompt_text")
    if isinstance(user_prompt, str) and user_prompt:
        return user_prompt
    if target.get("prompt_format") == "qwen_chat_v1":
        raise ValueError(
            "chat-formatted target is missing user_prompt_text")
    return prompt


@torch.no_grad()
def flat_reader_candidates(
        gate, features, allowed_blocks, top_m):
    """O(n) reader-row selection ceiling used only for attribution."""
    started = time.perf_counter()
    reader_scores = gate(
        features["q_feat"][:, :, -1:],
        features["k_feat"],
    )
    blocks, scores = rank_reader_candidate_blocks(
        reader_scores, top_m=top_m, allowed_blocks=allowed_blocks)
    _sync(features["q_feat"].device)
    elapsed = time.perf_counter() - started
    del reader_scores
    return blocks, scores, {
        "selection_seconds": elapsed,
        "selection_complexity": "flat_O(n)_diagnostic_ceiling",
        "tree_candidate_union": None,
    }


@torch.no_grad()
def tree_reader_candidates(
        gate, features, allowed_blocks, top_m, *,
        beam=8, radix=2, layer_chunk=4):
    """Recursive traversal plus bounded reranking of its reader leaf union."""
    started = time.perf_counter()
    levels = build_key_tree(features["k_feat"], radix=radix)
    selected_ids = traverse_to_leaf_ids(
        gate, features["q_feat"], levels, beam=beam, radix=radix,
        layer_chunk=layer_chunk)
    reader_ids = selected_ids[:, :, -1, :].reshape(-1)
    reader_ids = reader_ids[reader_ids >= 0].unique(sorted=True)
    allowed_set = set(int(block) for block in allowed_blocks)
    candidates = [
        int(block) for block in reader_ids.detach().cpu().tolist()
        if int(block) in allowed_set
    ]
    if not candidates:
        raise ValueError(
            "recursive traversal returned no source-document reader blocks")
    candidate_tensor = torch.tensor(
        candidates, dtype=torch.long, device=features["k_feat"].device)
    candidate_keys = features["k_feat"].index_select(2, candidate_tensor)
    reader_scores = gate(
        features["q_feat"][:, :, -1:],
        candidate_keys,
    )
    blocks, scores = rank_reader_candidate_blocks(
        reader_scores, top_m=top_m, allowed_blocks=allowed_blocks,
        block_ids=candidates)
    _sync(features["q_feat"].device)
    elapsed = time.perf_counter() - started
    del levels, selected_ids, reader_ids, candidate_tensor
    del candidate_keys, reader_scores
    return blocks, scores, {
        "selection_seconds": elapsed,
        "selection_complexity": "recursive_O(log_n)_plus_bounded_rerank",
        "tree_candidate_union": len(candidates),
    }


def _run_dense_packet(
        model, tokenizer, packet, target, *,
        repeats, max_new_tokens):
    encoded = tokenizer(packet.prompt, return_tensors="pt")
    samples = []
    answers = []
    device = next(model.parameters()).device
    for _repeat in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        token_ids, timing = _generate(
            model, encoded, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(token_ids, skip_special_tokens=True)
        answers.append(answer)
        sample = dict(timing)
        if device.type == "cuda":
            sample["peak_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1e9)
            sample["peak_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved(device) / 1e9)
        else:
            sample["peak_memory_allocated_gb"] = 0.0
            sample["peak_memory_reserved_gb"] = 0.0
        samples.append(sample)
    score = score_retrival(
        answers[0], target["needle"],
        reference_answers=target.get("reference_answers"))
    return {
        "answer": answers[0],
        "answer_repeat_match": (
            len(set(answer.strip() for answer in answers)) == 1),
        "exact": bool(score["exact"]),
        "fuzzy": float(score["fuzzy"]),
        "prefill_seconds": _median(samples, "prefill_seconds"),
        "decode_seconds": _median(samples, "decode_seconds"),
        "seconds": _median(samples, "seconds"),
        "peak_memory_allocated_gb": _median(
            samples, "peak_memory_allocated_gb"),
        "peak_memory_reserved_gb": _median(
            samples, "peak_memory_reserved_gb"),
        "timing_samples": samples,
    }


def _mode_aggregate(cases, mode):
    rows = [case["modes"][mode] for case in cases]
    lengths = {}
    for case, row in zip(cases, rows):
        bucket = str(case["requested_length"])
        lengths.setdefault(bucket, []).append(row)
    return {
        "cases": len(rows),
        "exact_count": sum(int(row["exact"]) for row in rows),
        "exact_rate": (
            sum(int(row["exact"]) for row in rows) / max(1, len(rows))),
        "by_requested_length": {
            length: {
                "cases": len(items),
                "exact_count": sum(int(item["exact"]) for item in items),
                "exact_rate": (
                    sum(int(item["exact"]) for item in items)
                    / max(1, len(items))),
            }
            for length, items in sorted(lengths.items())
        },
        "median_compiled_prompt_tokens": statistics.median(
            row["packet"]["compiled_prompt_tokens"] for row in rows),
        "median_compression_fraction": statistics.median(
            row["packet"]["compression_fraction"] for row in rows),
        "median_prefill_seconds": statistics.median(
            row["prefill_seconds"] for row in rows),
        "needle_block_recall": (
            sum(int(row["selected_contains_needle"]) for row in rows)
            / max(1, len(rows))),
        "expanded_needle_recall": (
            sum(int(row["expanded_contains_needle"]) for row in rows)
            / max(1, len(rows))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--prompt-bank")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"), default="paragraph")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--yarn-factor", type=float, default=1.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_evidence_compiler_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the dense evidence-reader run")
    if args.candidate_blocks < 1:
        raise SystemExit("--candidate-blocks must be >= 1")
    if args.block_radius < 0:
        raise SystemExit("--block-radius must be >= 0")
    if args.beam < 1 or args.radix < 2 or args.selector_layer_chunk < 1:
        raise SystemExit("invalid beam/radix/selector-layer-chunk")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise SystemExit("--repeats and --max-new-tokens must be >= 1")

    paths = expand_paths(args.targets)
    if args.max_cases is not None:
        paths = paths[:args.max_cases]
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    gate, gate_config = load_gate(args.gate, args.selector_device)
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = _load_model(
        args.model, "sdpa", dtype, args.load_offload_dir,
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    warmup_prompt = tokenizer(
        "Read the evidence and answer briefly.",
        return_tensors="pt")
    _warmup(model, warmup_prompt)

    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    case_results = []
    for case_index, path in enumerate(paths, start=1):
        print(
            f"[{case_index}/{len(paths)}] {os.path.basename(path)}",
            flush=True)
        prompt, target = reconstruct_teacher_prompt(
            tokenizer, path, bank_path=args.prompt_bank)
        user_prompt = _user_prompt_text(prompt, target)
        question = target["question"]
        layout = locate_prompt_layout(
            tokenizer, prompt, user_prompt, question)
        if len(layout.input_ids) != int(target["seq_len"]):
            raise ValueError(
                f"{path}: offset tokenization has {len(layout.input_ids)} "
                f"tokens, expected {target['seq_len']}")
        block_size = int(target["block_size"])
        allowed_blocks = document_block_ids(layout, block_size)

        feature_started = time.perf_counter()
        features = load_selector_features(
            path, device=args.selector_device)
        _sync(args.selector_device)
        feature_load_seconds = time.perf_counter() - feature_started
        meta = features["meta"]
        if (
                meta["num_layers"] != gate_config["num_layers"]
                or meta["head_dim"] != gate_config["head_dim"]):
            raise ValueError(f"{path}: gate/feature shape mismatch")
        if len(layout.input_ids) != meta["seq_len"]:
            raise ValueError(f"{path}: prompt/selector length mismatch")
        allowed_blocks = tuple(
            block for block in allowed_blocks if block < meta["kb"])
        if not allowed_blocks:
            raise ValueError(f"{path}: source document has no selector blocks")

        selections = {}
        if "oracle" in args.modes:
            selections["oracle"] = {
                "blocks": [int(target["needle_block"])],
                "scores": None,
                "selection_seconds": 0.0,
                "selection_complexity": "oracle_attribution_only",
                "tree_candidate_union": None,
            }
        if "flat" in args.modes:
            blocks, scores, timing = flat_reader_candidates(
                gate, features, allowed_blocks, args.candidate_blocks)
            selections["flat"] = {
                "blocks": blocks, "scores": scores, **timing}
        if "tree" in args.modes:
            blocks, scores, timing = tree_reader_candidates(
                gate, features, allowed_blocks, args.candidate_blocks,
                beam=args.beam, radix=args.radix,
                layer_chunk=args.selector_layer_chunk)
            selections["tree"] = {
                "blocks": blocks, "scores": scores, **timing}

        modes = {}
        for mode in args.modes:
            selection = selections[mode]
            compile_started = time.perf_counter()
            packet = compile_evidence_packet(
                tokenizer, prompt, user_prompt, question,
                selection["blocks"], block_size,
                block_radius=args.block_radius,
                boundary=args.boundary)
            compile_seconds = time.perf_counter() - compile_started
            if packet.compiled_prompt_tokens > maximum_context:
                raise ValueError(
                    f"{path} {mode}: compact prompt has "
                    f"{packet.compiled_prompt_tokens} tokens, exceeding "
                    f"configured context {maximum_context}")
            generated = _run_dense_packet(
                model, tokenizer, packet, target,
                repeats=args.repeats,
                max_new_tokens=args.max_new_tokens)
            generated.update({
                "selection": selection,
                "feature_load_seconds": feature_load_seconds,
                "compile_seconds": compile_seconds,
                "packet": packet.metadata(),
                "selected_contains_needle": (
                    int(target["needle_block"]) in packet.selected_blocks),
                "expanded_contains_needle": (
                    int(target["needle_block"]) in packet.expanded_blocks),
                # Saved features are an offline diagnostic input. This sum is
                # useful for local profiling but is not deployable TTFT.
                "diagnostic_total_seconds": (
                    feature_load_seconds
                    + float(selection["selection_seconds"])
                    + compile_seconds
                    + generated["seconds"]),
            })
            modes[mode] = generated
            print(
                f"  {mode}: exact={int(generated['exact'])} "
                f"tokens={packet.compiled_prompt_tokens} "
                f"blocks={selection['blocks']}",
                flush=True)

        case_results.append({
            "target": os.path.abspath(path),
            "case_id": target["case_id"],
            "requested_length": int(
                target.get("requested_length", target["seq_len"])),
            "seq_len": int(target["seq_len"]),
            "block_size": block_size,
            "needle_block": int(target["needle_block"]),
            "modes": modes,
        })
        del features, prompt, target, layout
        gc.collect()

    report = {
        "kind": "spruce_evidence_compiler_eval_v1",
        "model": args.model,
        "runtime": runtime_metadata(),
        "gate": os.path.abspath(args.gate),
        "selector_feature_provenance": (
            "saved teacher-extraction Q/K features; extraction time excluded"),
        "config": {
            "modes": list(args.modes),
            "candidate_blocks": args.candidate_blocks,
            "block_radius": args.block_radius,
            "boundary": args.boundary,
            "beam": args.beam,
            "radix": args.radix,
            "selector_layer_chunk": args.selector_layer_chunk,
            "selector_device": args.selector_device,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
        },
        "rope": yarn_metadata(
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=(
                args.original_max_position_embeddings)),
        "cases": case_results,
        "aggregate": {
            mode: _mode_aggregate(case_results, mode)
            for mode in args.modes
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    model = _free_model(model)
    del gate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
