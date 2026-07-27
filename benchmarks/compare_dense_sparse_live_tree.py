"""Benchmark dense Qwen against live SPRUCE tree traversal + sparse prefill.

Routes are generated from the trained gate for every target and repeat. Exported
selected-block artifacts are deliberately not accepted. The current selector
input is the pooled Q/K feature data stored in teacher targets; feature loading
and offline feature extraction are reported explicitly and excluded from live
tree time.
"""
import argparse
from contextlib import nullcontext
import gc
import glob
import json
import math
import os
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from configs.long_context import (
    QWEN_NATIVE_CONTEXT,
    configure_tokenizer,
    context_limit,
    yarn_metadata,
)
from benchmarks.compare_dense_sparse import (
    _free_model,
    _generate,
    _load_model,
    _model_device,
    _set_attention_backend,
    _warmup,
    runtime_metadata,
)
from eval.score import score_retrival
from interfaces.validator import validate_selected_blocks
from kernels.sparse_prefill import (
    KERNEL_VARIANTS,
    SPRUCE_TRITON_SPARSE_PREFILL,
    kernel_runtime_metadata,
    register_triton_sparse_prefill_attention,
)
from scripts.eval_tree_traversal import load_gate, traverse_to_leaf_ids
from scripts.export_selected_blocks import selected_ids_to_blocks
from scripts.route_overrides import (
    candidate_span,
    charged_route_density,
    dense_candidate_routes,
    dense_evidence_routes,
    dense_reader_routes,
    force_needle_routes,
    teacher_dual_top_p_routes,
    teacher_topk_routes,
)
from selector.targets import load_selector_features, load_teacher
from sparse.attention import (
    SPARSE_PREFILL_ATTENTION,
    register_sparse_prefill_attention,
)
from selector.tree import build_key_tree
from teacher.prompt_replay import reconstruct_teacher_prompt


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        paths.extend(matches)
    return list(dict.fromkeys(os.path.abspath(path) for path in paths if os.path.isfile(path)))


def _sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(samples, key):
    return float(statistics.median(sample[key] for sample in samples))


def summarize_timings(samples, keys):
    return {key: _median(samples, key) for key in keys}


def prepare_cases(paths, tokenizer, prompt_bank=None):
    cases = []
    for path in paths:
        prompt, target = reconstruct_teacher_prompt(tokenizer, path, bank_path=prompt_bank)
        encoded = tokenizer(prompt, return_tensors="pt")
        seq_len = int(target["seq_len"])
        if encoded.input_ids.shape[1] != seq_len:
            raise ValueError(f"{path}: reconstructed {encoded.input_ids.shape[1]} tokens, expected {seq_len}")
        cases.append({
            "target": path,
            "case_key": f"{target['case_id']}|len{seq_len}|{os.path.basename(path)}",
            "case_id": target["case_id"],
            "needle": target["needle"],
            "reference_answers": target.get("reference_answers"),
            "needle_block": int(target["needle_block"]),
            "seq_len": seq_len,
            "requested_length": int(target.get("requested_length", seq_len)),
            "block_size": int(target["block_size"]),
            "num_layers": int(target["pooledK"].shape[1]),
            "num_groups": int(target["pooledK"].shape[2]),
            "rope": target.get("rope"),
            "encoded": {name: value.cpu() for name, value in encoded.items()},
        })
        del target
    return cases


@torch.no_grad()
def live_route(gate, gate_config, target_path, *, beam, radix, k_selected,
               local_window, selector_device, model_device=None, validate=False,
               selector_dtype="float32", selector_layer_chunk=4,
               features=None, sink_blocks=0):
    """Build and traverse one tree now, returning compact selected blocks."""
    selector_device = torch.device(selector_device)
    if features is None:
        load_started = time.perf_counter()
        features = load_selector_features(target_path, device=selector_device)
        _sync(selector_device)
        feature_load_seconds = time.perf_counter() - load_started
    else:
        feature_load_seconds = 0.0
    meta = features["meta"]
    if (meta["num_layers"] != gate_config["num_layers"]
            or meta["head_dim"] != gate_config["head_dim"]):
        raise ValueError(f"{target_path}: teacher feature shape mismatches gate config")

    started = time.perf_counter()
    levels = build_key_tree(features["k_feat"], radix=radix)
    _sync(selector_device)
    tree_build_seconds = time.perf_counter() - started

    dtype_map = {
        "float32": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if selector_dtype not in dtype_map:
        raise ValueError(f"unsupported selector_dtype {selector_dtype!r}")
    compute_dtype = dtype_map[selector_dtype]
    autocast = (
        torch.autocast(selector_device.type, dtype=compute_dtype)
        if compute_dtype is not None else nullcontext()
    )
    started = time.perf_counter()
    with autocast:
        selected_ids = traverse_to_leaf_ids(
            gate, features["q_feat"], levels, beam=beam, radix=radix,
            layer_chunk=selector_layer_chunk)
    _sync(selector_device)
    tree_traversal_seconds = time.perf_counter() - started

    started = time.perf_counter()
    selected = selected_ids_to_blocks(
        selected_ids, k_selected=k_selected, local_window=local_window,
        key_blocks=meta["kb"], sink_blocks=sink_blocks)
    _sync(selector_device)
    route_pack_seconds = time.perf_counter() - started

    needle_block = meta["needle_block"]
    if needle_block >= 0:
        final_reader = selected[0, :, :, -1]
        needle_hits = (final_reader == needle_block).any(dim=-1)
        route_quality = {
            "needle_layer_group_hit_rate": float(needle_hits.float().mean().cpu()),
            "needle_any_group_all_layers_rate": float(
                needle_hits.any(dim=-1).float().mean().cpu()),
        }
    else:
        route_quality = {
            "needle_layer_group_hit_rate": float("nan"),
            "needle_any_group_all_layers_rate": float("nan"),
        }

    transfer_seconds = 0.0
    if model_device is not None and selected.device != torch.device(model_device):
        started = time.perf_counter()
        selected = selected.to(model_device)
        _sync(model_device)
        transfer_seconds = time.perf_counter() - started

    if validate:
        validate_selected_blocks(selected.detach().cpu(), local_window=local_window)

    del selected_ids, levels, features
    gc.collect()
    selector_seconds = (
        tree_build_seconds + tree_traversal_seconds + route_pack_seconds + transfer_seconds)
    return selected, {
        "feature_load_seconds": feature_load_seconds,
        "tree_build_seconds": tree_build_seconds,
        "tree_traversal_seconds": tree_traversal_seconds,
        "route_pack_seconds": route_pack_seconds,
        "route_transfer_seconds": transfer_seconds,
        "selector_seconds": selector_seconds,
        **route_quality,
    }


def sparse_prefill_kwargs(
        backend, selected, block_size, kernel_variant, dense_layers=()):
    """Prefill kwargs for either sparse backend; only Triton takes a variant."""
    kwargs = {
        "selected_blocks": selected,
        "block_size": block_size,
        "validate_selected_blocks_input": False,
    }
    if backend == "triton":
        kwargs["kernel_variant"] = kernel_variant
        kwargs["dense_layers"] = tuple(int(layer) for layer in dense_layers)
    return kwargs


def _route_quality(selected, needle_block):
    if needle_block is None or needle_block < 0:
        return {
            "needle_layer_group_hit_rate": float("nan"),
            "needle_any_group_all_layers_rate": float("nan"),
        }
    final_reader = selected[0, :, :, -1]
    needle_hits = (final_reader == needle_block).any(dim=-1)
    return {
        "needle_layer_group_hit_rate": float(needle_hits.float().mean().cpu()),
        "needle_any_group_all_layers_rate": float(
            needle_hits.any(dim=-1).float().mean().cpu()),
    }


def _warmup_inputs(encoded, warmup_tokens):
    return {name: value[:, :warmup_tokens] for name, value in encoded.items()}


def _warmup_selected(case, k_selected, warmup_tokens, device):
    qblocks = math.ceil(warmup_tokens / case["block_size"])
    # Repeated causal block 0 deliberately exercises every K loop slot during
    # Triton autotuning without requiring a long warm-up prompt.
    selected = torch.zeros(
        (1, case["num_layers"], case["num_groups"], qblocks, k_selected),
        dtype=torch.int32, device=device)
    return selected


def _run_sparse_cases(
        model, gate, gate_config, cases, args, tokenizer, *,
        beam, k_selected,
        sparse_implementation=SPRUCE_TRITON_SPARSE_PREFILL):
    backend = getattr(args, "backend", "triton")
    route_mode = getattr(args, "route_mode", "learned")
    device = _model_device(model)
    results = {}
    for case_index, case in enumerate(cases):
        samples, answers = [], []
        load_started = time.perf_counter()
        features = load_selector_features(
            case["target"], device=args.selector_device)
        _sync(args.selector_device)
        feature_load_seconds = time.perf_counter() - load_started

        candidate_blocks = None
        if route_mode == "dense-candidates":
            # Deployable candidate selection: the gate's own leaf scores at
            # the reader row, max-aggregated over layers and KV groups. No
            # teacher mass, no needle metadata.
            with torch.no_grad():
                reader_scores = gate(
                    features["q_feat"][:, :, -1:], features["k_feat"])
            pooled_scores = reader_scores[:, :, 0, :].amax(dim=(0, 1))
            m = min(int(getattr(args, "candidate_blocks", 8)),
                    pooled_scores.shape[0])
            candidate_blocks = pooled_scores.topk(m).indices.tolist()
            del reader_scores, pooled_scores

        teacher_routes = None
        if route_mode in ("teacher-top8", "teacher-dual-top-p"):
            teacher = load_teacher(case["target"], device="cpu")
            if route_mode == "teacher-top8":
                teacher_routes = teacher_topk_routes(
                    teacher["target"], k_selected=k_selected,
                    local_window=args.local_window)
            else:
                teacher_routes = teacher_dual_top_p_routes(
                    teacher["target"], top_p=args.teacher_top_p,
                    block_size=case["block_size"],
                    sink_tokens=args.spot_sink_tokens,
                    recency_tokens=args.spot_recency_tokens,
                    k_min_tokens=args.spot_k_min_tokens)
            del teacher
            gc.collect()

        for repeat in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            selected, route_timing = live_route(
                gate, gate_config, case["target"], beam=beam, radix=args.radix,
                k_selected=k_selected, local_window=args.local_window,
                selector_device=args.selector_device, model_device=device,
                validate=(repeat == 0), selector_dtype=args.selector_dtype,
                selector_layer_chunk=args.selector_layer_chunk,
                features=features,
                sink_blocks=int(getattr(args, "sink_blocks", 0)),
            )
            route_timing["feature_load_seconds"] = feature_load_seconds
            if route_mode == "oracle-needle" and case["needle_block"] >= 0:
                selected = force_needle_routes(selected, case["needle_block"])
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
            elif route_mode == "dense-reader":
                selected = dense_reader_routes(selected)
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
            elif (route_mode == "oracle-dense-evidence"
                    and case["needle_block"] >= 0):
                selected = dense_evidence_routes(
                    selected, case["needle_block"],
                    neighborhood=getattr(args, "evidence_neighborhood", 0))
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
            elif route_mode == "dense-candidates":
                neighborhood = int(getattr(args, "candidate_neighborhood", 0))
                selected = dense_candidate_routes(
                    selected, candidate_blocks, neighborhood=neighborhood)
                span = candidate_span(
                    candidate_blocks, selected.shape[3], neighborhood)
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
                route_timing["candidate_blocks"] = list(candidate_blocks)
                route_timing["candidate_contains_needle"] = (
                    int(case["needle_block"]) in candidate_blocks)
                route_timing["candidate_neighborhood"] = neighborhood
                # Windows overlap, so the densified-row count is measured here
                # rather than derived from M*(2w+1): this is the real cost.
                route_timing["densified_rows"] = len(span)
                route_timing["span_contains_needle"] = int(
                    int(case["needle_block"]) in span)
            elif route_mode == "teacher-top8":
                selected = teacher_routes.to(device)
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
            elif route_mode == "teacher-dual-top-p":
                selected = teacher_routes.to(device)
                route_timing.update(
                    _route_quality(selected, case["needle_block"]))
                route_timing.update({
                    "teacher_top_p": float(args.teacher_top_p),
                    "spot_sink_tokens": int(args.spot_sink_tokens),
                    "spot_recency_tokens": int(args.spot_recency_tokens),
                    "spot_k_min_tokens": int(args.spot_k_min_tokens),
                    "oracle_route_width": int(selected.shape[-1]),
                })
            route_timing.update(charged_route_density(
                selected, getattr(args, "dense_layers", ())))
            if route_mode != "learned" and repeat == 0:
                validate_selected_blocks(
                    selected.detach().cpu(), local_window=args.local_window)
            _set_attention_backend(model, sparse_implementation)
            token_ids, model_timing = _generate(
                model, case["encoded"], args.max_new_tokens,
                prefill_kwargs=sparse_prefill_kwargs(
                    backend, selected, case["block_size"], args.kernel_variant,
                    dense_layers=getattr(args, "dense_layers", ())),
            )
            answer = tokenizer.decode(token_ids, skip_special_tokens=True)
            answers.append(answer)
            sample = {**route_timing, **model_timing}
            if device.type == "cuda":
                sample["peak_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 1e9)
                sample["peak_memory_reserved_gb"] = (
                    torch.cuda.max_memory_reserved(device) / 1e9)
            else:
                sample["peak_memory_allocated_gb"] = 0.0
                sample["peak_memory_reserved_gb"] = 0.0
            sample["live_prefill_seconds"] = sample["selector_seconds"] + sample["prefill_seconds"]
            sample["live_total_seconds"] = sample["selector_seconds"] + sample["seconds"]
            samples.append(sample)
            del selected
        del features
        gc.collect()

        timing_keys = (
            "feature_load_seconds", "tree_build_seconds", "tree_traversal_seconds",
            "route_pack_seconds", "route_transfer_seconds", "selector_seconds",
            "prefill_seconds", "decode_seconds", "seconds", "live_prefill_seconds",
            "live_total_seconds", "needle_layer_group_hit_rate",
            "needle_any_group_all_layers_rate",
            "peak_memory_allocated_gb", "peak_memory_reserved_gb",
        )
        results[case["case_key"]] = {
            **score_retrival(
                answers[0], case["needle"],
                reference_answers=case.get("reference_answers")),
            **summarize_timings(samples, timing_keys),
            "answer_repeat_match": len(set(answer.strip() for answer in answers)) == 1,
            "timing_samples": samples,
        }
        print(
            f"sparse {case_index + 1}/{len(cases)} {case['case_id']} "
            f"tree={results[case['case_key']]['selector_seconds']:.4f}s "
            f"prefill={results[case['case_key']]['prefill_seconds']:.4f}s",
            flush=True,
        )
    return results


def _run_dense_cases(model, cases, args, tokenizer):
    device = _model_device(model)
    results = {}
    for case_index, case in enumerate(cases):
        samples, answers = [], []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _set_attention_backend(model, "sdpa")
            token_ids, timing = _generate(
                model, case["encoded"], args.max_new_tokens)
            answers.append(tokenizer.decode(token_ids, skip_special_tokens=True))
            if device.type == "cuda":
                timing["peak_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 1e9)
                timing["peak_memory_reserved_gb"] = (
                    torch.cuda.max_memory_reserved(device) / 1e9)
            else:
                timing["peak_memory_allocated_gb"] = 0.0
                timing["peak_memory_reserved_gb"] = 0.0
            samples.append(timing)
        results[case["case_key"]] = {
            **score_retrival(
                answers[0], case["needle"],
                reference_answers=case.get("reference_answers")),
            **summarize_timings(samples, (
                "prefill_seconds", "decode_seconds", "seconds",
                "peak_memory_allocated_gb", "peak_memory_reserved_gb")),
            "answer_repeat_match": len(set(answer.strip() for answer in answers)) == 1,
            "timing_samples": samples,
        }
        print(
            f"dense  {case_index + 1}/{len(cases)} {case['case_id']} "
            f"prefill={results[case['case_key']]['prefill_seconds']:.4f}s",
            flush=True,
        )
    return results


def aggregate_results(case_results):
    if any(case["dense"] is None for case in case_results):
        # --skip-dense diagnostic runs: retrieval accuracy against the known
        # needle only, no paired dense timing or accuracy comparison.
        return {
            "cases": len(case_results),
            "sparse_exact_rate": (
                sum(case["sparse"]["exact"] for case in case_results)
                / len(case_results)),
            "mean_sparse_fuzzy": statistics.mean(
                case["sparse"]["fuzzy"] for case in case_results),
            "dense_skipped": True,
        }
    kernel_speedups = [case["comparison"]["kernel_prefill_speedup"] for case in case_results]
    live_speedups = [case["comparison"]["live_prefill_speedup"] for case in case_results]
    total_speedups = [case["comparison"]["live_total_speedup"] for case in case_results]
    sparse_only = sum(
        bool(case["sparse"]["exact"]) and not bool(case["dense"]["exact"])
        for case in case_results)
    dense_only = sum(
        bool(case["dense"]["exact"]) and not bool(case["sparse"]["exact"])
        for case in case_results)
    both_correct = sum(
        bool(case["dense"]["exact"]) and bool(case["sparse"]["exact"])
        for case in case_results)
    neither_correct = len(case_results) - sparse_only - dense_only - both_correct
    discordant = sparse_only + dense_only
    if discordant:
        smaller = min(sparse_only, dense_only)
        mcnemar_p = min(
            1.0,
            2.0 * sum(
                math.comb(discordant, index)
                for index in range(smaller + 1))
            / (2 ** discordant),
        )
    else:
        mcnemar_p = 1.0
    sparse_rate = (
        sum(case["sparse"]["exact"] for case in case_results)
        / len(case_results))
    dense_rate = (
        sum(case["dense"]["exact"] for case in case_results)
        / len(case_results))
    aggregate = {
        "cases": len(case_results),
        "answers_match_rate": sum(case["answers_match"] for case in case_results) / len(case_results),
        "sparse_exact_rate": sparse_rate,
        "dense_exact_rate": dense_rate,
        "sparse_minus_dense_exact_rate": sparse_rate - dense_rate,
        "paired_exact_counts": {
            "sparse_only_correct": sparse_only,
            "dense_only_correct": dense_only,
            "both_correct": both_correct,
            "neither_correct": neither_correct,
            "exact_mcnemar_two_sided_p": mcnemar_p,
        },
        "mean_sparse_fuzzy": statistics.mean(case["sparse"]["fuzzy"] for case in case_results),
        "mean_dense_fuzzy": statistics.mean(case["dense"]["fuzzy"] for case in case_results),
        "median_kernel_prefill_speedup": statistics.median(kernel_speedups),
        "median_live_prefill_speedup": statistics.median(live_speedups),
        "median_live_total_speedup": statistics.median(total_speedups),
        "sum_kernel_prefill_speedup": (
            sum(case["dense"]["prefill_seconds"] for case in case_results)
            / sum(case["sparse"]["prefill_seconds"] for case in case_results)),
        "sum_live_prefill_speedup": (
            sum(case["dense"]["prefill_seconds"] for case in case_results)
            / sum(case["sparse"]["live_prefill_seconds"] for case in case_results)),
    }
    if all(
            "peak_memory_allocated_gb" in case["sparse"]
            and "peak_memory_allocated_gb" in case["dense"]
            for case in case_results):
        aggregate.update({
            "max_sparse_peak_memory_allocated_gb": max(
                case["sparse"]["peak_memory_allocated_gb"]
                for case in case_results),
            "max_dense_peak_memory_allocated_gb": max(
                case["dense"]["peak_memory_allocated_gb"]
                for case in case_results),
        })
    return aggregate


def resolve_k_sweep(*, beam, local_window, k_selected=None, k_values=None):
    """Return validated ``(K, effective_beam)`` pairs for one or many runs."""
    if k_values:
        values = list(dict.fromkeys(int(value) for value in k_values))
    else:
        values = [
            int(k_selected)
            if k_selected is not None
            else int(beam + local_window + 1)
        ]
    minimum = local_window + 1
    if any(value < minimum for value in values):
        raise ValueError(
            f"every K must be >= {minimum} to fit the forced local window")
    # Reserve the maximum local-window width. Smaller K runs reduce the tree
    # beam too, rather than selecting beam nodes and truncating them by block ID.
    return [
        (value, min(beam, max(1, value - minimum)))
        for value in values
    ]


def build_case_results(cases, sparse_results, dense_results):
    case_results = []
    for case in cases:
        sparse = sparse_results[case["case_key"]]
        dense = dense_results.get(case["case_key"]) if dense_results else None
        case_results.append({
            "case_id": case["case_id"], "seq_len": case["seq_len"],
            "requested_length": case["requested_length"],
            "block_size": case["block_size"], "needle_block": case["needle_block"],
            "teacher_target": case["target"], "sparse": sparse, "dense": dense,
            "answers_match": (
                sparse["answer"].strip() == dense["answer"].strip()
                if dense is not None else None),
            "comparison": {
                "kernel_prefill_speedup": dense["prefill_seconds"] / sparse["prefill_seconds"],
                "live_prefill_speedup": dense["prefill_seconds"] / sparse["live_prefill_seconds"],
                "live_total_speedup": dense["seconds"] / sparse["live_total_seconds"],
            } if dense is not None else {},
        })
    return case_results


def selector_metadata(args, *, beam, k_selected):
    return {
        "mode": "live_candidate_only_tree_traversal",
        "route_representation": "compact_leaf_ids",
        "feature_source": "pooled_qk_loaded_from_teacher_target",
        "feature_extraction_included": False,
        "feature_file_loading_included": False,
        "compute_dtype": args.selector_dtype,
        "layer_chunk": args.selector_layer_chunk,
        "beam": beam, "radix": args.radix,
        "k_selected": k_selected, "local_window": args.local_window,
        "sink_blocks": int(getattr(args, "sink_blocks", 0)),
        "dense_layers": list(getattr(args, "dense_layers", ())),
        "route_mode": getattr(args, "route_mode", "learned"),
        "backend": getattr(args, "backend", "triton"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, help="trained selector checkpoint")
    parser.add_argument("--targets", nargs="+", required=True, help="held-out teacher targets/globs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--prompt-bank")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--radix", type=int, default=2)
    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument(
        "--k-selected", type=int, default=None,
        help="single selected-block width; default beam + local_window + 1")
    k_group.add_argument(
        "--k-values", type=int, nargs="+",
        help="benchmark several selected-block widths in one run, e.g. 10 12 14 16 18")
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--selector-device", default="cuda")
    parser.add_argument(
        "--selector-dtype", choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="tree scoring precision; float16 is faster but must be accuracy-validated",
    )
    parser.add_argument(
        "--selector-layer-chunk", type=int, default=4,
        help="selector layers scored per batched candidate operation")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--yarn-factor", type=float, default=1.0,
        help="static YaRN factor; use 4.0 for Qwen's 131072-token configuration")
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument(
        "--kernel-variant", choices=KERNEL_VARIANTS, default="single_head",
        help="single_head is the measured control; other choices are isolated kernel ablations")
    parser.add_argument(
        "--route-mode",
        choices=("learned", "oracle-needle", "teacher-top8",
                 "teacher-dual-top-p", "dense-reader",
                 "oracle-dense-evidence", "dense-candidates"),
        default="learned",
        help="learned: gate traversal routes (production); oracle-needle: "
             "learned routes widened by one slot with the evidence block "
             "forced into every causal row (sufficiency control); teacher-top8: "
             "routes packed from the teacher's top-k mass, no gate influence "
             "on selection (label-ceiling control); teacher-dual-top-p: exact "
             "dense-teacher dual-top-p routes adapted to sparse prefill "
             "(nondeployable construction ceiling); dense-reader: learned "
             "routes but the final reader row attends every causal block "
             "(reader-budget control)")
    parser.add_argument(
        "--teacher-top-p", type=float, default=0.9,
        help="teacher-dual-top-p only: residual nucleus mass in (0,1]")
    parser.add_argument(
        "--spot-sink-tokens", type=int, default=128,
        help="teacher-dual-top-p only: reserved absolute sink prefix in tokens")
    parser.add_argument(
        "--spot-recency-tokens", type=int, default=256,
        help="teacher-dual-top-p only: reserved causal recency window in tokens")
    parser.add_argument(
        "--spot-k-min-tokens", type=int, default=512,
        help="teacher-dual-top-p only: minimum total selected budget in tokens")
    parser.add_argument(
        "--backend", choices=("triton", "pytorch"), default="triton",
        help="sparse prefill implementation; pytorch is the correctness "
             "reference and the only option without Triton (Windows)")
    parser.add_argument(
        "--candidate-blocks", type=int, default=8,
        help="dense-candidates only: number of gate-scored reader-row blocks "
             "whose query rows are densified (retrieve-then-re-encode, no "
             "oracle knowledge)")
    parser.add_argument(
        "--sink-blocks", type=int, default=0,
        help="force the first N key blocks (attention sink) into every causal "
             "row, outside the K budget, like the local window. 0 leaves the "
             "sink competing for learned slots, so rows that rank it below K "
             "drop it entirely")
    parser.add_argument(
        "--candidate-neighborhood", type=int, default=0,
        help="dense-candidates only: also densify +-N query rows around each "
             "candidate. 0 reproduces scattered top-M densification; N>0 tests "
             "whether repair must be contiguous around the evidence")
    parser.add_argument(
        "--evidence-neighborhood", type=int, default=0,
        help="oracle-dense-evidence only: also densify the query rows of "
             "blocks within +-N of the evidence (boundary-straddle control)")
    parser.add_argument(
        "--dense-layers", type=int, nargs="*", default=[],
        help="zero-based decoder layers dispatched to dense SDPA while all "
             "other layers keep compact Triton sparse routes; dense layers "
             "are charged in reported attention density")
    parser.add_argument(
        "--skip-dense", action="store_true",
        help="skip the paired dense run (8GB laptops cannot fit a full-length "
             "dense SDPA prefill); sparse retrieval is still scored against "
             "the target's known needle")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--plot", help="optional efficiency/accuracy PNG; also writes an adjacent CSV")
    parser.add_argument(
        "--load-offload-dir", default=os.path.join(tempfile.gettempdir(), "spruce_hf_load_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the live-tree benchmark")
    if args.route_mode != "learned" and args.k_values:
        raise SystemExit(
            "--route-mode overrides are only supported for the single-K path")
    if args.dense_layers and args.backend != "triton":
        raise SystemExit("--dense-layers currently requires --backend triton")
    if not 0.0 < args.teacher_top_p <= 1.0:
        raise SystemExit("--teacher-top-p must be in (0,1]")
    if any(value < 0 for value in (
            args.spot_sink_tokens, args.spot_recency_tokens,
            args.spot_k_min_tokens)):
        raise SystemExit("SpotAttention token budgets must be non-negative")
    args.dense_layers = sorted(set(args.dense_layers))
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise SystemExit("--repeats and --max-new-tokens must be >= 1")
    if args.warmup_tokens < 0:
        raise SystemExit("--warmup-tokens must be non-negative")
    if (args.beam < 1 or args.radix < 2 or args.local_window < 0
            or args.selector_layer_chunk < 1):
        raise SystemExit("invalid beam/radix/local-window configuration")
    try:
        k_sweep = resolve_k_sweep(
            beam=args.beam, local_window=args.local_window,
            k_selected=args.k_selected, k_values=args.k_values)
    except ValueError as error:
        raise SystemExit(str(error)) from error

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
    cases = prepare_cases(paths, tokenizer, prompt_bank=args.prompt_bank)
    layer_counts = {case["num_layers"] for case in cases}
    if len(layer_counts) != 1:
        raise SystemExit(f"targets disagree on model layer count: {layer_counts}")
    num_layers = next(iter(layer_counts))
    if any(layer < 0 or layer >= num_layers for layer in args.dense_layers):
        raise SystemExit(
            f"--dense-layers must be in [0,{num_layers}), got "
            f"{args.dense_layers}")
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    longest = max(case["seq_len"] for case in cases)
    if longest > maximum_context:
        raise SystemExit(
            f"longest target {longest} exceeds configured context "
            f"{maximum_context}; use --yarn-factor 4.0 for 128K")
    mismatched_rope = [
        case["target"] for case in cases
        if case["rope"] is not None
        and float(case["rope"].get("factor", 1.0)) != float(args.yarn_factor)
    ]
    if mismatched_rope:
        raise SystemExit(
            "target YaRN factor does not match --yarn-factor: "
            f"{mismatched_rope[0]}")
    block_sizes = {case["block_size"] for case in cases}
    if block_sizes != {64}:
        raise SystemExit(f"Triton benchmark requires block_size=64, got {sorted(block_sizes)}")
    gate, gate_config = load_gate(args.gate, args.selector_device)
    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    warmup_tokens = min(args.warmup_tokens, cases[0]["seq_len"])

    if args.backend == "triton":
        register_triton_sparse_prefill_attention()
        sparse_implementation = SPRUCE_TRITON_SPARSE_PREFILL
    else:
        register_sparse_prefill_attention()
        sparse_implementation = SPARSE_PREFILL_ATTENTION
    sparse_model = _load_model(
        args.model, sparse_implementation, dtype,
        args.load_offload_dir, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    sparse_results_by_k = {}
    kernel_metadata_by_k = {}
    device = _model_device(sparse_model)
    for k_selected, effective_beam in k_sweep:
        if warmup_tokens:
            _set_attention_backend(sparse_model, sparse_implementation)
            _warmup(
                sparse_model, _warmup_inputs(cases[0]["encoded"], warmup_tokens),
                prefill_kwargs=sparse_prefill_kwargs(
                    args.backend,
                    _warmup_selected(cases[0], k_selected, warmup_tokens, device),
                    cases[0]["block_size"], args.kernel_variant,
                    dense_layers=args.dense_layers),
            )
        print(
            f"K sweep: K={k_selected} effective_beam={effective_beam}",
            flush=True)
        sparse_results_by_k[k_selected] = _run_sparse_cases(
            sparse_model, gate, gate_config, cases, args, tokenizer,
            beam=effective_beam, k_selected=k_selected,
            sparse_implementation=sparse_implementation)
        kernel_metadata_by_k[k_selected] = (
            kernel_runtime_metadata(args.kernel_variant)
            if args.backend == "triton"
            else {"backend": "pytorch_reference"})
    sparse_model = _free_model(sparse_model)
    del gate
    gc.collect()
    torch.cuda.empty_cache()

    if args.skip_dense:
        dense_results = {}
    else:
        dense_model = _load_model(
            args.model, "sdpa", dtype, args.load_offload_dir,
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=args.original_max_position_embeddings)
        if warmup_tokens:
            _warmup(dense_model, _warmup_inputs(cases[0]["encoded"], warmup_tokens))
        dense_results = _run_dense_cases(dense_model, cases, args, tokenizer)
        dense_model = _free_model(dense_model)

    sweep_reports = []
    for k_selected, effective_beam in k_sweep:
        case_results = build_case_results(
            cases, sparse_results_by_k[k_selected], dense_results)
        sweep_reports.append({
            "model": args.model,
            "runtime": runtime_metadata(),
            "gate": os.path.abspath(args.gate),
            "selector": selector_metadata(
                args, beam=effective_beam, k_selected=k_selected),
            "kernel": kernel_metadata_by_k[k_selected],
            "rope": yarn_metadata(
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_max_position_embeddings),
            "repeats": args.repeats, "warmup_tokens": warmup_tokens,
            "cases": case_results,
            "aggregate": aggregate_results(case_results),
        })

    if len(sweep_reports) == 1:
        report = sweep_reports[0]
    else:
        report = {
            "model": args.model,
            "runtime": runtime_metadata(),
            "gate": os.path.abspath(args.gate),
            "k_values": [item[0] for item in k_sweep],
            "repeats": args.repeats,
            "warmup_tokens": warmup_tokens,
            "rope": yarn_metadata(
                yarn_factor=args.yarn_factor,
                original_max_position_embeddings=args.original_max_position_embeddings),
            "sweeps": sweep_reports,
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    for sweep_report in sweep_reports:
        print(
            f"K={sweep_report['selector']['k_selected']} "
            f"beam={sweep_report['selector']['beam']}")
        print(json.dumps(sweep_report["aggregate"], indent=2))
    print(f"full report -> {args.out}")
    if args.plot:
        from benchmarks.plot_efficiency_accuracy import save_efficiency_accuracy_plot
        plot_root, plot_ext = os.path.splitext(args.plot)
        plot_ext = plot_ext or ".png"
        for sweep_report in sweep_reports:
            k_selected = sweep_report["selector"]["k_selected"]
            plot_arg = (
                args.plot if len(sweep_reports) == 1
                else f"{plot_root}_K{k_selected}{plot_ext}")
            plot_path, csv_path = save_efficiency_accuracy_plot(
                sweep_report, plot_arg)
            print(f"K={k_selected} scaling plot -> {plot_path}")
            print(f"K={k_selected} scaling data -> {csv_path}")


if __name__ == "__main__":
    main()
