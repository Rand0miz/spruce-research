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

from benchmarks.compare_dense_sparse import (
    _free_model,
    _generate,
    _load_model,
    _model_device,
    _set_attention_backend,
    _warmup,
)
from eval.score import score_retrival
from interfaces.validator import validate_selected_blocks
from kernels.sparse_prefill import (
    KERNEL_VARIANTS,
    SPRUCE_TRITON_SPARSE_PREFILL,
    register_triton_sparse_prefill_attention,
)
from scripts.eval_tree_traversal import load_gate, traverse_to_leaf_ids
from scripts.export_selected_blocks import selected_ids_to_blocks
from selector.targets import load_selector_features
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
            "needle_block": int(target["needle_block"]),
            "seq_len": seq_len,
            "block_size": int(target["block_size"]),
            "num_layers": int(target["pooledK"].shape[1]),
            "num_groups": int(target["pooledK"].shape[2]),
            "encoded": {name: value.cpu() for name, value in encoded.items()},
        })
        del target
    return cases


@torch.no_grad()
def live_route(gate, gate_config, target_path, *, beam, radix, k_selected,
               local_window, selector_device, model_device=None, validate=False,
               selector_dtype="float32", selector_layer_chunk=4):
    """Build and traverse one tree now, returning compact selected blocks."""
    selector_device = torch.device(selector_device)
    load_started = time.perf_counter()
    features = load_selector_features(target_path, device=selector_device)
    _sync(selector_device)
    feature_load_seconds = time.perf_counter() - load_started
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
        key_blocks=meta["kb"])
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
        beam, k_selected):
    device = _model_device(model)
    results = {}
    for case_index, case in enumerate(cases):
        samples, answers = [], []
        for repeat in range(args.repeats):
            selected, route_timing = live_route(
                gate, gate_config, case["target"], beam=beam, radix=args.radix,
                k_selected=k_selected, local_window=args.local_window,
                selector_device=args.selector_device, model_device=device,
                validate=(repeat == 0), selector_dtype=args.selector_dtype,
                selector_layer_chunk=args.selector_layer_chunk,
            )
            _set_attention_backend(model, SPRUCE_TRITON_SPARSE_PREFILL)
            token_ids, model_timing = _generate(
                model, case["encoded"], args.max_new_tokens,
                prefill_kwargs={
                    "selected_blocks": selected,
                    "block_size": case["block_size"],
                    "kernel_variant": args.kernel_variant,
                    "validate_selected_blocks_input": False,
                },
            )
            answer = tokenizer.decode(token_ids, skip_special_tokens=True)
            answers.append(answer)
            sample = {**route_timing, **model_timing}
            sample["live_prefill_seconds"] = sample["selector_seconds"] + sample["prefill_seconds"]
            sample["live_total_seconds"] = sample["selector_seconds"] + sample["seconds"]
            samples.append(sample)
            del selected

        timing_keys = (
            "feature_load_seconds", "tree_build_seconds", "tree_traversal_seconds",
            "route_pack_seconds", "route_transfer_seconds", "selector_seconds",
            "prefill_seconds", "decode_seconds", "seconds", "live_prefill_seconds",
            "live_total_seconds", "needle_layer_group_hit_rate",
            "needle_any_group_all_layers_rate",
        )
        results[case["case_key"]] = {
            **score_retrival(answers[0], case["needle"]),
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
    results = {}
    for case_index, case in enumerate(cases):
        samples, answers = [], []
        for _ in range(args.repeats):
            _set_attention_backend(model, "sdpa")
            token_ids, timing = _generate(
                model, case["encoded"], args.max_new_tokens)
            answers.append(tokenizer.decode(token_ids, skip_special_tokens=True))
            samples.append(timing)
        results[case["case_key"]] = {
            **score_retrival(answers[0], case["needle"]),
            **summarize_timings(samples, ("prefill_seconds", "decode_seconds", "seconds")),
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
    kernel_speedups = [case["comparison"]["kernel_prefill_speedup"] for case in case_results]
    live_speedups = [case["comparison"]["live_prefill_speedup"] for case in case_results]
    total_speedups = [case["comparison"]["live_total_speedup"] for case in case_results]
    return {
        "cases": len(case_results),
        "answers_match_rate": sum(case["answers_match"] for case in case_results) / len(case_results),
        "sparse_exact_rate": sum(case["sparse"]["exact"] for case in case_results) / len(case_results),
        "dense_exact_rate": sum(case["dense"]["exact"] for case in case_results) / len(case_results),
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
        dense = dense_results[case["case_key"]]
        case_results.append({
            "case_id": case["case_id"], "seq_len": case["seq_len"],
            "block_size": case["block_size"], "needle_block": case["needle_block"],
            "teacher_target": case["target"], "sparse": sparse, "dense": dense,
            "answers_match": sparse["answer"].strip() == dense["answer"].strip(),
            "comparison": {
                "kernel_prefill_speedup": dense["prefill_seconds"] / sparse["prefill_seconds"],
                "live_prefill_speedup": dense["prefill_seconds"] / sparse["live_prefill_seconds"],
                "live_total_speedup": dense["seconds"] / sparse["live_total_seconds"],
            },
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
        "--kernel-variant", choices=KERNEL_VARIANTS, default="single_head",
        help="single_head restores the measured run2 fast path; tiled_gqa is the run3 ablation")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--plot", help="optional efficiency/accuracy PNG; also writes an adjacent CSV")
    parser.add_argument(
        "--load-offload-dir", default=os.path.join(tempfile.gettempdir(), "spruce_hf_load_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton live-tree benchmark")
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
    cases = prepare_cases(paths, tokenizer, prompt_bank=args.prompt_bank)
    block_sizes = {case["block_size"] for case in cases}
    if block_sizes != {64}:
        raise SystemExit(f"Triton benchmark requires block_size=64, got {sorted(block_sizes)}")
    gate, gate_config = load_gate(args.gate, args.selector_device)
    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    warmup_tokens = min(args.warmup_tokens, cases[0]["seq_len"])

    register_triton_sparse_prefill_attention()
    sparse_model = _load_model(
        args.model, SPRUCE_TRITON_SPARSE_PREFILL, dtype, args.load_offload_dir)
    sparse_results_by_k = {}
    device = _model_device(sparse_model)
    for k_selected, effective_beam in k_sweep:
        if warmup_tokens:
            _set_attention_backend(sparse_model, SPRUCE_TRITON_SPARSE_PREFILL)
            _warmup(
                sparse_model, _warmup_inputs(cases[0]["encoded"], warmup_tokens),
                prefill_kwargs={
                    "selected_blocks": _warmup_selected(
                        cases[0], k_selected, warmup_tokens, device),
                    "block_size": cases[0]["block_size"],
                    "kernel_variant": args.kernel_variant,
                    "validate_selected_blocks_input": False,
                },
            )
        print(
            f"K sweep: K={k_selected} effective_beam={effective_beam}",
            flush=True)
        sparse_results_by_k[k_selected] = _run_sparse_cases(
            sparse_model, gate, gate_config, cases, args, tokenizer,
            beam=effective_beam, k_selected=k_selected)
    sparse_model = _free_model(sparse_model)
    del gate
    gc.collect()
    torch.cuda.empty_cache()

    dense_model = _load_model(args.model, "sdpa", dtype, args.load_offload_dir)
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
            "gate": os.path.abspath(args.gate),
            "selector": selector_metadata(
                args, beam=effective_beam, k_selected=k_selected),
            "kernel": {"variant": args.kernel_variant},
            "repeats": args.repeats, "warmup_tokens": warmup_tokens,
            "cases": case_results,
            "aggregate": aggregate_results(case_results),
        })

    if len(sweep_reports) == 1:
        report = sweep_reports[0]
    else:
        report = {
            "model": args.model,
            "gate": os.path.abspath(args.gate),
            "k_values": [item[0] for item in k_sweep],
            "repeats": args.repeats,
            "warmup_tokens": warmup_tokens,
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
