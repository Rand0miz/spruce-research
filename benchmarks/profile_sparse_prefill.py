"""Profile one live-tree SPRUCE prefill without decode.

Exports a Chrome trace, a human-readable operator table, and a JSON summary.
The profiler covers the Qwen prefill forward only. Live selector timing is
measured separately so model and selector costs remain distinguishable.
"""
import argparse
import json
import os
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
    _load_model,
    _model_device,
    _set_attention_backend,
    _warmup,
)
from benchmarks.compare_dense_sparse_live_tree import (
    _warmup_inputs,
    _warmup_selected,
    live_route,
    prepare_cases,
)
from kernels.sparse_prefill import (
    KERNEL_VARIANTS,
    SPRUCE_TRITON_SPARSE_PREFILL,
    kernel_runtime_metadata,
    register_triton_sparse_prefill_attention,
)
from scripts.eval_tree_traversal import load_gate


def _event_time(event, device_name, legacy_name):
    value = getattr(event, device_name, None)
    if value is None:
        value = getattr(event, legacy_name, 0.0)
    return float(value)


def summarize_events(events, row_limit):
    """Return profiler totals and the highest self-device-time operators."""
    rows = []
    for event in events:
        rows.append({
            "name": event.key,
            "calls": int(event.count),
            "self_device_time_us": _event_time(
                event, "self_device_time_total", "self_cuda_time_total"),
            "device_time_us": _event_time(
                event, "device_time_total", "cuda_time_total"),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
        })
    rows.sort(key=lambda row: row["self_device_time_us"], reverse=True)
    return {
        "time_unit": "microseconds",
        "total_self_device_time_us": sum(
            row["self_device_time_us"] for row in rows),
        "total_self_cpu_time_us": sum(
            row["self_cpu_time_us"] for row in rows),
        "top_operators": rows[:row_limit],
    }


def _default_artifact_path(out_path, suffix):
    root, _ = os.path.splitext(out_path)
    return root + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--target", required=True, help="one held-out teacher target")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--prompt-bank")
    parser.add_argument("--kernel-variant", choices=KERNEL_VARIANTS, default="single_head")
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--beam", type=int)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--selector-device", default="cuda")
    parser.add_argument(
        "--selector-dtype", choices=("float32", "float16", "bfloat16"),
        default="float16")
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--yarn-factor", type=float, default=1.0,
        help="static YaRN factor; use 4.0 for Qwen's 131072-token configuration")
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--out", required=True, help="JSON summary path")
    parser.add_argument("--trace", help="Chrome trace path; defaults beside --out")
    parser.add_argument("--table", help="operator table path; defaults beside --out")
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(tempfile.gettempdir(), "spruce_hf_load_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for sparse-prefill profiling")
    if args.k_selected < args.local_window + 1:
        raise SystemExit("--k-selected cannot fit the forced local window")
    if args.selector_layer_chunk < 1 or args.row_limit < 1:
        raise SystemExit("--selector-layer-chunk and --row-limit must be >= 1")
    if args.warmup_tokens < 0:
        raise SystemExit("--warmup-tokens must be non-negative")
    beam = args.beam
    if beam is None:
        beam = max(1, args.k_selected - (args.local_window + 1))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    cases = prepare_cases([os.path.abspath(args.target)], tokenizer, args.prompt_bank)
    case = cases[0]
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    if case["seq_len"] > maximum_context:
        raise SystemExit(
            f"target length {case['seq_len']} exceeds configured context "
            f"{maximum_context}; use --yarn-factor 4.0 for 128K")
    if (case["rope"] is not None
            and float(case["rope"].get("factor", 1.0))
            != float(args.yarn_factor)):
        raise SystemExit("target YaRN factor does not match --yarn-factor")
    if case["block_size"] != 64:
        raise SystemExit("the Triton profiler currently requires block_size=64")

    gate, gate_config = load_gate(args.gate, args.selector_device)
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    register_triton_sparse_prefill_attention()
    model = _load_model(
        args.model, SPRUCE_TRITON_SPARSE_PREFILL,
        dtype, args.load_offload_dir, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    device = _model_device(model)
    _set_attention_backend(model, SPRUCE_TRITON_SPARSE_PREFILL)

    warmup_tokens = min(args.warmup_tokens, case["seq_len"])
    if warmup_tokens:
        _warmup(
            model, _warmup_inputs(case["encoded"], warmup_tokens),
            prefill_kwargs={
                "selected_blocks": _warmup_selected(
                    case, args.k_selected, warmup_tokens, device),
                "block_size": case["block_size"],
                "kernel_variant": args.kernel_variant,
                "validate_selected_blocks_input": False,
            },
        )

    selected, selector_timing = live_route(
        gate, gate_config, case["target"],
        beam=beam, radix=args.radix, k_selected=args.k_selected,
        local_window=args.local_window,
        selector_device=args.selector_device, model_device=device,
        validate=True, selector_dtype=args.selector_dtype,
        selector_layer_chunk=args.selector_layer_chunk,
    )
    inputs = {
        name: value.to(device)
        for name, value in case["encoded"].items()
    }

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    with torch.profiler.profile(
            activities=activities,
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=False) as profiler:
        with torch.inference_mode(), torch.profiler.record_function(
                "spruce_prefill"):
            output = model.base_model(
                **inputs, use_cache=True,
                selected_blocks=selected,
                block_size=case["block_size"],
                kernel_variant=args.kernel_variant,
                validate_selected_blocks_input=False,
            )
            logits = model.get_output_embeddings()(
                output.last_hidden_state[:, -1:, :])
    torch.cuda.synchronize(device)
    profiled_wall_seconds = time.perf_counter() - wall_started

    trace_path = args.trace or _default_artifact_path(args.out, ".trace.json")
    table_path = args.table or _default_artifact_path(args.out, ".ops.txt")
    for path in (args.out, trace_path, table_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    profiler.export_chrome_trace(trace_path)
    events = profiler.key_averages()
    try:
        table = events.table(
            sort_by="self_cuda_time_total", row_limit=args.row_limit)
    except KeyError:
        table = events.table(
            sort_by="self_device_time_total", row_limit=args.row_limit)
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(table)
        handle.write("\n")

    report = {
        "model": args.model,
        "case_id": case["case_id"],
        "teacher_target": case["target"],
        "seq_len": case["seq_len"],
        "block_size": case["block_size"],
        "rope": yarn_metadata(
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=args.original_max_position_embeddings),
        "kernel": kernel_runtime_metadata(args.kernel_variant),
        "selector": {
            "beam": beam,
            "k_selected": args.k_selected,
            "radix": args.radix,
            "local_window": args.local_window,
            "dtype": args.selector_dtype,
            "layer_chunk": args.selector_layer_chunk,
            "feature_extraction_included": False,
            "feature_file_loading_included": False,
            **selector_timing,
        },
        "profile": {
            "scope": "qwen_prefill_forward_plus_final_lm_head",
            "use_cache": True,
            "profiled_wall_seconds": profiled_wall_seconds,
            "record_shapes": args.record_shapes,
            "profile_memory": args.profile_memory,
            **summarize_events(events, args.row_limit),
        },
        "artifacts": {
            "trace": os.path.abspath(trace_path),
            "operator_table": os.path.abspath(table_path),
        },
        "next_token_id": int(logits[:, -1].argmax(dim=-1).item()),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(table)
    print(json.dumps({
        "kernel": report["kernel"],
        "selector_seconds": selector_timing["selector_seconds"],
        "profiled_wall_seconds": profiled_wall_seconds,
        "trace": report["artifacts"]["trace"],
        "operator_table": report["artifacts"]["operator_table"],
        "summary": os.path.abspath(args.out),
    }, indent=2))

    del logits, output, selected, inputs, gate
    model = _free_model(model)


if __name__ == "__main__":
    main()
