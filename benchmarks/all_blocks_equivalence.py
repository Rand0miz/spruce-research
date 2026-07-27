"""Real-scale dense-equivalence test for the complete sparse-prefill path.

This is a correctness experiment, not a speed benchmark.  It constructs a
validator-compliant ``selected_blocks`` tensor containing every causal block,
then sends the same dense-verified natural prompt through:

1. dense SDPA,
2. the SPRUCE PyTorch reference backend, and
3. the SPRUCE Triton backend.

With every causal block selected, routing has no pruning decision left to
make.  Any material divergence therefore belongs to route construction,
route consumption, attention integration, or the kernel rather than to the
selector.
"""
import argparse
import gc
import json
import math
import os
import platform
import sys
import tempfile
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from benchmarks.compare_dense_sparse import _set_attention_backend
from configs.long_context import QWEN_NATIVE_CONTEXT, configure_tokenizer, load_model_config
from interfaces.validator import PAD_VALUE, validate_selected_blocks
from kernels.sparse_prefill import (
    SPRUCE_TRITON_SPARSE_PREFILL,
    register_triton_sparse_prefill_attention,
)
from sparse.attention import SPARSE_PREFILL_ATTENTION, register_sparse_prefill_attention


MANIFEST_KIND = "spruce_dense_screen_manifest_v1"
DEFAULT_ACCEPTANCE = {
    "require_finite": True,
    "require_same_top1": True,
    "minimum_top10_overlap": 0.9,
    "minimum_logits_cosine": 0.999,
    "maximum_final_hidden_relative_rmse": 0.02,
}


def all_causal_selected_blocks(
    *, num_layers, num_kv_groups, seq_len, block_size=64,
):
    """Return every causal block in the frozen selected-block interface."""
    if min(num_layers, num_kv_groups, seq_len, block_size) < 1:
        raise ValueError("layers, groups, sequence length, and block size must be positive")
    query_blocks = math.ceil(seq_len / block_size)
    keys = torch.arange(query_blocks, dtype=torch.int32)
    queries = torch.arange(query_blocks, dtype=torch.int32)[:, None]
    rows = keys[None, :].expand(query_blocks, query_blocks)
    rows = torch.where(rows <= queries, rows, torch.full_like(rows, PAD_VALUE))
    selected = rows[None, None, None].expand(
        1, num_layers, num_kv_groups, query_blocks, query_blocks
    ).clone()
    validate_selected_blocks(selected)
    return selected


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{path} is not a {MANIFEST_KIND} manifest")
    records = [
        record for record in manifest.get("candidates", [])
        if record.get("accepted") and record.get("status") == "completed"
    ]
    if not records:
        raise ValueError(f"{path} has no completed, accepted prompts")
    return manifest, records


def select_matched_records(records, requested_lengths):
    """Choose the same natural case/depth/seed at every requested length."""
    requested_lengths = tuple(int(length) for length in requested_lengths)
    requested_set = set(requested_lengths)
    grouped = {}
    for record in records:
        key = (
            record["source_case_id"],
            float(record["depth"]),
            int(record["seed"]),
            int(record.get("variant", 0)),
        )
        grouped.setdefault(key, {})[int(record["requested_length"])] = record

    candidates = [
        (key, by_length) for key, by_length in grouped.items()
        if requested_set.issubset(by_length)
    ]
    if not candidates:
        raise ValueError(
            "no accepted case/depth/seed is shared by requested lengths "
            f"{sorted(requested_set)}"
        )
    # Middle-depth evidence is the least specialized default.  Tie-breakers
    # are stable so rerunning the same manifest selects the same prompts.
    candidates.sort(key=lambda item: (
        abs(item[0][1] - 0.5), item[0][0], item[0][2], item[0][3]))
    _, by_length = candidates[0]
    return [by_length[length] for length in requested_lengths]


def tensor_error(actual, expected):
    """Compact numerical comparison without retaining an extra full tensor."""
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    delta = actual.float() - expected.float()
    expected_float = expected.float()
    mse = delta.square().mean()
    reference_mse = expected_float.square().mean()
    result = {
        "actual_finite": bool(torch.isfinite(actual).all().item()),
        "expected_finite": bool(torch.isfinite(expected).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(mse.sqrt().item()),
        "reference_rms": float(reference_mse.sqrt().item()),
    }
    result["relative_rmse"] = (
        result["rmse"] / result["reference_rms"]
        if result["reference_rms"] else float("inf")
    )
    return result


def logits_error(actual, expected, *, top_k=10):
    result = tensor_error(actual, expected)
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat[None], expected_flat[None]).item()
    actual_top = torch.topk(actual_flat, k=top_k)
    expected_top = torch.topk(expected_flat, k=top_k)
    result.update({
        "cosine": float(cosine),
        "top1_same": bool(actual_top.indices[0] == expected_top.indices[0]),
        "actual_top_ids": [int(value) for value in actual_top.indices.tolist()],
        "expected_top_ids": [int(value) for value in expected_top.indices.tolist()],
        "top10_overlap": len(
            set(actual_top.indices.tolist()) & set(expected_top.indices.tolist())
        ) / top_k,
    })
    return result


def passes_acceptance(logits, final_hidden, criteria=DEFAULT_ACCEPTANCE):
    finite = (
        logits["actual_finite"]
        and logits["expected_finite"]
        and final_hidden["actual_finite"]
        and final_hidden["expected_finite"]
    )
    return bool(
        (finite or not criteria["require_finite"])
        and (logits["top1_same"] or not criteria["require_same_top1"])
        and logits["top10_overlap"] >= criteria["minimum_top10_overlap"]
        and logits["cosine"] >= criteria["minimum_logits_cosine"]
        and final_hidden["relative_rmse"]
        <= criteria["maximum_final_hidden_relative_rmse"]
    )


def runtime_metadata():
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    import transformers
    result["transformers"] = transformers.__version__
    try:
        import triton
        result["triton"] = triton.__version__
    except ImportError:
        result["triton"] = None
    properties = torch.cuda.get_device_properties(0)
    result["gpu"] = {
        "name": properties.name,
        "total_memory_gib": properties.total_memory / 1024**3,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    return result


def hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"could not find hidden tensor in layer output {type(output)!r}")


@contextmanager
def capture_dense_layers(model, references):
    handles = []
    for layer_index, layer in enumerate(model.base_model.layers):
        def save_reference(_module, _inputs, output, index=layer_index):
            references[index] = hidden_tensor(output).detach().to(
                device="cpu", dtype=torch.float16).clone()
        handles.append(layer.register_forward_hook(save_reference))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def compare_layers(model, references, metrics):
    handles = []
    for layer_index, layer in enumerate(model.base_model.layers):
        def compare_reference(_module, _inputs, output, index=layer_index):
            hidden = hidden_tensor(output).detach()
            reference = references[index].to(hidden.device, non_blocking=False)
            metrics.append({"layer": index, **tensor_error(hidden, reference)})
            del reference
        handles.append(layer.register_forward_hook(compare_reference))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(model, encoded, implementation, *, selected=None, block_size=64,
              kernel_variant="single_head", dense_references=None):
    _set_attention_backend(model, implementation)
    kwargs = {"use_cache": False}
    if selected is not None:
        kwargs.update(
            selected_blocks=selected,
            block_size=block_size,
            validate_selected_blocks_input=False,
        )
        if implementation == SPRUCE_TRITON_SPARSE_PREFILL:
            kwargs["kernel_variant"] = kernel_variant

    layer_metrics = []
    if dense_references is None:
        hook_context = capture_dense_layers(model, {})
        references = hook_context
    else:
        hook_context = compare_layers(model, dense_references, layer_metrics)
        references = None

    synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), hook_context:
        output = model.base_model(**encoded, **kwargs)
        final_hidden = output.last_hidden_state.detach()
        logits = model.get_output_embeddings()(final_hidden[:, -1:, :])
    synchronize()
    seconds = time.perf_counter() - started
    return {
        "output": output,
        "final_hidden": final_hidden,
        "logits": logits.detach().float().cpu(),
        "seconds": seconds,
        "layer_metrics": layer_metrics,
        "references_context": references,
    }


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def release_run(run):
    for key in ("output", "final_hidden"):
        value = run.pop(key, None)
        if value is not None:
            del value
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16_384, 32_768])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--backends", nargs="+", choices=("pytorch", "triton"),
        default=["pytorch", "triton"])
    parser.add_argument(
        "--kernel-variant", choices=("single_head", "tiled_gqa", "query_tiled"),
        default="single_head")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--yarn-factor", type=float, default=1.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(tempfile.gettempdir(), "spruce_all_blocks_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("all-block equivalence requires a CUDA GPU")
    if args.block_size != 64 and "triton" in args.backends:
        raise SystemExit("the Triton backend currently requires --block-size 64")

    _, records = load_manifest(args.manifest)
    selected_records = select_matched_records(records, args.lengths)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    register_sparse_prefill_attention()
    register_triton_sparse_prefill_attention()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    config = load_model_config(
        args.model, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    os.makedirs(args.load_offload_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder=args.load_offload_dir,
        attn_implementation="sdpa",
        config=config,
    ).eval()
    device = next(model.parameters()).device
    num_layers = len(model.base_model.layers)
    num_kv_groups = int(model.config.num_key_value_heads)

    report = {
        "kind": "spruce_all_blocks_equivalence_v1",
        "model": args.model,
        "block_size": args.block_size,
        "dtype": args.dtype,
        "backends": args.backends,
        "kernel_variant": args.kernel_variant,
        "acceptance": DEFAULT_ACCEPTANCE,
        "manifest": os.path.abspath(args.manifest),
        "runtime": runtime_metadata(),
        "cases": [],
    }
    atomic_json_dump(report, args.out)

    for record in selected_records:
        prompt = record["prompt_text"]
        encoded = tokenizer(prompt, return_tensors="pt")
        seq_len = int(encoded["input_ids"].shape[1])
        if seq_len != int(record["seq_len"]):
            raise RuntimeError(
                f"{record['candidate_id']} tokenized to {seq_len}, "
                f"manifest records {record['seq_len']}")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        selected_cpu = all_causal_selected_blocks(
            num_layers=num_layers,
            num_kv_groups=num_kv_groups,
            seq_len=seq_len,
            block_size=args.block_size,
        )
        selected = selected_cpu.to(device)
        query_blocks = selected.shape[3]
        case_report = {
            "candidate_id": record["candidate_id"],
            "source_case_id": record["source_case_id"],
            "requested_length": int(record["requested_length"]),
            "seq_len": seq_len,
            "depth": float(record["depth"]),
            "needle_block": int(record["needle_block"]),
            "query_blocks": query_blocks,
            "selected_shape": list(selected.shape),
            "causal_block_entries_per_layer_group": query_blocks * (query_blocks + 1) // 2,
            "runs": {},
        }
        report["cases"].append(case_report)
        atomic_json_dump(report, args.out)

        dense_references = {}
        _set_attention_backend(model, "sdpa")
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), capture_dense_layers(model, dense_references):
            dense_output = model.base_model(**encoded, use_cache=False)
            dense_hidden = dense_output.last_hidden_state.detach()
            dense_logits = model.get_output_embeddings()(
                dense_hidden[:, -1:, :]).detach().float().cpu()
        synchronize()
        case_report["runs"]["dense"] = {
            "seconds": time.perf_counter() - started,
            "next_token_id": int(dense_logits.argmax(dim=-1).item()),
            "next_token": tokenizer.decode(dense_logits.argmax(dim=-1).flatten()),
        }
        dense_final_cpu = dense_hidden.to(
            device="cpu", dtype=torch.float16).clone()
        del dense_output, dense_hidden
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json_dump(report, args.out)

        implementations = {
            "pytorch": SPARSE_PREFILL_ATTENTION,
            "triton": SPRUCE_TRITON_SPARSE_PREFILL,
        }
        for backend in args.backends:
            run = run_model(
                model,
                encoded,
                implementations[backend],
                selected=selected,
                block_size=args.block_size,
                kernel_variant=args.kernel_variant,
                dense_references=dense_references,
            )
            final_reference = dense_final_cpu.to(run["final_hidden"].device)
            final_hidden_metrics = tensor_error(run["final_hidden"], final_reference)
            del final_reference
            run_report = {
                "seconds": run["seconds"],
                "next_token_id": int(run["logits"].argmax(dim=-1).item()),
                "next_token": tokenizer.decode(run["logits"].argmax(dim=-1).flatten()),
                "logits": logits_error(run["logits"], dense_logits),
                "final_hidden": final_hidden_metrics,
                "layers": run["layer_metrics"],
            }
            run_report["passed"] = passes_acceptance(
                run_report["logits"], run_report["final_hidden"])
            case_report["runs"][backend] = run_report
            atomic_json_dump(report, args.out)
            release_run(run)

        del dense_references, dense_final_cpu, dense_logits, selected, selected_cpu, encoded
        gc.collect()
        torch.cuda.empty_cache()

    report["passed"] = all(
        run["passed"]
        for case in report["cases"]
        for name, run in case["runs"].items()
        if name != "dense"
    )
    atomic_json_dump(report, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
