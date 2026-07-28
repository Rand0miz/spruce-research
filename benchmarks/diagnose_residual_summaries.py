"""Tune deterministic residual-summary prototypes on development prompts.

This diagnostic compares dense SDPA with the exact-only SPRUCE PyTorch
reference and P={1,2,4} deterministic residual summaries. It samples the same
token positions from every attention-module output and decoder-layer output,
then reports globally weighted RMSE without retaining full-length hidden
states on CPU.

It is a development diagnostic, not a paper latency benchmark. Frozen
evaluation prompts must not be passed here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from benchmarks.compare_dense_sparse import (
    _free_model,
    _load_model,
    _model_device,
    _set_attention_backend,
    runtime_metadata,
)
from benchmarks.compare_dense_sparse_live_tree import (
    _sync,
    live_route,
    prepare_cases,
)
from configs.long_context import QWEN_NATIVE_CONTEXT, configure_tokenizer
from interfaces.residual_summaries import (
    build_residual_summary_nodes,
    validate_residual_summary_nodes,
)
from interfaces.validator import validate_selected_blocks
from scripts.eval_tree_traversal import load_gate
from scripts.route_overrides import dense_candidate_routes
from selector.targets import load_selector_features
from sparse.attention import (
    SPARSE_PREFILL_ATTENTION,
    register_sparse_prefill_attention,
)
from sparse.config import SUMMARY_PROTOTYPES
from sparse.summaries import residual_attention_density


def hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return hidden_tensor(output[0])
    raise TypeError(f"could not find hidden tensor in {type(output)!r}")


def sampled_positions(seq_len: int, count: int) -> torch.Tensor:
    """Return stable, unique token positions including both endpoints."""
    if seq_len < 1 or count < 1:
        raise ValueError("seq_len and sample count must be positive")
    count = min(seq_len, count)
    positions = torch.linspace(0, seq_len - 1, steps=count).round().long()
    return positions.unique(sorted=True)


def error_sums(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}"
        )
    delta = actual.float() - reference.float()
    reference_float = reference.float()
    return {
        "squared_error_sum": float(delta.square().sum().item()),
        "reference_square_sum": float(reference_float.square().sum().item()),
        "element_count": int(delta.numel()),
        "max_abs": float(delta.abs().max().item()),
    }


def combine_error_sums(items) -> dict:
    items = list(items)
    if not items:
        raise ValueError("at least one error item is required")
    return {
        "squared_error_sum": sum(item["squared_error_sum"] for item in items),
        "reference_square_sum": sum(
            item["reference_square_sum"] for item in items
        ),
        "element_count": sum(item["element_count"] for item in items),
        "max_abs": max(item["max_abs"] for item in items),
    }


def finish_error(sums: dict) -> dict:
    count = int(sums["element_count"])
    if count < 1:
        raise ValueError("error metric has no elements")
    rmse = math.sqrt(float(sums["squared_error_sum"]) / count)
    reference_rms = math.sqrt(float(sums["reference_square_sum"]) / count)
    return {
        **sums,
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": (
            rmse / reference_rms if reference_rms else float("inf")
        ),
    }


def select_smallest_near_best(
    rmse_by_prototypes: dict[int, float],
    *,
    tolerance: float = 0.05,
) -> tuple[int, float]:
    """Select the smallest P no more than ``tolerance`` above the best RMSE."""
    if not rmse_by_prototypes:
        raise ValueError("no prototype settings were provided")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    invalid = [
        prototypes
        for prototypes, value in rmse_by_prototypes.items()
        if prototypes not in SUMMARY_PROTOTYPES
        or not math.isfinite(float(value))
        or float(value) < 0
    ]
    if invalid:
        raise ValueError(f"invalid prototype/RMSE settings: {invalid}")
    best = min(float(value) for value in rmse_by_prototypes.values())
    threshold = best * (1.0 + tolerance)
    eligible = sorted(
        prototypes
        for prototypes, value in rmse_by_prototypes.items()
        if float(value) <= threshold
    )
    return eligible[0], threshold


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _sample_hidden(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if value.dim() != 3:
        raise ValueError(
            "sampled diagnostics expect [batch,tokens,hidden], "
            f"got {tuple(value.shape)}"
        )
    return value.index_select(1, positions.to(value.device))


@contextmanager
def capture_dense_samples(model, positions, attention_refs, hidden_refs):
    handles = []
    for layer_index, layer in enumerate(model.base_model.layers):
        def capture_attention(
            _module, _inputs, output, index=layer_index
        ):
            attention_refs[index] = _sample_hidden(
                hidden_tensor(output).detach(), positions
            ).to(device="cpu", dtype=torch.float16).clone()

        def capture_hidden(_module, _inputs, output, index=layer_index):
            hidden_refs[index] = _sample_hidden(
                hidden_tensor(output).detach(), positions
            ).to(device="cpu", dtype=torch.float16).clone()

        handles.append(layer.self_attn.register_forward_hook(capture_attention))
        handles.append(layer.register_forward_hook(capture_hidden))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def compare_sparse_samples(
    model,
    positions,
    attention_refs,
    hidden_refs,
    attention_errors,
    hidden_errors,
):
    handles = []
    for layer_index, layer in enumerate(model.base_model.layers):
        def compare_attention(
            _module, _inputs, output, index=layer_index
        ):
            actual = _sample_hidden(hidden_tensor(output).detach(), positions)
            reference = attention_refs[index].to(actual.device)
            attention_errors[index] = error_sums(actual, reference)

        def compare_hidden(_module, _inputs, output, index=layer_index):
            actual = _sample_hidden(hidden_tensor(output).detach(), positions)
            reference = hidden_refs[index].to(actual.device)
            hidden_errors[index] = error_sums(actual, reference)

        handles.append(layer.self_attn.register_forward_hook(compare_attention))
        handles.append(layer.register_forward_hook(compare_hidden))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _layer_report(errors: dict[int, dict]) -> list[dict]:
    return [
        {"layer": layer_index, **finish_error(errors[layer_index])}
        for layer_index in sorted(errors)
    ]


def _build_route(
    gate,
    gate_config,
    case,
    args,
    *,
    device,
):
    features = load_selector_features(
        case["target"], device=args.selector_device
    )
    selected, timing = live_route(
        gate,
        gate_config,
        case["target"],
        beam=args.beam,
        radix=args.radix,
        k_selected=args.k_selected,
        local_window=args.local_window,
        selector_device=args.selector_device,
        model_device=device,
        validate=True,
        selector_dtype=args.selector_dtype,
        selector_layer_chunk=args.selector_layer_chunk,
        features=features,
        sink_blocks=args.sink_blocks,
    )
    if args.route_mode == "dense-candidates":
        with torch.no_grad():
            reader_scores = gate(
                features["q_feat"][:, :, -1:], features["k_feat"]
            )
        pooled_scores = reader_scores[:, :, 0, :].amax(dim=(0, 1))
        candidate_count = min(args.candidate_blocks, pooled_scores.shape[0])
        candidates = pooled_scores.topk(candidate_count).indices.tolist()
        selected = dense_candidate_routes(
            selected,
            candidates,
            neighborhood=args.candidate_neighborhood,
        )
        timing["candidate_blocks"] = candidates
    validate_selected_blocks(
        selected.detach().cpu(), local_window=args.local_window
    )
    residual = build_residual_summary_nodes(
        selected, validate_selected=False
    )
    validate_residual_summary_nodes(
        selected.detach().cpu(),
        residual.detach().cpu(),
        validate_selected=False,
    )
    del features
    gc.collect()
    return selected, residual, timing


def _forward(
    model,
    encoded,
    *,
    selected=None,
    residual=None,
    prototypes=None,
    positions=None,
    attention_refs=None,
    hidden_refs=None,
):
    kwargs = {"use_cache": False}
    attention_errors = {}
    hidden_errors = {}
    if selected is None:
        context = capture_dense_samples(
            model, positions, attention_refs, hidden_refs
        )
    else:
        kwargs.update(
            selected_blocks=selected,
            block_size=64,
            validate_selected_blocks_input=False,
        )
        if prototypes is not None:
            kwargs.update(
                residual_summaries=True,
                residual_summary_nodes=residual,
                summary_prototypes=prototypes,
                summary_mode="mean",
                validate_residual_summary_nodes_input=False,
            )
        context = compare_sparse_samples(
            model,
            positions,
            attention_refs,
            hidden_refs,
            attention_errors,
            hidden_errors,
        )
    _sync(_model_device(model))
    started = time.perf_counter()
    with torch.inference_mode(), context:
        output = model.base_model(**encoded, **kwargs)
        final_hidden = output.last_hidden_state[:, -1:, :].detach()
        logits = model.get_output_embeddings()(final_hidden)
    _sync(_model_device(model))
    seconds = time.perf_counter() - started
    return {
        "output": output,
        "final_hidden": final_hidden,
        "next_token_id": int(logits[:, -1].argmax(dim=-1).item()),
        "seconds": seconds,
        "attention_errors": attention_errors,
        "hidden_errors": hidden_errors,
    }


def _release_forward(result):
    for name in ("output", "final_hidden"):
        value = result.pop(name, None)
        if value is not None:
            del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggregate_cases(cases, settings):
    aggregate = {}
    lengths = sorted({case["requested_length"] for case in cases})
    for setting in settings:
        by_length = {}
        for requested_length in lengths:
            matching = [
                case["settings"][setting]
                for case in cases
                if case["requested_length"] == requested_length
            ]
            by_length[str(requested_length)] = {
                "cases": len(matching),
                "attention_output": finish_error(combine_error_sums(
                    item["attention_output"] for item in matching
                )),
                "sampled_hidden": finish_error(combine_error_sums(
                    item["sampled_hidden"] for item in matching
                )),
                "final_hidden": finish_error(combine_error_sums(
                    item["final_hidden"] for item in matching
                )),
            }
        matching = [case["settings"][setting] for case in cases]
        aggregate[setting] = {
            "cases": len(matching),
            "attention_output": finish_error(combine_error_sums(
                item["attention_output"] for item in matching
            )),
            "sampled_hidden": finish_error(combine_error_sums(
                item["sampled_hidden"] for item in matching
            )),
            "final_hidden": finish_error(combine_error_sums(
                item["final_hidden"] for item in matching
            )),
            "by_requested_length": by_length,
        }
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct"
    )
    parser.add_argument("--prompt-bank")
    parser.add_argument(
        "--prototypes",
        type=int,
        nargs="+",
        choices=SUMMARY_PROTOTYPES,
        default=list(SUMMARY_PROTOTYPES),
    )
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument(
        "--route-mode",
        choices=("learned", "dense-candidates"),
        default="dense-candidates",
    )
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--candidate-neighborhood", type=int, default=0)
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--sink-blocks", type=int, default=0)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--selector-device", default="cuda")
    parser.add_argument(
        "--selector-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--selector-layer-chunk", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--yarn-factor", type=float, default=1.0)
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=QWEN_NATIVE_CONTEXT,
    )
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_residual_diagnostic_offload"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.sample_tokens < 1:
        raise SystemExit("--sample-tokens must be positive")
    prototypes = sorted(set(args.prototypes))
    paths = [os.path.abspath(path) for path in args.targets]
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise SystemExit("all --targets must be existing files")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer,
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
    )
    cases = prepare_cases(paths, tokenizer, prompt_bank=args.prompt_bank)
    gate, gate_config = load_gate(args.gate, args.selector_device)
    register_sparse_prefill_attention()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = _load_model(
        args.model,
        "sdpa",
        dtype,
        args.load_offload_dir,
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
    )
    device = _model_device(model)

    report = {
        "kind": "spruce_residual_summary_diagnostic_v1",
        "scope": "development_only_do_not_use_frozen_prompts",
        "model": args.model,
        "runtime": runtime_metadata(),
        "gate": os.path.abspath(args.gate),
        "route": {
            "mode": args.route_mode,
            "candidate_blocks": args.candidate_blocks,
            "candidate_neighborhood": args.candidate_neighborhood,
            "k_selected": args.k_selected,
            "sink_blocks": args.sink_blocks,
            "beam": args.beam,
            "local_window": args.local_window,
        },
        "sample_tokens_requested": args.sample_tokens,
        "prototypes": prototypes,
        "cases": [],
    }
    atomic_json_dump(report, args.out)

    for case_index, case in enumerate(cases):
        encoded = {
            name: value.to(device)
            for name, value in case["encoded"].items()
        }
        positions = sampled_positions(case["seq_len"], args.sample_tokens)
        selected, residual, route_timing = _build_route(
            gate, gate_config, case, args, device=device
        )
        density_by_p = {
            str(prototype_count): residual_attention_density(
                selected,
                residual,
                seq_len=case["seq_len"],
                block_size=case["block_size"],
                prototypes=prototype_count,
            )
            for prototype_count in prototypes
        }

        attention_refs = {}
        hidden_refs = {}
        _set_attention_backend(model, "sdpa")
        dense = _forward(
            model,
            encoded,
            positions=positions,
            attention_refs=attention_refs,
            hidden_refs=hidden_refs,
        )
        dense_final = dense["final_hidden"].to(
            device="cpu", dtype=torch.float16
        ).clone()
        dense_token = dense["next_token_id"]
        dense_seconds = dense["seconds"]
        _release_forward(dense)

        case_report = {
            "case_id": case["case_id"],
            "target": case["target"],
            "seq_len": case["seq_len"],
            "requested_length": case["requested_length"],
            "sample_tokens": int(positions.numel()),
            "sample_positions": [int(value) for value in positions.tolist()],
            "dense_next_token_id": dense_token,
            "dense_seconds": dense_seconds,
            "route_timing": route_timing,
            "density_by_prototypes": density_by_p,
            "settings": {},
        }
        settings = [("exact", None)] + [
            (f"P{prototype_count}", prototype_count)
            for prototype_count in prototypes
        ]
        for setting, prototype_count in settings:
            _set_attention_backend(model, SPARSE_PREFILL_ATTENTION)
            sparse = _forward(
                model,
                encoded,
                selected=selected,
                residual=residual,
                prototypes=prototype_count,
                positions=positions,
                attention_refs=attention_refs,
                hidden_refs=hidden_refs,
            )
            final_reference = dense_final.to(sparse["final_hidden"].device)
            final_error = error_sums(
                sparse["final_hidden"], final_reference
            )
            attention_error = combine_error_sums(
                sparse["attention_errors"].values()
            )
            hidden_error = combine_error_sums(
                sparse["hidden_errors"].values()
            )
            case_report["settings"][setting] = {
                "seconds": sparse["seconds"],
                "next_token_id": sparse["next_token_id"],
                "next_token_matches_dense": (
                    sparse["next_token_id"] == dense_token
                ),
                "attention_output": attention_error,
                "sampled_hidden": hidden_error,
                "final_hidden": final_error,
                "attention_layers": _layer_report(
                    sparse["attention_errors"]
                ),
                "hidden_layers": _layer_report(
                    sparse["hidden_errors"]
                ),
            }
            _release_forward(sparse)
        report["cases"].append(case_report)
        atomic_json_dump(report, args.out)
        print(
            f"{case_index + 1}/{len(cases)} {case['case_id']} "
            f"len={case['seq_len']} complete",
            flush=True,
        )
        del (
            encoded,
            selected,
            residual,
            dense_final,
            attention_refs,
            hidden_refs,
        )
        gc.collect()
        torch.cuda.empty_cache()

    setting_names = ["exact"] + [
        f"P{prototype_count}" for prototype_count in prototypes
    ]
    report["aggregate"] = aggregate_cases(
        report["cases"], setting_names
    )
    rmse_by_p = {
        prototype_count: report["aggregate"][f"P{prototype_count}"][
            "attention_output"
        ]["rmse"]
        for prototype_count in prototypes
    }
    selected_prototypes, threshold = select_smallest_near_best(
        rmse_by_p, tolerance=0.05
    )
    baseline_by_length = report["aggregate"]["exact"][
        "by_requested_length"
    ]
    selected_by_length = report["aggregate"][f"P{selected_prototypes}"][
        "by_requested_length"
    ]
    hidden_reduction = {}
    for length, baseline_metrics in baseline_by_length.items():
        baseline_rmse = baseline_metrics["final_hidden"]["relative_rmse"]
        selected_rmse = selected_by_length[length]["final_hidden"][
            "relative_rmse"
        ]
        hidden_reduction[length] = (
            (baseline_rmse - selected_rmse) / baseline_rmse
            if baseline_rmse
            else float("-inf")
        )
    report["selection"] = {
        "rule": "smallest P with attention-output RMSE <= 1.05 * best",
        "rmse_by_prototypes": {
            str(key): value for key, value in rmse_by_p.items()
        },
        "threshold": threshold,
        "selected_prototypes": selected_prototypes,
        "final_hidden_relative_rmse_reduction_vs_exact_by_length": (
            hidden_reduction
        ),
    }
    atomic_json_dump(report, args.out)
    model = _free_model(model)
    print(json.dumps(report["selection"], indent=2))
    print(f"full diagnostic -> {args.out}")


if __name__ == "__main__":
    main()

