"""Run one resumable unscreened natural-prompt YaRN length bucket.

Every case compares ordinary dense Qwen against the frozen pre-Qwen lexical
hierarchy plus exact-text compilation. Prompt synthesis is deterministic test
harness work and is reported separately, outside both request timers. The
output JSON is atomically updated after every completed paired case.
"""
import argparse
import hashlib
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

from benchmarks.benchmark_pre_qwen_e2e import (
    _summarize_samples,
    aggregate_cases,
    run_compiled_request,
    run_dense_request,
)
from benchmarks.compare_dense_sparse import (
    _free_model,
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
from eval.natural_context import (
    build_natural_prompt_calibrated,
    format_instruct_chat_prompt,
)
from scripts.extract_teacher_targets import (
    depth_tag,
    load_prompt_bank,
    needle_block_index,
)


REPORT_KIND = "spruce_pre_qwen_natural_yarn_length_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)


def _case_id(case, length, depth, seed):
    return (
        f"{case['id']}_L{int(length)}_"
        f"d{depth_tag(float(depth))}_s{int(seed)}"
    )


def _prompt_target(
        tokenizer, case, requested_length, depth, seed,
        block_size, generation_reserve):
    started = time.perf_counter()
    prompt_budget = int(requested_length) - int(generation_reserve)
    if prompt_budget < 1:
        raise ValueError("generation reserve consumes the prompt budget")
    prompt, needle, natural_units, user_prompt = (
        build_natural_prompt_calibrated(
            tokenizer,
            prompt_budget,
            case,
            float(depth),
            seed=int(seed),
            prompt_formatter=lambda content: format_instruct_chat_prompt(
                tokenizer, content),
            return_content=True,
        )
    )
    encoded = tokenizer(prompt)
    seq_len = len(encoded["input_ids"])
    if seq_len > prompt_budget:
        raise ValueError(
            f"generated prompt has {seq_len} tokens, budget {prompt_budget}")
    target = {
        "id": _case_id(case, requested_length, depth, seed),
        "source_case_id": case["id"],
        "genre": case.get("genre"),
        "builder": "natural_prose_v1_unscreened_paper",
        "screened": False,
        "requested_length": int(requested_length),
        "prompt_token_budget": prompt_budget,
        "generation_reserve": int(generation_reserve),
        "depth": float(depth),
        "seed": int(seed),
        "seq_len": seq_len,
        "block_size": int(block_size),
        "needle_block": needle_block_index(
            tokenizer, prompt, needle, int(block_size)),
        "needle": needle,
        "evidence": case["evidence"],
        "question": case["question"],
        "answers": list(case["answers"]),
        "reference_answers": list(case["answers"]),
        "prompt_format": "qwen_chat_v1",
        "prompt_text": prompt,
        "user_prompt_text": user_prompt,
        "natural_units": int(natural_units),
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest().upper(),
    }
    return target, time.perf_counter() - started


def _expected_config(args, prompt_bank_hash):
    return {
        "model": args.model,
        "prompt_bank": os.path.abspath(args.prompt_bank),
        "prompt_bank_sha256": prompt_bank_hash,
        "prompt_screening": "none",
        "requested_length": int(args.length),
        "depths": [float(depth) for depth in args.depths],
        "seed": int(args.seed),
        "block_size": int(args.block_size),
        "generation_reserve": int(args.max_new_tokens),
        "candidate_blocks": int(args.candidate_blocks),
        "block_radius": int(args.block_radius),
        "boundary": args.boundary,
        "beam": int(args.beam),
        "feature_dim": int(args.feature_dim),
        "unigram_fraction": float(args.unigram_fraction),
        "idf_power": float(args.idf_power),
        "radix": int(args.radix),
        "repeats": int(args.repeats),
        "max_new_tokens": int(args.max_new_tokens),
        "dtype": args.dtype,
        "yarn_factor": float(args.yarn_factor),
        "original_max_position_embeddings": int(
            args.original_max_position_embeddings),
    }


def _load_resume(path, config):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"{path} is not a {REPORT_KIND} report")
    if report.get("config") != config:
        raise ValueError(
            "existing report configuration differs from requested run")
    return report


def _base_report(args, config, cases):
    return {
        "kind": REPORT_KIND,
        "status": "running",
        "model": args.model,
        "runtime": runtime_metadata(),
        "config": config,
        "rope": yarn_metadata(
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=(
                args.original_max_position_embeddings)),
        "suite": {
            "distribution": "sealed_unscreened_natural_prose",
            "prompt_bank_cases": len(cases),
            "paired_cases_expected": len(cases) * len(args.depths),
            "prompt_synthesis_timed_separately": True,
            "request_timing_scope": (
                "same fully charged scope as spruce_pre_qwen_e2e_v1"),
            "dense_label": "Dense YaRN teacher",
            "compiled_label": "SPRUCE beam-16 evidence compiler",
        },
        "cases": [],
        "aggregate": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument(
        "--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--candidate-blocks", type=int, default=4)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("block", "paragraph"),
        default="paragraph")
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--unigram-fraction", type=float, default=0.5)
    parser.add_argument("--idf-power", type=float, default=2.0)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"),
        default="float16")
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_natural_yarn_offload"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not os.path.isfile(args.prompt_bank):
        raise SystemExit(f"prompt bank not found: {args.prompt_bank}")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise SystemExit("--repeats and --max-new-tokens must be >= 1")
    if args.beam < args.candidate_blocks:
        raise SystemExit("--beam must be >= --candidate-blocks")
    if not args.depths or any(
            not 0.0 <= float(depth) <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0, 1]")
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
    if not 1 <= args.length <= maximum_context:
        raise SystemExit(
            f"--length must be in [1,{maximum_context}], got {args.length}")

    cases = load_prompt_bank(args.prompt_bank)
    prompt_bank_hash = _sha256(args.prompt_bank)
    config = _expected_config(args, prompt_bank_hash)
    try:
        report = _load_resume(args.out, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if report is None:
        report = _base_report(args, config, cases)
        _atomic_json(args.out, report)
    completed_ids = [
        case["candidate_id"] for case in report["cases"]
    ]
    if len(completed_ids) != len(set(completed_ids)):
        raise SystemExit("existing report contains duplicate candidate IDs")
    completed = set(completed_ids)
    expected_ids = [
        _case_id(case, args.length, depth, args.seed)
        for case in cases for depth in args.depths
    ]
    unexpected = completed - set(expected_ids)
    if unexpected:
        raise SystemExit(
            "existing report contains unexpected candidate IDs: "
            + ", ".join(sorted(unexpected)))
    if (
            set(expected_ids) == completed
            and report.get("status") == "completed"
            and report.get("prompt_build_seconds") is not None):
        print(f"resume: complete -> {args.out}", flush=True)
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tokenizer, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
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

    total = len(expected_ids)
    try:
        for source_case in cases:
            for depth in args.depths:
                candidate_id = _case_id(
                    source_case, args.length, depth, args.seed)
                if candidate_id in completed:
                    print(f"resume: {candidate_id}", flush=True)
                    continue
                print(
                    f"[{len(completed) + 1}/{total}] {candidate_id}",
                    flush=True)
                target, prompt_build_seconds = _prompt_target(
                    tokenizer, source_case, args.length, depth, args.seed,
                    args.block_size, args.max_new_tokens)
                dense_samples = []
                compiled_samples = []
                for repeat in range(args.repeats):
                    modes = (
                        ("dense", "compiled")
                        if (len(completed) + repeat) % 2
                        else ("compiled", "dense")
                    )
                    for mode in modes:
                        if mode == "dense":
                            dense_samples.append(run_dense_request(
                                model, tokenizer, target,
                                max_new_tokens=args.max_new_tokens))
                        else:
                            compiled_samples.append(run_compiled_request(
                                model, tokenizer, target,
                                candidate_blocks=args.candidate_blocks,
                                block_radius=args.block_radius,
                                boundary=args.boundary,
                                beam=args.beam,
                                feature_dim=args.feature_dim,
                                unigram_fraction=args.unigram_fraction,
                                idf_power=args.idf_power,
                                radix=args.radix,
                                max_new_tokens=args.max_new_tokens))
                dense = _summarize_samples(dense_samples)
                compiled = _summarize_samples(compiled_samples)
                row = {
                    "candidate_id": candidate_id,
                    "source_case_id": source_case["id"],
                    "genre": source_case.get("genre"),
                    "requested_length": int(args.length),
                    "prompt_token_budget": target["prompt_token_budget"],
                    "seq_len": target["seq_len"],
                    "depth": float(depth),
                    "seed": int(args.seed),
                    "natural_units": target["natural_units"],
                    "prompt_sha256": target["prompt_sha256"],
                    "prompt_build_seconds": prompt_build_seconds,
                    "needle_block": target["needle_block"],
                    "dense": dense,
                    "compiled": compiled,
                    "paired_outcome": (
                        "both_correct"
                        if dense["exact"] and compiled["exact"]
                        else "dense_only"
                        if dense["exact"]
                        else "compiled_only"
                        if compiled["exact"]
                        else "neither_correct"
                    ),
                }
                report["cases"].append(row)
                completed.add(candidate_id)
                report["aggregate"] = aggregate_cases(report["cases"])
                # Keep the report resumable-but-incomplete until the final
                # ordered aggregate and prompt-build summary are persisted.
                report["status"] = "running"
                _atomic_json(args.out, report)
                print(
                    f"  dense={int(dense['exact'])} "
                    f"compiled={int(compiled['exact'])} "
                    f"outcome={row['paired_outcome']} "
                    f"speedup={dense['request_seconds'] / compiled['request_seconds']:.3f}x "
                    f"tokens={compiled['input_tokens']} "
                    f"expanded_evidence={int(compiled['expanded_contains_needle'])}",
                    flush=True)
    finally:
        model = _free_model(model)

    ordered = {identifier: index for index, identifier in enumerate(expected_ids)}
    report["cases"].sort(key=lambda row: ordered[row["candidate_id"]])
    report["aggregate"] = aggregate_cases(report["cases"])
    report["status"] = "completed"
    report["prompt_build_seconds"] = {
        "sum": sum(case["prompt_build_seconds"] for case in report["cases"]),
        "median": statistics.median(
            case["prompt_build_seconds"] for case in report["cases"]),
    }
    _atomic_json(args.out, report)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
