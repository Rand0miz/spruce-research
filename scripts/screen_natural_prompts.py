"""Screen natural-prose prompts with dense Qwen before teacher extraction.

Only prompts whose concise reference answer is recovered by dense generation
are marked ``accepted``. The JSON manifest stores every exact prompt, dense
answer, score, and construction seed. Pass that manifest to
``extract_teacher_targets.py --verified-manifest ...`` to extract full teacher
targets without regenerating or silently changing the screened text.
"""
import argparse
import json
import os
import sys
import tempfile

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.compare_dense_sparse import (
    _free_model,
    _generate,
    _load_model,
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
from eval.score import score_concise_retrieval, score_retrieval
from scripts.extract_teacher_targets import (
    depth_tag,
    load_prompt_bank,
    needle_block_index,
)


DEFAULT_BANK = os.path.join(
    ROOT, "scripts", "prompt_banks", "natural_train.json")
MANIFEST_KIND = "spruce_dense_screen_manifest_v1"


def candidate_id(case_id, requested_length, depth, seed):
    return (
        f"{case_id}_L{int(requested_length)}_"
        f"d{depth_tag(float(depth))}_s{int(seed)}"
    )


def build_candidates(cases, lengths, depths, variants, seed):
    """Return deterministic candidate specifications without building text."""
    if variants < 1:
        raise ValueError("variants must be >= 1")
    candidates = []
    for requested_length in lengths:
        for case in cases:
            for depth in depths:
                for variant in range(variants):
                    prompt_seed = int(seed) + variant
                    candidates.append({
                        "candidate_id": candidate_id(
                            case["id"], requested_length, depth, prompt_seed),
                        "source_case_id": case["id"],
                        "requested_length": int(requested_length),
                        "depth": float(depth),
                        "seed": prompt_seed,
                        "variant": variant,
                    })
    return candidates


def summarize_candidates(records):
    accepted = sum(bool(record.get("accepted")) for record in records)
    completed = sum(record.get("status") == "completed" for record in records)
    by_length = {}
    for record in records:
        length = str(record["requested_length"])
        bucket = by_length.setdefault(
            length, {"completed": 0, "accepted": 0, "rejected": 0})
        if record.get("status") == "completed":
            bucket["completed"] += 1
            if record.get("accepted"):
                bucket["accepted"] += 1
            else:
                bucket["rejected"] += 1
    return {
        "candidates": len(records),
        "completed": completed,
        "accepted": accepted,
        "rejected": completed - accepted,
        "acceptance_rate": accepted / completed if completed else 0.0,
        "by_length": by_length,
    }


def atomic_save_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, path)


def _load_existing(path, resume):
    if not os.path.isfile(path):
        return None
    if not resume:
        raise FileExistsError(
            f"{path} already exists; pass --resume or choose another --out")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{path} is not a {MANIFEST_KIND} manifest")
    return manifest


def validate_resume_config(manifest, expected_config):
    actual = manifest.get("config")
    if actual != expected_config:
        raise ValueError(
            "existing screening manifest configuration does not match this "
            f"run:\nexisting={actual}\nrequested={expected_config}")


def _write_accepted_manifest(path, manifest):
    if not path:
        return
    accepted = [
        record for record in manifest["candidates"]
        if record.get("accepted")
    ]
    output = {
        **manifest,
        "candidates": accepted,
        "summary": summarize_candidates(accepted),
        "source_manifest": os.path.abspath(manifest["output"]),
        "accepted_only": True,
    }
    atomic_save_manifest(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--prompt-bank", default=DEFAULT_BANK)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[16_384, 32_768])
    parser.add_argument(
        "--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--yarn-factor", type=float, default=1.0)
    parser.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT)
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(
            tempfile.gettempdir(), "spruce_dense_screen_offload"))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--accepted-out",
        help="optional second manifest containing accepted prompts only")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("dense screening requires a CUDA GPU")
    if args.block_size < 1:
        raise SystemExit("--block-size must be >= 1")
    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise SystemExit("--depths must be in [0, 1]")
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=(
            args.original_max_position_embeddings))
    if any(length < 1 or length > maximum_context for length in args.lengths):
        raise SystemExit(
            f"--lengths must be in [1, {maximum_context}] for the configured "
            "RoPE settings")

    cases = load_prompt_bank(args.prompt_bank)
    case_by_id = {case["id"]: case for case in cases}
    specifications = build_candidates(
        cases, args.lengths, args.depths, args.variants, args.seed)
    screen_config = {
        "model": args.model,
        "prompt_bank": os.path.abspath(args.prompt_bank),
        "lengths": list(args.lengths),
        "depths": list(args.depths),
        "variants": args.variants,
        "seed": args.seed,
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": "qwen_chat_v1",
        "acceptance": "retrieved_and_concise",
        "dtype": args.dtype,
        "rope": yarn_metadata(
            yarn_factor=args.yarn_factor,
            original_max_position_embeddings=(
                args.original_max_position_embeddings)),
    }
    existing = _load_existing(args.out, args.resume)
    if existing:
        validate_resume_config(existing, screen_config)
    records_by_id = {
        record["candidate_id"]: record
        for record in (existing or {}).get("candidates", [])
        if record.get("status") == "completed"
    }

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

    manifest = {
        "kind": MANIFEST_KIND,
        "output": os.path.abspath(args.out),
        "config": screen_config,
        "runtime": runtime_metadata(),
        "candidates": [],
        "summary": {},
    }
    try:
        for index, specification in enumerate(specifications, start=1):
            key = specification["candidate_id"]
            if key in records_by_id:
                record = records_by_id[key]
                manifest["candidates"].append(record)
                print(
                    f"resume {index}/{len(specifications)} {key} "
                    f"accepted={record['accepted']}",
                    flush=True,
                )
                continue

            case = case_by_id[specification["source_case_id"]]
            prompt, needle, natural_units, user_prompt = (
                build_natural_prompt_calibrated(
                tokenizer,
                specification["requested_length"],
                case,
                specification["depth"],
                seed=specification["seed"],
                prompt_formatter=lambda content: format_instruct_chat_prompt(
                    tokenizer, content),
                return_content=True,
            ))
            encoded = tokenizer(prompt, return_tensors="pt")
            seq_len = int(encoded.input_ids.shape[1])
            token_ids, timing = _generate(
                model, encoded, args.max_new_tokens)
            answer = tokenizer.decode(token_ids, skip_special_tokens=True)
            score = score_retrieval(
                answer, needle, reference_answers=case["answers"])
            concise_score = score_concise_retrieval(
                answer, case["answers"])
            accepted = bool(score["exact"] and concise_score["exact"])
            record = {
                **specification,
                "id": key,
                "builder": "verified_prompt_v1",
                "status": "completed",
                "accepted": accepted,
                "model": args.model,
                "rope": manifest["config"]["rope"],
                "prompt_format": "qwen_chat_v1",
                "seq_len": seq_len,
                "block_size": args.block_size,
                "needle_block": needle_block_index(
                    tokenizer, prompt, needle, args.block_size),
                "needle": needle,
                "evidence": case["evidence"],
                "question": case["question"],
                "answers": list(case["answers"]),
                "reference_answers": list(case["answers"]),
                "prompt_text": prompt,
                "user_prompt_text": user_prompt,
                "natural_units": natural_units,
                "dense_answer": answer,
                "dense_exact": bool(score["exact"]),
                "dense_fuzzy": float(score["fuzzy"]),
                "dense_concise_exact": bool(concise_score["exact"]),
                "dense_timing": timing,
            }
            manifest["candidates"].append(record)
            manifest["summary"] = summarize_candidates(
                manifest["candidates"])
            atomic_save_manifest(args.out, manifest)
            _write_accepted_manifest(args.accepted_out, manifest)
            print(
                f"screen {index}/{len(specifications)} {key} "
                f"len={seq_len} accepted={record['accepted']} "
                f"answer={answer.strip()!r}",
                flush=True,
            )
            del encoded, token_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model = _free_model(model)

    manifest["summary"] = summarize_candidates(manifest["candidates"])
    atomic_save_manifest(args.out, manifest)
    _write_accepted_manifest(args.accepted_out, manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"screening manifest -> {os.path.abspath(args.out)}", flush=True)
    if args.accepted_out:
        print(
            f"accepted manifest  -> {os.path.abspath(args.accepted_out)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
