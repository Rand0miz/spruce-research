import argparse
import gc
import json
import os
import random
import re
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.haystack import build_haystack_calibrated
from eval.natural_context import build_natural_prompt_calibrated
from configs.long_context import (
    QWEN_NATIVE_CONTEXT,
    configure_tokenizer,
    context_limit,
    load_model_config,
    yarn_metadata,
)
from teacher.chunked_extract import get_pooled_targets, get_selector_features

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_BANK = os.path.join(ROOT, "scripts", "prompt_banks", "train.json")
DEFAULT_HELDOUT_BANK = os.path.join(ROOT, "scripts", "prompt_banks", "heldout.json")
VERIFIED_MANIFEST_KIND = "spruce_dense_screen_manifest_v1"


def _safe_id(s):
    """Filesystem-safe prompt id fragment."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s.strip())
    return s.strip("-") or "prompt"


def load_prompt_bank(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'cases' list")
    for i, case in enumerate(cases):
        builder = case.get("builder", "calibrated_units_v1")
        required = (
            {"id", "needle", "evidence", "question", "answers"}
            if builder == "natural_prose_v1"
            else {"id", "needle", "filler", "question"}
        )
        missing = required - set(case)
        if missing:
            raise ValueError(f"{path} case {i} missing keys: {sorted(missing)}")
        for key in required:
            if key == "answers":
                if (not isinstance(case[key], list) or not case[key]
                        or any(
                            not isinstance(answer, str) or not answer
                            for answer in case[key])):
                    raise ValueError(
                        f"{path} case {i} key 'answers' must be a non-empty "
                        "list of non-empty strings")
            elif not isinstance(case[key], str) or not case[key]:
                raise ValueError(f"{path} case {i} key {key!r} must be a non-empty string")
        if builder not in {"calibrated_units_v1", "natural_prose_v1"}:
            raise ValueError(
                f"{path} case {i} has unknown builder {builder!r}")
        if (builder == "natural_prose_v1"
                and case["needle"] not in case["evidence"]):
            raise ValueError(
                f"{path} case {i} needle must occur inside evidence")
    return cases


def load_verified_manifest(path):
    """Load dense-correct exact prompts produced by the screening script."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("kind") != VERIFIED_MANIFEST_KIND:
        raise ValueError(
            f"{path} is not a {VERIFIED_MANIFEST_KIND} manifest")
    records = [
        record for record in data.get("candidates", [])
        if record.get("status") == "completed" and record.get("accepted")
    ]
    if not records:
        raise ValueError(f"{path} contains no accepted prompts")
    required = {
        "id", "prompt_text", "needle", "question", "answers", "depth",
        "requested_length", "seq_len", "block_size", "model", "rope",
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"{path} accepted record {index} is missing "
                f"{sorted(missing)}")
        if record["needle"] not in record["prompt_text"]:
            raise ValueError(
                f"{path} accepted record {index} does not contain its needle")
        if record.get("prompt_format") == "qwen_chat_v1":
            user_prompt = record.get("user_prompt_text")
            if (not isinstance(user_prompt, str)
                    or not user_prompt.endswith(record["question"])
                    or user_prompt not in record["prompt_text"]):
                raise ValueError(
                    f"{path} accepted chat record {index} does not preserve "
                    "its exact user prompt and final question")
        elif not record["prompt_text"].endswith(record["question"]):
            raise ValueError(
                f"{path} accepted record {index} does not end with its question")
    return records


def build_verified_jobs(records, maximum_context):
    jobs = []
    for record in records:
        requested_length = int(record["requested_length"])
        if requested_length > maximum_context:
            raise ValueError(
                f"verified prompt {record['id']} requests {requested_length} "
                f"tokens but configured context is {maximum_context}")
        jobs.append((requested_length, record, float(record["depth"])))
    return jobs


def _check_depth(depth, name="depth"):
    depth = float(depth)
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {depth}")
    return depth


def choose_depth(args, rng):
    """Pick one needle depth according to CLI precedence."""
    if args.depth is not None:
        return _check_depth(args.depth, "--depth")

    if args.depths is not None:
        if not args.depths:
            raise ValueError("--depths must contain at least one value")
        depths = [_check_depth(d, "--depths") for d in args.depths]
        return float(rng.choice(depths))

    lo, hi = args.depth_range
    lo = _check_depth(lo, "--depth-range MIN")
    hi = _check_depth(hi, "--depth-range MAX")
    if lo > hi:
        raise ValueError(f"--depth-range MIN must be <= MAX, got {lo} {hi}")
    return float(rng.uniform(lo, hi))


def select_cases(cases, run_all, rng):
    """Return all prompt-bank cases or one randomly selected case."""
    return list(cases) if run_all else [rng.choice(cases)]


def build_jobs(args, cases, rng, maximum_context):
    """Construct sampled, balanced-cycle, or exhaustive depth jobs."""
    chosen_cases = select_cases(cases, args.all, rng)
    mode = getattr(args, "depth_mode", "sample")
    configured_depths = None
    if mode != "sample":
        if args.depth is not None:
            configured_depths = [_check_depth(args.depth, "--depth")]
        elif args.depths:
            configured_depths = [
                _check_depth(depth, "--depths") for depth in args.depths]
        else:
            raise ValueError(
                f"--depth-mode {mode} requires --depth or --depths")

    jobs = []
    for target_len in args.lengths:
        if target_len > maximum_context:
            raise ValueError(
                f"requested length {target_len} exceeds configured context "
                f"{maximum_context}; use --yarn-factor 4.0 for 128K")
        for case_index, case in enumerate(chosen_cases):
            if mode == "grid":
                depths = configured_depths
            elif mode == "cycle":
                depths = [
                    configured_depths[case_index % len(configured_depths)]]
            else:
                depths = [choose_depth(args, rng)]
            jobs.extend(
                (target_len, case, float(depth)) for depth in depths)
    return jobs


def depth_tag(depth):
    return f"{depth:.6f}".rstrip("0").rstrip(".")


def needle_block_index(tok, prompt, needle, block_size):
    """Which key-block the needle lands in (so downstream knows where the signal is)."""
    char_idx = prompt.index(needle)
    prefix_len = len(tok(prompt[:char_idx])["input_ids"])
    return prefix_len // block_size


def save_heatmap(pooled_stack, n_blk, seq_len, depth, path):
    """Block heatmap of teacher target mass, summed over all layers+heads -> [qb, kb].
    Log-scaled (mass spans orders of magnitude). Red dashed = needle key-block,
    white dotted = reader (question) row = last query block; their crossing is
    where the model looks at the needle when answering. Same convention as
    ks1_lite.save_heatmap so figures are comparable across scripts.
    """
    import matplotlib
    matplotlib.use("Agg")                       # headless: write PNG straight to disk
    import matplotlib.pyplot as plt
    import numpy as np

    agg = pooled_stack[0].float().sum(dim=(0, 1))  # [1,L,H,qb,kb] -> [qb,kb]
    grid = np.log1p(agg.detach().cpu().to(torch.float32).numpy())
    qb, kb = grid.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.axvline(n_blk + 0.5, color="red", lw=1.0, ls="--")   # needle key-block
    ax.axvline(n_blk - 0.5, color="red", lw=1.0, ls="--")
    ax.axhline(qb - 1, color="white", lw=1.0, ls=":")        # reader (question) row
    ax.set_title(f"teacher pooled mass  len={seq_len} depth={depth}\n"
                 f"needle key-block={n_blk}/{kb} (red), reader row (white)")
    ax.set_xlabel("key block")
    ax.set_ylabel("query block")
    fig.colorbar(im, ax=ax, label="log(1 + mass)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768],
                    help="context lengths to extract; defaults to 16k and 32k")
    ap.add_argument(
        "--yarn-factor", type=float, default=1.0,
        help="static YaRN factor; use 4.0 for Qwen's 131072-token configuration")
    ap.add_argument(
        "--original-max-position-embeddings", type=int,
        default=QWEN_NATIVE_CONTEXT,
        help="native context used to derive the YaRN limit; default: 32768")
    ap.add_argument(
        "--features-only", action="store_true",
        help="save only O(n) selector Q/K prototypes, not quadratic dense "
             "teacher mass; intended for 64K/128K traversal and replay")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--depth", type=float, default=None,
                    help="single fixed needle depth; overrides --depths and --depth-range")
    ap.add_argument("--depths", type=float, nargs="+", default=None,
                    help="candidate needle depths to randomly sample from; overrides --depth-range")
    ap.add_argument(
        "--depth-mode", choices=("sample", "cycle", "grid"),
        default="sample",
        help="sample one configured depth per job (default), cycle depths "
             "evenly over cases, or run the full case×depth grid")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.05, 0.95],
                    metavar=("MIN", "MAX"),
                    help="continuous random depth range used when --depth/--depths are omitted; default: 0.05 0.95")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for random case/depth selection")
    ap.add_argument("--store-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default=os.path.join(ROOT, "teacher_targets"))
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU run; default refuses to run without CUDA")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="skip writing the per-length pooled-mass heatmap PNG")
    ap.add_argument("--heldout", action="store_true",
                    help="use scripts/prompt_banks/heldout.json instead of train.json")
    ap.add_argument(
        "--prompt-bank",
        help=(
            "explicit prompt-bank JSON; overrides --heldout and supports "
            "natural_prose_v1 held-out cases"
        ),
    )
    ap.add_argument(
        "--verified-manifest",
        help=(
            "dense-screening manifest; extract only accepted exact prompts "
            "and bypass prompt regeneration"
        ),
    )
    ap.add_argument("--all", action="store_true",
                    help="run every scenario in the selected prompt bank; otherwise pick one randomly")
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="resume a large extraction by leaving existing target files unchanged")
    ap.add_argument(
        "--partition-by-length", action="store_true",
        help="write each requested-length bucket into its own subdirectory")
    ap.add_argument("--offload", action="store_true",
                    help="stream fp16 weights from CPU RAM layer-by-layer so a bigger model "
                         "(3B/32k, 7B) fits 8GB VRAM. Targets are bit-identical to a full-GPU "
                         "fp16 run -- only slower. Needs CPU RAM >= model size (3B=6GB, 7B=14GB).")
    ap.add_argument("--gpu-budget", type=float, default=4.0,
                    help="GiB of VRAM accelerate may fill with weights when --offload is set; "
                         "the rest stream from CPU. Lower = smaller peak, more streaming. "
                         "Leave headroom for activations (~1-2GB at 32k).")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU not available (torch is CPU-only or driver missing). "
            "Refusing to run on CPU. Install a CUDA torch build "
            "(pip install torch --index-url https://download.pytorch.org/whl/cu126 "
            "--force-reinstall --no-deps), or pass --allow-cpu to override.")

    rng = random.Random(args.seed)
    if args.verified_manifest and (args.prompt_bank or args.heldout):
        raise SystemExit(
            "--verified-manifest cannot be combined with --prompt-bank or "
            "--heldout")
    if args.verified_manifest:
        bank_path = args.verified_manifest
        cases = load_verified_manifest(bank_path)
        selected_bank = os.path.splitext(os.path.basename(bank_path))[0]
    else:
        bank_path = (
            args.prompt_bank
            if args.prompt_bank
            else (DEFAULT_HELDOUT_BANK if args.heldout else DEFAULT_TRAIN_BANK)
        )
        cases = load_prompt_bank(bank_path)
        selected_bank = (
            os.path.splitext(os.path.basename(bank_path))[0]
            if args.prompt_bank
            else ("heldout" if args.heldout else "train")
        )

    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    configure_tokenizer(
        tok, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    model_config = load_model_config(
        args.model, yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    maximum_context = context_limit(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)
    rope_config = yarn_metadata(
        yarn_factor=args.yarn_factor,
        original_max_position_embeddings=args.original_max_position_embeddings)

    if args.offload:
        # Stream weights: accelerate keeps ~gpu_budget GiB of layers resident and moves
        # the rest to CPU RAM, shuttling each to VRAM only for its forward. Peak VRAM =
        # resident weights + activation transient, not the whole model. Same fp16 numbers
        # as a full-GPU run. NOTE: no .to(DEVICE) -- device_map owns placement.
        if DEVICE != "cuda":
            raise SystemExit("--offload needs a CUDA GPU to stream weights onto.")
        offload_dir = os.path.join(args.out, "_offload")
        os.makedirs(offload_dir, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", attn_implementation="sdpa",
            config=model_config,
            device_map="auto",
            max_memory={0: f"{args.gpu_budget}GiB", "cpu": "48GiB"},
            offload_folder=offload_dir,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", attn_implementation="sdpa",
            config=model_config,
        ).to(DEVICE).eval()

    store_dtype = getattr(torch, args.store_dtype)
    try:
        jobs = (
            build_verified_jobs(cases, maximum_context)
            if args.verified_manifest
            else build_jobs(args, cases, rng, maximum_context)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"prompt_bank={selected_bank} cases={len(cases)} jobs={len(jobs)} "
          f"all={args.all} depth_mode={args.depth_mode} seed={args.seed}")

    for target_len, case, depth in jobs:
        prompt_builder = case.get("builder", "calibrated_units_v1")
        if prompt_builder == "verified_prompt_v1":
            full = case["prompt_text"]
            ndl = case["needle"]
            prompt = full
            filler_units = None
            natural_units = case.get("natural_units")
            if case["model"] != args.model:
                raise RuntimeError(
                    f"verified prompt {case['id']} was screened with "
                    f"{case['model']}, not {args.model}")
            screened_factor = float(case["rope"].get("factor", 1.0))
            if screened_factor != float(args.yarn_factor):
                raise RuntimeError(
                    f"verified prompt {case['id']} used YaRN factor "
                    f"{screened_factor}, not {args.yarn_factor}")
            if int(case["block_size"]) != int(args.block):
                raise RuntimeError(
                    f"verified prompt {case['id']} used block size "
                    f"{case['block_size']}, not {args.block}")
        elif prompt_builder == "natural_prose_v1":
            full, ndl, natural_units = build_natural_prompt_calibrated(
                tok, target_len, case, depth,
                seed=0 if args.seed is None else args.seed,
            )
            prompt = full
            filler_units = None
        else:
            prompt, ndl, filler_units = build_haystack_calibrated(
                tok, target_len, case["needle"], depth, case["filler"],
                suffix=case["question"])
            full = prompt + case["question"]
            natural_units = None
        calibrated_len = len(tok(full)["input_ids"])
        if (prompt_builder == "verified_prompt_v1"
                and calibrated_len != int(case["seq_len"])):
            raise RuntimeError(
                f"verified prompt {case['id']} reconstructed to "
                f"{calibrated_len} tokens, expected {case['seq_len']}")
        if target_len - calibrated_len > args.block:
            raise RuntimeError(
                f"prompt calibration missed target {target_len} by "
                f"{target_len - calibrated_len} tokens")
        n_blk = needle_block_index(tok, full, ndl, args.block)
        if (prompt_builder == "verified_prompt_v1"
                and n_blk != int(case["needle_block"])):
            raise RuntimeError(
                f"verified prompt {case['id']} evidence moved from block "
                f"{case['needle_block']} to {n_blk}")
        case_id = _safe_id(case["id"])
        dtag = depth_tag(depth)
        target_dir = (
            os.path.join(args.out, str(target_len))
            if args.partition_by_length else args.out)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(
            target_dir,
            f"teacher_{case_id}_len{calibrated_len}_blk{args.block}_d{dtag}.pt")
        if args.skip_existing and os.path.isfile(path):
            print(f"skip existing {path}", flush=True)
            del prompt, full
            continue

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if args.features_only:
            pooledQ_stack, pooledK_stack, seq_len = get_selector_features(
                model, tok, full, device=DEVICE, block_size=args.block)
            pooled_stack = None
        else:
            pooled_stack, pooledQ_stack, pooledK_stack, seq_len = get_pooled_targets(
                model, tok, full, device=DEVICE, block_size=args.block)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        if seq_len > maximum_context:
            raise RuntimeError(
                f"seq_len {seq_len} exceeded configured context {maximum_context}")

        if seq_len != calibrated_len:
            raise RuntimeError(
                f"token count changed between calibration ({calibrated_len}) "
                f"and extraction ({seq_len})")
        # Replace each large tensor with its storage dtype one at a time. Since
        # get_pooled_targets already cleared the layerwise capture dictionaries,
        # reassignment releases the FP32/model-dtype stack before converting the
        # next tensor instead of retaining every source and converted copy.
        if pooled_stack is not None:
            pooled_stack = pooled_stack.to(store_dtype)
        pooledQ_stack = pooledQ_stack.to(store_dtype)
        pooledK_stack = pooledK_stack.to(store_dtype)
        save_payload = {
            "pooled": pooled_stack,      # [1, L, H, qb, kb]      TARGET
            "pooledQ": pooledQ_stack,    # [1, L, G, qb, P, d]    selector input
            "pooledK": pooledK_stack,    # [1, L, G, kb, P, d]    selector input
            "head_dim": pooledQ_stack.shape[-1],
            "num_kv_heads": pooledK_stack.shape[2],
            "seq_len": seq_len,
            "requested_length": target_len,
            "haystack_token_budget": target_len,
            "prompt_builder": prompt_builder,
            "block_size": args.block,
            "model": args.model,
            "prompt_bank": selected_bank,
            "case_id": case["id"],
            "needle": case["needle"],
            "question": case["question"],
            "depth": depth,
            "needle_block": n_blk,
            "num_layers": pooledK_stack.shape[1],
            "num_heads": int(model.config.num_attention_heads),
            "proto": pooledK_stack.shape[-2],
            "store_dtype": args.store_dtype,
            "artifact_type": (
                "selector_features" if args.features_only else "teacher_target"),
            "features_only": bool(args.features_only),
            "rope": rope_config,
        }
        if filler_units is not None:
            save_payload["filler_units"] = filler_units
        if natural_units is not None:
            # Storing the exact diverse prompt avoids coupling replay to future
            # prose-generator revisions. The text is small compared with Q/K.
            save_payload.update({
                "prompt_text": full,
                "natural_units": natural_units,
                "reference_answers": list(case["answers"]),
            })
        if prompt_builder == "verified_prompt_v1":
            save_payload.update({
                "prompt_text": full,
                "prompt_format": case.get("prompt_format", "raw_text_v1"),
                "reference_answers": list(case["answers"]),
                "dense_screen": {
                    "accepted": True,
                    "dense_answer": case.get("dense_answer"),
                    "dense_exact": case.get("dense_exact"),
                    "dense_fuzzy": case.get("dense_fuzzy"),
                    "source_case_id": case.get("source_case_id"),
                    "candidate_id": case["id"],
                    "manifest": os.path.abspath(args.verified_manifest),
                },
            })
            if case.get("user_prompt_text") is not None:
                save_payload["user_prompt_text"] = case["user_prompt_text"]
        torch.save(save_payload, path)
        print(f"case={case_id}  len={seq_len:>6}  blocks={pooledK_stack.shape[3]:>4}  "
              f"depth={depth:.4f}  needle_block={n_blk:>4}  peak={peak_gb:.2f}GB  "
              f"saved={path}")

        if not args.no_heatmap and pooled_stack is not None:
            png = path[:-3] + ".png"
            save_heatmap(pooled_stack, n_blk, seq_len, depth, png)
            print(f"          heatmap={png}")

        # Do not let the completed target overlap the next forward. This is
        # especially important because assignment to the next get_pooled_targets
        # result happens only after that entire extraction has completed.
        del save_payload
        del pooledQ_stack, pooledK_stack
        if pooled_stack is not None:
            del pooled_stack
        del prompt, full
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
