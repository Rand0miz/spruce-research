"""Replay one exact prompt through dense or SPRUCE sparse-prefill Qwen.

This is the Stage 3.2 integration harness.  It consumes an offline
``selected_blocks`` artifact created by ``export_selected_blocks.py``.  The
artifact is valid only for the exact prompt/tokenization that produced its
teacher target, so this script refuses a token-length mismatch.

The sparse path is a Python correctness reference, not a performance test.
Use ``--compare-dense`` only at a length where dense eager attention fits.
"""
import argparse
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from interfaces.validator import validate_selected_blocks
from sparse.attention import SPARSE_PREFILL_ATTENTION, register_sparse_prefill_attention
from teacher.prompt_replay import reconstruct_teacher_prompt


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def run_once(model_name, implementation, inputs, selected, block_size, torch_dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=implementation,
    ).eval()
    device = model_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    kwargs = {"use_cache": False}
    if implementation == SPARSE_PREFILL_ATTENTION:
        # The CPU artifact was validated before model loading.  Avoid repeating
        # the Python-row validator inside every decoder layer.
        kwargs.update(
            selected_blocks=selected.to(device),
            block_size=block_size,
            validate_selected_blocks_input=False,
        )
    # Do not call the CausalLM wrapper: its [batch, seq_len, vocab] logits are
    # enormous at 16K-32K (and converting them to fp32 caused the reported OOM).
    # The final hidden state is enough to obtain the next-token distribution,
    # which is the relevant replay result for the prompt's answer.
    with torch.inference_mode():
        result = model.get_base_model()(**inputs, **kwargs)
        final_hidden = result.last_hidden_state[:, -1:, :]
        logits = model.get_output_embeddings()(final_hidden).detach().float().cpu()
    del result, final_hidden, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--selected-blocks", required=True, help=".pt produced by export_selected_blocks.py")
    parser.add_argument(
        "--teacher-target",
        help="source teacher target .pt; defaults to the artifact's saved source path",
    )
    parser.add_argument(
        "--prompt-bank",
        help="optional prompt-bank JSON override; normally inferred from teacher-target metadata",
    )
    parser.add_argument("--compare-dense", action="store_true", help="also run dense eager and report logit differences")
    parser.add_argument("--reconstruct-only", action="store_true", help="verify prompt reconstruction, then exit before loading Qwen")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--save-logits",
        help="optional output .pt path for final-position (next-token) sparse logits",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    artifact = torch.load(args.selected_blocks, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "selected_blocks" not in artifact or "meta" not in artifact:
        raise SystemExit("--selected-blocks must be an export_selected_blocks.py artifact")
    selected = artifact["selected_blocks"]
    meta = artifact["meta"]
    validate_selected_blocks(selected)
    block_size = int(meta["block_size"])

    teacher_target = args.teacher_target or meta.get("source")
    if not teacher_target:
        raise SystemExit("pass --teacher-target: the selected-block artifact has no source target path")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        prompt, teacher_meta = reconstruct_teacher_prompt(
            tokenizer, teacher_target, bank_path=args.prompt_bank)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"could not reconstruct source prompt: {error}") from error
    encoded = tokenizer(prompt, return_tensors="pt")
    seq_len = int(encoded.input_ids.shape[1])
    if seq_len != int(meta["seq_len"]):
        raise SystemExit(
            f"token-length mismatch: reconstructed prompt tokenized to {seq_len}, but selected-block "
            f"artifact was generated for {meta['seq_len']}. Use its matching teacher target/tokenizer.")
    expected_blocks = (seq_len + block_size - 1) // block_size
    if selected.shape[3] != expected_blocks:
        raise SystemExit(
            f"selected_blocks has {selected.shape[3]} query blocks; expected {expected_blocks}.")

    if args.reconstruct_only:
        print(
            f"prompt reconstruction passed: case={teacher_meta['case_id']} seq_len={seq_len} "
            f"needle_block={teacher_meta['needle_block']}"
        )
        return

    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    register_sparse_prefill_attention()
    sparse_logits = run_once(args.model, SPARSE_PREFILL_ATTENTION, encoded, selected, block_size, dtype)
    print(
        f"sparse replay passed: case={teacher_meta['case_id']} next-token logits shape={tuple(sparse_logits.shape)} "
        f"block_size={block_size}"
    )

    if args.save_logits:
        torch.save({"logits": sparse_logits, "selected_blocks": args.selected_blocks, "meta": meta}, args.save_logits)
        print(f"saved sparse logits to {args.save_logits}")

    if args.compare_dense:
        dense_logits = run_once(args.model, "eager", encoded, selected, block_size, dtype)
        diff = (sparse_logits - dense_logits).abs()
        print(f"dense comparison: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}")


if __name__ == "__main__":
    main()
