"""Compare dense Qwen generation with SPRUCE Triton sparse-prefill + dense decode."""
import argparse
import gc
import json
import os
import sys
import tempfile
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eval.score import score_retrival
from interfaces.validator import validate_selected_blocks
from kernels.sparse_prefill import (
    SPRUCE_TRITON_SPARSE_PREFILL,
    register_triton_sparse_prefill_attention,
)
from teacher.prompt_replay import reconstruct_teacher_prompt


def _model_device(model):
    return next(model.parameters()).device


def _load_model(model_name, implementation, dtype, offload_dir):
    from transformers import AutoModelForCausalLM
    os.makedirs(offload_dir, exist_ok=True)
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": "cuda:0"},
        low_cpu_mem_usage=True, offload_state_dict=True, offload_folder=offload_dir,
        attn_implementation=implementation,
    ).eval()


def _set_attention_backend(model, implementation):
    """Switch all config holders before cached decode begins."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = implementation


def _generate(model, inputs, max_new_tokens, *, prefill_kwargs=None, decode_backend="sdpa"):
    """Greedy sparse-prefill / dense-decode generation without HF kwargs filtering."""
    device = _model_device(model)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prefill_kwargs = prefill_kwargs or {}
    base_model = model.base_model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = base_model(**inputs, use_cache=True, **prefill_kwargs)
        next_logits = model.get_output_embeddings()(output.last_hidden_state[:, -1:, :])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        generated = []
        for step in range(max_new_tokens):
            next_token = next_logits[:, -1].argmax(dim=-1)
            generated.append(next_token)
            if step + 1 == max_new_tokens:
                break

            # The initial pass was sparse only. Every following one-token
            # cached call is ordinary dense SDPA decode.
            _set_attention_backend(model, decode_backend)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token[:, None])], dim=-1)
            position = attention_mask.long().cumsum(-1)[:, -1:] - 1
            cache_position = torch.tensor(
                [input_ids.shape[1] + step], device=device, dtype=torch.long)
            output = base_model(
                input_ids=next_token[:, None], attention_mask=attention_mask,
                position_ids=position, cache_position=cache_position,
                past_key_values=output.past_key_values, use_cache=True,
            )
            next_logits = model.get_output_embeddings()(output.last_hidden_state[:, -1:, :])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    return torch.stack(generated, dim=1)[0].cpu(), {
        "seconds": total_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": total_seconds - prefill_seconds,
    }


def _free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--selected-blocks", required=True)
    parser.add_argument("--teacher-target", help="defaults to source recorded in selected-block artifact")
    parser.add_argument("--prompt-bank")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument(
        "--load-offload-dir", default=os.path.join(tempfile.gettempdir(), "spruce_hf_load_offload"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    artifact = torch.load(args.selected_blocks, map_location="cpu", weights_only=False)
    selected, meta = artifact["selected_blocks"], artifact["meta"]
    validate_selected_blocks(selected)
    teacher_target = args.teacher_target or meta.get("source")
    if not teacher_target:
        raise SystemExit("--teacher-target is required when artifact metadata has no source path")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt, teacher = reconstruct_teacher_prompt(tokenizer, teacher_target, bank_path=args.prompt_bank)
    encoded = tokenizer(prompt, return_tensors="pt")
    if encoded.input_ids.shape[1] != meta["seq_len"]:
        raise SystemExit("reconstructed prompt token count does not match selected-block artifact")
    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    register_triton_sparse_prefill_attention()
    sparse_model = _load_model(args.model, SPRUCE_TRITON_SPARSE_PREFILL, dtype, args.load_offload_dir)
    sparse_ids, sparse_timing = _generate(
        sparse_model, encoded, args.max_new_tokens,
        prefill_kwargs={
            "selected_blocks": selected.to(_model_device(sparse_model)),
            "block_size": int(meta["block_size"]),
            "validate_selected_blocks_input": False,
        },
    )
    sparse_answer = tokenizer.decode(sparse_ids, skip_special_tokens=True)
    _free_model(sparse_model)

    dense_model = _load_model(args.model, "sdpa", dtype, args.load_offload_dir)
    dense_ids, dense_timing = _generate(dense_model, encoded, args.max_new_tokens)
    dense_answer = tokenizer.decode(dense_ids, skip_special_tokens=True)
    _free_model(dense_model)

    result = {
        "model": args.model, "case_id": teacher["case_id"], "seq_len": int(meta["seq_len"]),
        "block_size": int(meta["block_size"]), "needle_block": int(teacher["needle_block"]),
        "sparse": {**score_retrival(sparse_answer, teacher["needle"]), **sparse_timing},
        "dense": {**score_retrival(dense_answer, teacher["needle"]), **dense_timing},
        "answers_match": sparse_answer.strip() == dense_answer.strip(),
        "selected_blocks": args.selected_blocks, "teacher_target": teacher_target,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
