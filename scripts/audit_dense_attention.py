"""Where does dense Qwen actually attend the evidence?

Token-level audit for one teacher target: registers a recording SDPA wrapper,
runs dense prefill + greedy decode, and reports per layer/head attention mass
on the evidence block for (a) the prompt's question-token rows and (b) every
decode step. Block-pooled teacher mass hid this signal (LOG 2026-07-26); this
script measures it unpooled.

The recorder delegates the actual attention output to SDPA, so the forward is
numerically dense — the decoded answer must match the dense-verified answer or
the capture is broken.
"""
import argparse
import json
import math
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

AUDIT_ATTENTION = "spruce_audit_attention"

# Filled by main() before the forward runs; the registered wrapper reads it.
_AUDIT = {}


def block_mass(probs, block_size, kb):
    """[H, T] token attention rows -> [H, kb] block mass sums."""
    H, T = probs.shape
    pad = kb * block_size - T
    if pad:
        probs = F.pad(probs, (0, pad))
    return probs.view(H, kb, block_size).sum(dim=-1)


def block_rank(block_scores, block):
    """[H, kb] -> [H] 0-based rank of `block` per head (0 = highest mass)."""
    return (block_scores > block_scores[:, [block]]).sum(dim=-1)


def find_question_span(prompt_ids, question_ids):
    """Last occurrence of ``question_ids`` as a subsequence of ``prompt_ids``.

    Returns (start, end) token indices (end exclusive) or None.
    """
    n, m = len(prompt_ids), len(question_ids)
    if m == 0 or m > n:
        return None
    for start in range(n - m, -1, -1):
        if prompt_ids[start:start + m] == question_ids:
            return (start, start + m)
    return None


def _layer_index(module):
    for name in ("layer_idx", "layer_id"):
        value = getattr(module, name, None)
        if value is not None:
            return int(value)
    raise ValueError("audit attention needs module.layer_idx")


def _record(layer, probs, store):
    """probs [H, T] fp32 on GPU -> pooled evidence stats appended to store."""
    pooled = block_mass(probs, _AUDIT["block_size"], _AUDIT["kb"])
    needle = _AUDIT["needle_block"]
    store.setdefault(layer, []).append({
        "mass": pooled[:, needle].cpu(),
        "rank": block_rank(pooled, needle).cpu(),
        "top": pooled.argmax(dim=-1).cpu(),
    })
    del pooled


def audit_attention_forward(module, query, key, value, attention_mask,
                            **kwargs):
    """Recording dense-attention backend.

    query [B,Hq,Tq,D], key/value [B,Hkv,Tk,D] (post-RoPE, as every
    Transformers attention backend receives them). Records block-pooled
    evidence attention for audited rows, then delegates output to SDPA.
    """
    layer = _layer_index(module)
    B, Hq, Tq, D = query.shape
    Hkv, Tk = key.shape[1], key.shape[2]
    group = Hq // Hkv
    scale = kwargs.get("scaling") or 1.0 / math.sqrt(D)
    k_rep = key.repeat_interleave(group, dim=1) if group > 1 else key
    v_rep = value.repeat_interleave(group, dim=1) if group > 1 else value

    if _AUDIT:
        if Tq == Tk and Tq > 1:
            rows = _AUDIT["rows"].to(query.device)              # absolute positions
            q_rows = query[0].index_select(1, rows)             # [H, R, D]
            scores = torch.einsum("hrd,htd->hrt", q_rows.float(),
                                  k_rep[0].float()) * scale
            causal = (torch.arange(Tk, device=query.device)[None, :]
                      <= rows[:, None])                          # [R, Tk]
            scores = scores.masked_fill(~causal[None], float("-inf"))
            probs = scores.softmax(dim=-1)                       # [H, R, Tk]
            store = _AUDIT["prefill"]
            for r in range(rows.shape[0]):
                _record(layer, probs[:, r], store)
            del scores, probs
        elif Tq == 1:
            scores = torch.einsum("hrd,htd->hrt", query[0].float(),
                                  k_rep[0].float()) * scale      # [H, 1, Tk]
            probs = scores.softmax(dim=-1)[:, 0]                 # [H, Tk]
            _record(layer, probs, _AUDIT["decode"])
            del scores, probs

    # Ignore any materialized additive mask: this audit runs batch=1 unpadded
    # prompts, where causal SDPA is equivalent and a [1,H,T,T] mask tensor
    # forces the memory-hungry math path (12GB at 16K on fp32 upcast).
    del attention_mask
    output = F.scaled_dot_product_attention(
        query, k_rep, v_rep, is_causal=(Tq == Tk and Tq > 1), scale=scale)
    return output.transpose(1, 2).contiguous(), None


def register_audit_attention(name=AUDIT_ATTENTION):
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(name, audit_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name


def _stack_records(store, layers):
    """{layer: [records]} -> mass/rank/top nested lists [L][H][N]."""
    mass, rank, top = [], [], []
    for layer in range(layers):
        records = store.get(layer, [])
        if records:
            mass.append(torch.stack([r["mass"] for r in records], dim=1).tolist())
            rank.append(torch.stack([r["rank"] for r in records], dim=1).tolist())
            top.append(torch.stack([r["top"] for r in records], dim=1).tolist())
        else:
            mass.append([])
            rank.append([])
            top.append([])
    return mass, rank, top


def _summary_lines(label, store, layers, topk):
    lines = [f"{label}: layer  max_mass  best(head,row/step)  n_rank<{topk}"]
    for layer in range(layers):
        records = store.get(layer, [])
        if not records:
            continue
        mass = torch.stack([r["mass"] for r in records], dim=1)   # [H, N]
        rank = torch.stack([r["rank"] for r in records], dim=1)
        best = int(mass.argmax())
        h, n = divmod(best, mass.shape[1])
        hits = int((rank < topk).sum())
        lines.append(
            f"{label}: L{layer:02d}  {float(mass.max()):.4f}  (h{h},{n})  {hits}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="teacher target .pt")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-question-rows", type=int, default=96)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument(
        "--load-offload-dir",
        default=os.path.join(tempfile.gettempdir(), "spruce_hf_load_offload"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from benchmarks.compare_dense_sparse import _generate, _load_model
    from eval.score import score_retrival
    from teacher.prompt_replay import reconstruct_teacher_prompt

    raw = torch.load(args.target, map_location="cpu", weights_only=False)
    needle_block = int(raw["needle_block"])
    block_size = int(raw["block_size"])
    seq_len = int(raw["seq_len"])
    question = raw.get("question")
    kb = (seq_len + block_size - 1) // block_size
    if not 0 <= needle_block < kb:
        raise SystemExit(f"target has no valid needle_block ({needle_block})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt, meta = reconstruct_teacher_prompt(tokenizer, args.target)
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded.input_ids[0].tolist()
    if len(prompt_ids) != seq_len:
        raise SystemExit(
            f"reconstructed prompt tokenized to {len(prompt_ids)}, "
            f"target says {seq_len}")

    span = None
    if question:
        question_ids = tokenizer(question, add_special_tokens=False).input_ids
        span = find_question_span(prompt_ids, question_ids)
    if span is not None:
        rows = list(range(span[0], span[1]))
    else:
        rows = list(range(max(0, seq_len - block_size), seq_len))
    if (seq_len - 1) not in rows:
        rows.append(seq_len - 1)
    rows = rows[-args.max_question_rows:]

    _AUDIT.clear()
    _AUDIT.update({
        "rows": torch.tensor(rows, dtype=torch.long),
        "block_size": block_size, "kb": kb, "needle_block": needle_block,
        "prefill": {}, "decode": {},
    })

    register_audit_attention()
    model = _load_model(args.model, AUDIT_ATTENTION, torch.float16,
                        args.load_offload_dir)
    token_ids, timing = _generate(
        model, {name: value for name, value in encoded.items()},
        args.max_new_tokens, decode_backend=AUDIT_ATTENTION)
    answer = tokenizer.decode(token_ids, skip_special_tokens=True)
    score = score_retrival(
        answer, raw.get("needle"), reference_answers=raw.get("reference_answers"))
    layers = int(model.config.num_hidden_layers)
    del model

    print(f"case: {raw.get('case_id')}")
    print(f"needle_block: {needle_block}  audited prefill rows: {len(rows)} "
          f"(question span {'found' if span else 'NOT FOUND - fallback'})")
    print(f"answer: {answer!r}")
    print(f"exact: {score.get('exact')}  fuzzy: {score.get('fuzzy')}")
    for line in _summary_lines("prefill", _AUDIT["prefill"], layers, args.topk):
        print(line)
    for line in _summary_lines("decode", _AUDIT["decode"], layers, args.topk):
        print(line)

    if args.out:
        p_mass, p_rank, p_top = _stack_records(_AUDIT["prefill"], layers)
        d_mass, d_rank, d_top = _stack_records(_AUDIT["decode"], layers)
        report = {
            "target": os.path.abspath(args.target),
            "case_id": raw.get("case_id"),
            "model": args.model,
            "needle_block": needle_block,
            "block_size": block_size,
            "seq_len": seq_len,
            "question_span_found": span is not None,
            "audited_rows": rows,
            "answer": answer,
            "exact": score.get("exact"),
            "fuzzy": score.get("fuzzy"),
            "timing": timing,
            "prefill": {"evidence_mass": p_mass, "evidence_rank": p_rank,
                        "top_block": p_top},
            "decode": {"evidence_mass": d_mass, "evidence_rank": d_rank,
                       "top_block": d_top},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
