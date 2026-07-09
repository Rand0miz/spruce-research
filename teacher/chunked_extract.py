import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# layer_idx -> pooled [1, num_heads, q_blocks, k_blocks] (cpu, fp32)
_CAPTURE = {}
# layer_idx -> (pooledQ [1, H, q_blocks, d], pooledK [1, kv, k_blocks, d]) (cpu, fp32)
# These are the SELECTOR INPUTS (SeerAttention scores Wq*pooledQ . Wk*pooledK), captured
# here so training reads tensors only, no frozen Qwen in the loop. pooledK keeps the
# original kv_head count (pre-GQA-repeat) to match the selected_blocks kv_head_group axis.
_CAPTURE_QK = {}
_BLOCK = 64
# How many heads to score at once inside chunked_pool. The last query block scores
# against all keys -> a [b, H_CHUNK, block, k_len] fp32 strip (+ its softmax +
# padded copy). At 32K that strip is ~1GB across all heads; chunking heads caps
# the transient so it fits 8GB VRAM without spilling to shared RAM. Output is
# bit-identical to processing all heads at once (heads are independent here).
_HEAD_CHUNK = 2
_REGISTERED = False


def _mean_pool_seq(x, block):
    """[b, H, L, d] -> [b, H, nb, d]: mean of each block along the sequence dim.
    Last block may be partial; average over its real rows only (not the zero pad)."""
    b, H, L, d = x.shape
    nb = (L + block - 1) // block
    pad = nb * block - L
    if pad:
        x = F.pad(x, (0, 0, 0, pad))                # zero-pad the ragged tail
    x = x.view(b, H, nb, block, d)
    if pad:
        counts = x.new_full((nb, 1), float(block))
        counts[-1, 0] = block - pad                 # real rows in the last block
        return x.sum(dim=3) / counts.view(1, 1, nb, 1)
    return x.mean(dim=3)


def _repeat_kv(key, n_rep):
    """[b, n_kv, k, d] -> [b, n_kv*n_rep, k, d] (match eager pool_attention head count)."""
    if n_rep == 1:
        return key
    b, n_kv, k, d = key.shape
    return key[:, :, None, :, :].expand(b, n_kv, n_rep, k, d).reshape(b, n_kv * n_rep, k, d)


@torch.no_grad()
def chunked_pool(query, key, scaling, block_size=_BLOCK, head_chunk=None):
    """
    query: [b, H, q_len, d] post-RoPE
    key:   [b, H, k_len, d] post-RoPE, already repeated to H heads
    Returns pooled [b, H, q_blocks, k_blocks] == pool_attention(causal_softmax(q,k)),
    but never materializes q_len x k_len. Peak = one [b,head_chunk,block,end] score
    strip; head_chunk (default _HEAD_CHUNK) bounds the transient so 32K fits 8GB.
    """
    if head_chunk is None:
        head_chunk = _HEAD_CHUNK
    b, H, q_len, d = query.shape
    k_len = key.shape[2]
    qb = (q_len + block_size - 1) // block_size
    kb = (k_len + block_size - 1) // block_size
    pooled = query.new_zeros((b, H, qb, kb), dtype=torch.float32)

    for h0 in range(0, H, head_chunk):
        h1 = min(h0 + head_chunk, H)               # heads are independent -> per-slice math
        for I in range(qb):
            r0 = I * block_size
            r1 = min(r0 + block_size, q_len)
            rows = r1 - r0
            end = r1                               # causal: keys 0..r1-1 reachable
            kb_up = (end + block_size - 1) // block_size

            q_blk = query[:, h0:h1, r0:r1, :]      # [b,hc,rows,d]
            k_up = key[:, h0:h1, :end, :]          # [b,hc,end,d]
            s = torch.matmul(q_blk, k_up.transpose(-1, -2)) * scaling  # [b,hc,rows,end]

            # intra-strip causal mask (only the diagonal block actually needs it)
            g = torch.arange(r0, r1, device=query.device)[:, None]     # [rows,1] global q index
            j = torch.arange(end, device=query.device)[None, :]        # [1,end]  key index
            s = s.masked_fill((j > g)[None, None], float("-inf"))

            p = torch.softmax(s.float(), dim=-1)   # [b,hc,rows,end]  rows sum to 1
            pad = kb_up * block_size - end
            p = F.pad(p, (0, pad))                  # [b,hc,rows,kb_up*block]
            p = p.view(b, h1 - h0, rows, kb_up, block_size).sum(dim=-1)  # [b,hc,rows,kb_up]
            pooled[:, h0:h1, I, :kb_up] = p.sum(dim=2)  # sum over q-rows in block
    return pooled


def _capture_attention(module, query, key, value, attention_mask,
                       scaling=None, dropout=0.0, sliding_window=None, **kwargs):
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    n_rep = query.shape[1] // key.shape[1]
    key_rep = _repeat_kv(key, n_rep)
    pooled = chunked_pool(query, key_rep, scaling, _BLOCK)
    _CAPTURE[module.layer_idx] = pooled.cpu()
    # Selector inputs: mean-pool post-RoPE Q per query-block and original K per key-block.
    # K uses pre-repeat `key` -> [1, kv, kb, d] (one vector per kv_head_group).
    pooledQ = _mean_pool_seq(query, _BLOCK).cpu()   # [1, H, qb, d]
    pooledK = _mean_pool_seq(key, _BLOCK).cpu()     # [1, kv, kb, d]
    _CAPTURE_QK[module.layer_idx] = (pooledQ, pooledK)
    # Propagate through SDPA's FUSED CAUSAL path. We repeat KV to the full head
    # count ourselves instead of letting sdpa_attention_forward set enable_gqa=True:
    # on this build the only available fused kernel (mem-efficient) rejects GQA
    # broadcast (unequal q/kv heads) -> "No available kernel". Equal heads + no dense
    # mask keeps it on the fused path, so no [1,1,S,S] bias is materialized, and
    # is_causal=True is correct since chunked_pool applied its own causal mask.
    value_rep = _repeat_kv(value, n_rep)
    attn_output = F.scaled_dot_product_attention(
        query, key_rep, value_rep,
        attn_mask=None, dropout_p=dropout, scale=scaling, is_causal=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        ALL_ATTENTION_FUNCTIONS.register("chunked_capture", _capture_attention)
        _REGISTERED = True


@torch.no_grad()
def get_pooled_targets(model, tok, prompt, device="cuda", block_size=_BLOCK):
    """
    One forward pass; captures chunked pooled block-mass per layer.
    Returns (pooled_stack [1, num_layers, num_heads, q_blocks, k_blocks] fp32 cpu,
             pooledQ_stack [1, num_layers, H, q_blocks, d] fp32 cpu,
             pooledK_stack [1, num_layers, kv, k_blocks, d] fp32 cpu,
             seq_len).
    pooled_stack is the training TARGET; the Q/K stacks are the selector INPUTS.
    Never materializes a full attention matrix -> fits 16K-32K on 8GB.
    """
    global _BLOCK
    _BLOCK = block_size
    _ensure_registered()
    _CAPTURE.clear()
    _CAPTURE_QK.clear()

    prev = model.config._attn_implementation
    model.config._attn_implementation = "chunked_capture"
    for m in model.modules():                      # defensive: submodules cache the string
        cfg = getattr(m, "config", None)
        if cfg is not None:
            cfg._attn_implementation = "chunked_capture"
    try:
        inputs = tok(prompt, return_tensors="pt").to(device)
        # Force fused SDPA (flash/mem-efficient) for the forward-propagation path in
        # _capture_attention. Excluding MATH stops the 13GB full [1,H,L,L] fp32 score
        # matrix (SDPA's Windows fallback when flash isn't built) -> fits 8GB.
        # If no fused kernel is available it errors loudly instead of silently OOMing.
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends):
            # Run the base decoder (model.model), NOT the full CausalLM. The attention
            # hooks fire inside the layers, so we get every pooled target -- but we skip
            # lm_head, which would project the final hidden state to logits
            # [1, seq_len, vocab] ~= 10GB fp16 @ 32K/152K-vocab. We never use logits
            # here (no next-token prediction), so dropping the head keeps peak in VRAM
            # instead of spilling to shared RAM.
            model.model(**inputs, use_cache=False)
    finally:
        model.config._attn_implementation = prev
        for m in model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None:
                cfg._attn_implementation = prev

    order = sorted(_CAPTURE)
    pooled_stack = torch.stack([_CAPTURE[i] for i in order], dim=1)        # [1, L, H, qb, kb]
    pooledQ_stack = torch.stack([_CAPTURE_QK[i][0] for i in order], dim=1)  # [1, L, H, qb, d]
    pooledK_stack = torch.stack([_CAPTURE_QK[i][1] for i in order], dim=1)  # [1, L, kv, kb, d]
    return pooled_stack, pooledQ_stack, pooledK_stack, inputs["input_ids"].shape[1]
