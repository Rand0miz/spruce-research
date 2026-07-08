import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

# layer_idx -> pooled [1, num_heads, q_blocks, k_blocks] (cpu, fp32)
_CAPTURE = {}
_BLOCK = 64
_REGISTERED = False


def _repeat_kv(key, n_rep):
    """[b, n_kv, k, d] -> [b, n_kv*n_rep, k, d] (match eager pool_attention head count)."""
    if n_rep == 1:
        return key
    b, n_kv, k, d = key.shape
    return key[:, :, None, :, :].expand(b, n_kv, n_rep, k, d).reshape(b, n_kv * n_rep, k, d)


@torch.no_grad()
def chunked_pool(query, key, scaling, block_size=_BLOCK):
    """
    query: [b, H, q_len, d] post-RoPE
    key:   [b, H, k_len, d] post-RoPE, already repeated to H heads
    Returns pooled [b, H, q_blocks, k_blocks] == pool_attention(causal_softmax(q,k)),
    but never materializes q_len x k_len. Peak = one [b,H,block,end] score strip.
    """
    b, H, q_len, d = query.shape
    k_len = key.shape[2]
    qb = (q_len + block_size - 1) // block_size
    kb = (k_len + block_size - 1) // block_size
    pooled = query.new_zeros((b, H, qb, kb), dtype=torch.float32)

    for I in range(qb):
        r0 = I * block_size
        r1 = min(r0 + block_size, q_len)
        rows = r1 - r0
        end = r1                                   # causal: keys 0..r1-1 reachable
        kb_up = (end + block_size - 1) // block_size

        q_blk = query[:, :, r0:r1, :]              # [b,H,rows,d]
        k_up = key[:, :, :end, :]                  # [b,H,end,d]
        s = torch.matmul(q_blk, k_up.transpose(-1, -2)) * scaling  # [b,H,rows,end]

        # intra-strip causal mask (only the diagonal block actually needs it)
        g = torch.arange(r0, r1, device=query.device)[:, None]     # [rows,1] global q index
        j = torch.arange(end, device=query.device)[None, :]        # [1,end]  key index
        s = s.masked_fill((j > g)[None, None], float("-inf"))

        p = torch.softmax(s.float(), dim=-1)       # [b,H,rows,end]  rows sum to 1
        pad = kb_up * block_size - end
        p = F.pad(p, (0, pad))                      # [b,H,rows,kb_up*block]
        p = p.view(b, H, rows, kb_up, block_size).sum(dim=-1)      # [b,H,rows,kb_up]
        pooled[:, :, I, :kb_up] = p.sum(dim=2)     # sum over q-rows in block
    return pooled


def _capture_attention(module, query, key, value, attention_mask,
                       scaling=None, dropout=0.0, sliding_window=None, **kwargs):
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    n_rep = query.shape[1] // key.shape[1]
    pooled = chunked_pool(query, _repeat_kv(key, n_rep), scaling, _BLOCK)
    _CAPTURE[module.layer_idx] = pooled.cpu()
    # delegate the REAL attention so hidden states propagate (memory-safe)
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling,
                                  sliding_window=sliding_window, **kwargs)


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        ALL_ATTENTION_FUNCTIONS.register("chunked_capture", _capture_attention)
        _REGISTERED = True


@torch.no_grad()
def get_pooled_targets(model, tok, prompt, device="cuda", block_size=_BLOCK):
    """
    One forward pass; captures chunked pooled block-mass per layer.
    Returns (pooled_stack [1, num_layers, num_heads, q_blocks, k_blocks] fp32 cpu, seq_len).
    Never materializes a full attention matrix -> fits 16K-32K on 8GB.
    """
    global _BLOCK
    _BLOCK = block_size
    _ensure_registered()
    _CAPTURE.clear()

    prev = model.config._attn_implementation
    model.config._attn_implementation = "chunked_capture"
    for m in model.modules():                      # defensive: submodules cache the string
        cfg = getattr(m, "config", None)
        if cfg is not None:
            cfg._attn_implementation = "chunked_capture"
    try:
        inputs = tok(prompt, return_tensors="pt").to(device)
        model(**inputs, use_cache=False)
    finally:
        model.config._attn_implementation = prev
        for m in model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None:
                cfg._attn_implementation = prev

    layers = [_CAPTURE[i] for i in sorted(_CAPTURE)]   # each [1,H,qb,kb]
    pooled_stack = torch.stack(layers, dim=1)          # [1, L, H, qb, kb]
    return pooled_stack, inputs["input_ids"].shape[1]
