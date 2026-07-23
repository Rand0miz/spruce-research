import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# layer_idx -> pooled [1, num_heads, q_blocks, k_blocks] (cpu, fp32)
_CAPTURE = {}
# layer_idx -> (pooledQ [1, G, q_blocks, P, d], pooledK [1, G, k_blocks, P, d]) (cpu, MODEL
# DTYPE, e.g. bf16 -- NOT fp32; _capture_attention .cpu()s these without a .float() cast,
# so they carry whatever dtype the model ran in).
# These are the SELECTOR INPUTS (SeerAttention scores Wq*pooledQ . Wk*pooledK), captured
# here so training reads tensors only, no frozen Qwen in the loop. pooledK keeps the
# original kv_head count (pre-GQA-repeat) to match the selected_blocks kv_head_group axis.
_CAPTURE_QK = {}
_BLOCK = 64
_PROTO = 8              # prototypes per block (1 mean + 7 outliers), both Q and K sides
# How many heads to score at once inside chunked_pool. The last query block scores
# against all keys -> a [b, H_CHUNK, block, k_len] fp32 strip (+ its softmax +
# padded copy). At 32K that strip is ~1GB across all heads; chunking heads caps
# the transient so it fits 8GB VRAM without spilling to shared RAM. Output is
# bit-identical to processing all heads at once (heads are independent here).
_HEAD_CHUNK = 2
_REGISTERED = False


def _proto_pool_seq(x, block, P):
    """[b,H,L,d] -> [b,H,nb,P,d]: P prototype vectors per block.
    proto 0 = block mean (real rows only); protos 1..P-1 = the P-1 tokens
    farthest (squared-L2) from that mean. Short/ragged blocks (fewer than P-1
    real outlier rows) fill spare slots with the mean, so max-pooling over
    prototypes is unaffected. Selection uses only x -> reproducible at inference."""
    b, H, L, d = x.shape
    nb = (L + block - 1) // block
    pad = nb * block - L
    if pad:
        x = F.pad(x, (0, 0, 0, pad))                 # zero-pad ragged tail
    xb = x.view(b, H, nb, block, d)                  # [b,H,nb,block,d]

    counts = xb.new_full((nb,), float(block))
    if pad:
        counts[-1] = block - pad                     # real rows in last block
    valid = (torch.arange(block, device=x.device)[None, None, None, :]
             < counts.view(1, 1, nb, 1))             # [1,1,nb,block] bool
    vf = valid.to(xb.dtype)[..., None]               # [1,1,nb,block,1]
    mean = (xb * vf).sum(dim=3) / counts.view(1, 1, nb, 1)   # [b,H,nb,d]

    # Distance math always in fp32, regardless of x's dtype: a squared-L2 distance
    # overflows fp16 (max ~65504) for exactly the largest outliers -- the needles
    # this pooling exists to preserve. fp32 is ~33MB transient at 32K -- free. The
    # returned prototypes below still use x's original dtype; only this fp32 cast
    # is local to the distance computation.
    dist = (xb.float() - mean.float()[:, :, :, None, :]).pow(2).sum(dim=-1)  # [b,H,nb,block] fp32
    dist = dist.masked_fill(~valid, float("-inf"))   # never pick a pad row
    k = min(P - 1, block)
    idx = dist.topk(k, dim=-1).indices               # [b,H,nb,k]
    gidx = idx[..., None].expand(-1, -1, -1, -1, d)
    outliers = xb.gather(3, gidx)                     # [b,H,nb,k,d]
    # if a block had fewer than k real rows, some picked rows are pad: replace those
    # with the mean. Derived from the validity mask itself, NOT from dist==-inf --
    # a genuine (finite) distance can equal +inf under fp16 overflow for a real
    # extreme outlier, which torch.isinf would also flag as "pad" and silently
    # replace with the mean -- exactly inverting the point of this pooling.
    sel_valid = valid.expand(b, H, nb, block).gather(-1, idx)  # [b,H,nb,k] bool
    bad = (~sel_valid)[..., None]                      # [b,H,nb,k,1]
    outliers = torch.where(bad, mean[:, :, :, None, :].expand_as(outliers), outliers)

    protos = torch.cat([mean[:, :, :, None, :], outliers], dim=3)   # [b,H,nb,1+k,d]
    if 1 + k < P:                                     # only if block < P-1 (not at 64/8)
        extra = mean[:, :, :, None, :].expand(-1, -1, -1, P - (1 + k), -1)
        protos = torch.cat([protos, extra], dim=3)
    return protos                                     # [b,H,nb,P,d]


def _group_avg_tokens(x, G):
    """[b,H,L,d] -> [b,G,L,d]: average the H/G query heads inside each kv-group,
    per token. Done pre-pool so each prototype is a real per-group token vector."""
    b, H, L, d = x.shape
    assert H % G == 0, f"H={H} not divisible by G={G}"
    return x.view(b, G, H // G, L, d).mean(dim=2)


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
    # Selector inputs as P prototypes per block (mean + outliers), so a single-token
    # spike (the needle) survives pooling. K is pre-GQA-repeat -> already per kv-group.
    # Q is averaged H->G per token before pooling so prototypes are real per-group tokens.
    q_grouped = _group_avg_tokens(query, key.shape[1])     # [1, G, q_len, d]
    pooledQ = _proto_pool_seq(q_grouped, _BLOCK, _PROTO).cpu()   # [1, G, qb, P, d]
    pooledK = _proto_pool_seq(key, _BLOCK, _PROTO).cpu()        # [1, G, kb, P, d]
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


def _stack_and_clear_captures(order):
    """Stack one target's layer captures, then release the layerwise copies."""
    try:
        pooled_stack = torch.stack(
            [_CAPTURE[i] for i in order], dim=1)
        pooledQ_stack = torch.stack(
            [_CAPTURE_QK[i][0] for i in order], dim=1)
        pooledK_stack = torch.stack(
            [_CAPTURE_QK[i][1] for i in order], dim=1)
        return pooled_stack, pooledQ_stack, pooledK_stack
    finally:
        # torch.stack owns new storage. Keeping the per-layer tensors here
        # would otherwise retain a second full target until the next call.
        _CAPTURE.clear()
        _CAPTURE_QK.clear()


@torch.no_grad()
def get_pooled_targets(model, tok, prompt, device="cuda", block_size=_BLOCK):
    """
    One forward pass; captures chunked pooled block-mass per layer.
    Returns (pooled_stack [1, num_layers, num_heads, q_blocks, k_blocks] fp32 cpu,
             pooledQ_stack [1, num_layers, G, q_blocks, P, d] model-dtype cpu (e.g. bf16),
             pooledK_stack [1, num_layers, G, k_blocks, P, d] model-dtype cpu (e.g. bf16),
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

    seq_len = inputs["input_ids"].shape[1]
    order = sorted(_CAPTURE)
    pooled_stack, pooledQ_stack, pooledK_stack = _stack_and_clear_captures(order)
    del inputs
    return pooled_stack, pooledQ_stack, pooledK_stack, seq_len
