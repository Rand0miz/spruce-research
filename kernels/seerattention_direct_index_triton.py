"""Direct-index SPRUCE Triton sparse-prefill kernel.

Adapted from the online-softmax forward algorithm in Microsoft SeerAttention's
MIT-licensed ``block_sparse_attn.py``.  Unlike that mask-scanning kernel, this
version receives exactly K selected key-block IDs per query block and never
iterates over unselected causal blocks.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config(
                {"QUERY_TILE": 16, "HEAD_TILE": 1},
                num_warps=4, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 16, "HEAD_TILE": 2},
                num_warps=4, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 16, "HEAD_TILE": 4},
                num_warps=8, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 32, "HEAD_TILE": 1},
                num_warps=4, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 32, "HEAD_TILE": 2},
                num_warps=8, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 64, "HEAD_TILE": 1},
                num_warps=4, num_stages=2),
            triton.Config(
                {"QUERY_TILE": 64, "HEAD_TILE": 1},
                num_warps=8, num_stages=2),
        ],
        key=["D", "MAX_K", "GROUP_SIZE"],
    )
    @triton.jit
    def _direct_kernel(Q, K, V, Indices, Out, scale,
                       sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd,
                       siz, sih, sim, sik, soz, soh, som, sod, HKV, HQ, N,
                       GROUP_SIZE: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr, BN: tl.constexpr,
                       D: tl.constexpr, MAX_K: tl.constexpr,
                       QUERY_TILE: tl.constexpr, HEAD_TILE: tl.constexpr):
        query_tile = tl.program_id(0)
        hz = tl.program_id(1)
        head_tile = tl.program_id(2)
        kv_head, z = hz % HKV, hz // HKV
        q_head_start = kv_head * GROUP_SIZE + head_tile * HEAD_TILE
        q_heads = q_head_start + tl.arange(0, HEAD_TILE)
        valid_heads = q_heads < (kv_head + 1) * GROUP_SIZE
        Q += z * sqz
        K += z * skz + kv_head * skh
        V += z * svz + kv_head * svh
        q_start = query_tile * QUERY_TILE
        query_block = q_start // BLOCK_SIZE
        Indices += z * siz + kv_head * sih + query_block * sim
        om = q_start + tl.arange(0, QUERY_TILE)
        on = tl.arange(0, BN)
        od = tl.arange(0, D)
        q_mask = valid_heads[:, None, None] & (om[None, :, None] < N)
        q = tl.load(
            Q + q_heads[:, None, None] * sqh
            + om[None, :, None] * sqm + od[None, None, :] * sqd,
            mask=q_mask,
            other=0.0,
        )
        # Scale Q once instead of scaling every selected block's QK matrix.
        q = (q * (scale * 1.4426950408889634)).to(
            q.type.element_ty)
        acc = tl.zeros([HEAD_TILE, QUERY_TILE, D], tl.float32)
        l_i = tl.zeros([HEAD_TILE, QUERY_TILE], tl.float32)
        m_i = tl.full(
            [HEAD_TILE, QUERY_TILE], float("-inf"), tl.float32)

        # MAX_K is compile-time specialized from selected_blocks.shape[-1].
        # Every iteration is a useful selected block, except trailing -1 pads.
        for slot in range(0, MAX_K):
            block_id = tl.load(Indices + slot * sik)
            if block_id >= 0:
                start = block_id * BN
                key_tokens = on + start
                key_valid = key_tokens < N
                k_2d = tl.load(
                    K + key_tokens[None, :] * skn + od[:, None] * skd,
                    mask=key_valid[None, :],
                    other=0.0,
                )
                k = tl.broadcast_to(
                    k_2d[None, :, :], (HEAD_TILE, D, BN))
                qk = tl.dot(q, k)
                # All earlier blocks are fully causal. Only the diagonal block
                # needs a token-level triangular comparison.
                if block_id < query_block:
                    token_mask = (
                        valid_heads[:, None, None]
                        & (om[None, :, None] < N)
                        & key_valid[None, None, :]
                    )
                elif block_id == query_block:
                    token_mask = (
                        valid_heads[:, None, None]
                        & (om[None, :, None] < N)
                        & key_valid[None, None, :]
                        & (om[None, :, None] >= key_tokens[None, None, :])
                    )
                else:
                    token_mask = tl.full(
                        (HEAD_TILE, QUERY_TILE, BN), False, tl.int1)
                qk = tl.where(token_mask, qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, 2))
                p = tl.exp2(qk - m_new[:, :, None])
                alpha = tl.exp2(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, 2)
                acc *= alpha[:, :, None]
                v_2d = tl.load(
                    V + key_tokens[:, None] * svn + od[None, :] * svd,
                    mask=key_valid[:, None],
                    other=0.0,
                )
                v = tl.broadcast_to(
                    v_2d[None, :, :], (HEAD_TILE, BN, D))
                acc += tl.dot(p.to(v.type.element_ty), v)
                m_i = m_new

        denominator = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(
            Out + z * soz + q_heads[:, None, None] * soh
            + om[None, :, None] * som + od[None, None, :] * sod,
            (acc / denominator[:, :, None]).to(Out.type.element_ty),
            mask=q_mask,
        )


def direct_index_sparse_triton(q, k, v, indices, scale, block_size=64):
    """Causal block-sparse attention over exactly the supplied IDs.

    q is ``[B,Hq,T,D]``; k/v and indices retain native KV-group resolution
    ``[B,Hkv,T,D]`` / ``[B,Hkv,ceil(T/64),K]``. The kernel maps each query
    head to its KV head without materializing repeated K/V tensors. Inputs
    remain in the strided views produced by Qwen projections, and output is
    written directly as contiguous ``[B,T,Hq,D]`` for Qwen's output reshape.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed; install a CUDA-compatible triton build")
    if not q.is_cuda:
        raise RuntimeError("the direct-index Triton backend requires CUDA tensors")
    if block_size != 64:
        raise ValueError("direct-index kernel currently supports block_size=64 only")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k/v must match in batch, tokens, and head_dim")
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv:
        raise ValueError("query heads must be divisible by KV heads")
    qblocks = (T + block_size - 1) // block_size
    if indices.shape[:3] != (B, Hkv, qblocks) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [B,Hkv,ceil(T/64),K]")
    if D not in (64, 128):
        raise ValueError(f"kernel supports head_dim 64 or 128, got {D}")
    if any(x.stride(-1) != 1 for x in (q, k, v, indices)):
        raise ValueError("q, k, v, and indices must have a contiguous final dimension")
    out = torch.empty((B, T, Hq, D), device=q.device, dtype=q.dtype)
    # The kernel indexes output logically as [B,Hq,T,D]. Pass the matching
    # logical strides while retaining Qwen's contiguous [B,T,Hq,D] storage.
    out_strides = (out.stride(0), out.stride(2), out.stride(1), out.stride(3))
    grid = lambda meta: (
        triton.cdiv(T, meta["QUERY_TILE"]),
        B * Hkv,
        triton.cdiv(Hq // Hkv, meta["HEAD_TILE"]),
    )
    _direct_kernel[grid](
        q, k, v, indices, out, scale,
        *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *out_strides,
        Hkv, Hq, T,
        GROUP_SIZE=Hq // Hkv, BLOCK_SIZE=block_size, BN=block_size, D=D,
        MAX_K=indices.shape[-1],
    )
    return out
