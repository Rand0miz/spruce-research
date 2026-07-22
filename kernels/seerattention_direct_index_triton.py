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
    @triton.jit
    def _direct_kernel(Q, K, V, Indices, Out, scale,
                       sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd,
                       siz, sih, sim, sik, soz, soh, som, sod, HQ, N,
                       GROUP_SIZE: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, MAX_K: tl.constexpr):
        query_block = tl.program_id(0)
        hz = tl.program_id(1)
        q_head, z = hz % HQ, hz // HQ
        kv_head = q_head // GROUP_SIZE
        Q += z * sqz + q_head * sqh
        K += z * skz + kv_head * skh
        V += z * svz + kv_head * svh
        Indices += z * siz + kv_head * sih + query_block * sim
        om, on, od = query_block * BM + tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, D)
        q = tl.load(Q + om[:, None] * sqm + od[None, :] * sqd, mask=om[:, None] < N)
        acc = tl.zeros([BM, D], tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        m_i = tl.full([BM], float("-inf"), tl.float32)

        # MAX_K is compile-time specialized from selected_blocks.shape[-1].
        # Every iteration is a useful selected block, except trailing -1 pads.
        for slot in range(0, MAX_K):
            block_id = tl.load(Indices + slot * sik)
            if block_id >= 0:
                start = block_id * BN
                token_mask = (
                    (on[None, :] + start < N)
                    & (om[:, None] >= on[None, :] + start)
                )
                k = tl.load(
                    K + (on[None, :] + start) * skn + od[:, None] * skd,
                    mask=on[None, :] + start < N,
                )
                qk = tl.dot(q, k) * scale
                qk = tl.where(token_mask, qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_new[:, None])
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc *= alpha[:, None]
                v = tl.load(
                    V + (on[:, None] + start) * svn + od[None, :] * svd,
                    mask=on[:, None] + start < N,
                )
                acc += tl.dot(p.to(v.type.element_ty), v)
                m_i = m_new

        tl.store(
            Out + z * soz + h * soh + om[:, None] * som + od[None, :] * sod,
            (acc / l_i[:, None]).to(Out.type.element_ty),
            mask=om[:, None] < N,
        )


def direct_index_sparse_triton(q, k, v, indices, scale, block_size=64):
    """Causal block-sparse attention over exactly the supplied IDs.

    q is ``[B,Hq,T,D]``; k/v and indices retain native KV-group resolution
    ``[B,Hkv,T,D]`` / ``[B,Hkv,ceil(T/64),K]``. The kernel maps each query
    head to its KV head without materializing repeated K/V tensors.
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
    q, k, v, indices = (x.contiguous() for x in (q, k, v, indices))
    out = torch.empty_like(q)
    _direct_kernel[(qblocks, B * Hq)](
        q, k, v, indices, out, scale,
        *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *out.stride(), Hq, T,
        GROUP_SIZE=Hq // Hkv, BM=block_size, BN=block_size, D=D,
        MAX_K=indices.shape[-1], num_warps=4, num_stages=2,
    )
    return out
