"""SeerAttention block-sparse Triton forward kernel, adapted for SPRUCE.

Derived from ``seer_attn/kernels/block_sparse_attn.py`` in Microsoft
SeerAttention (MIT License, Copyright Microsoft Corporation; original author
Eric Lin).  This file retains the kernel's causal 64-token block algorithm;
SPRUCE supplies its own selected-block mask adapter in ``sparse_prefill.py``.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # import the toolkit on CPU; fail only if backend is used
    triton = tl = None


if triton is not None:
    @triton.jit
    def _inner(acc, l_i, m_i, q, col, mask_ptr, k_ptrs, v_ptrs, offs_m, offs_n,
               stride_kn, stride_vn, stride_mask_n, scale, seqlen, LAST: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
        enabled = tl.load(mask_ptr + col * stride_mask_n)
        if enabled:
            start = col * BN
            k = tl.load(k_ptrs + start * stride_kn,
                        mask=offs_n[None, :] + start < seqlen if LAST else None)
            qk = tl.dot(q, k) * scale
            if LAST:
                qk += tl.where(offs_m[:, None] >= start + offs_n[None, :], 0.0, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc *= alpha[:, None]
            v = tl.load(v_ptrs + start * stride_vn,
                        mask=offs_n[:, None] + start < seqlen if LAST else None)
            acc += tl.dot(p.to(v.type.element_ty), v)
            m_i = m_new
        return acc, l_i, m_i

    @triton.jit
    def _kernel(Q, K, V, scale, mask, Out,
                sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd,
                smz, smh, smm, smn, soz, soh, som, sod, H, N,
                BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
        block_row = tl.program_id(0)
        hz = tl.program_id(1)
        h, z = hz % H, hz // H
        Q += z * sqz + h * sqh
        K += z * skz + h * skh
        V += z * svz + h * svh
        mask += z * smz + h * smh + block_row * smm
        om, on, od = block_row * BM + tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, D)
        q = tl.load(Q + om[:, None] * sqm + od[None, :] * sqd, mask=om[:, None] < N)
        kp = K + on[None, :] * skn + od[:, None] * skd
        vp = V + on[:, None] * svn + od[None, :] * svd
        acc = tl.zeros([BM, D], tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        m_i = tl.full([BM], float("-inf"), tl.float32)
        last = tl.cdiv((block_row + 1) * BM, BN) - 1
        for col in range(0, last):
            acc, l_i, m_i = _inner(acc, l_i, m_i, q, col, mask, kp, vp, om, on,
                                    skn, svn, smn, scale, N, False, BM, BN)
        acc, l_i, m_i = _inner(acc, l_i, m_i, q, last, mask, kp, vp, om, on,
                                skn, svn, smn, scale, N, True, BM, BN)
        acc /= l_i[:, None]
        tl.store(Out + z * soz + h * soh + om[:, None] * som + od[None, :] * sod,
                 acc.to(Out.type.element_ty), mask=om[:, None] < N)


def block_sparse_triton(q, k, v, block_mask, scale, block_size=64):
    """Run the SeerAttention-derived causal kernel on contiguous BHSD tensors."""
    if triton is None:
        raise RuntimeError("Triton is not installed; install a CUDA-compatible triton build")
    if not q.is_cuda:
        raise RuntimeError("the Triton sparse-prefill backend requires CUDA tensors")
    if block_size != 64:
        raise ValueError("the vendored SeerAttention kernel currently supports block_size=64 only")
    if q.shape != k.shape or q.shape != v.shape or q.shape[2] != k.shape[2]:
        raise ValueError("q, k, and v must have matching [B,H,T,D] shapes")
    if q.shape[-1] not in (64, 128):
        raise ValueError(f"kernel supports head_dim 64 or 128, got {q.shape[-1]}")
    q, k, v, block_mask = (x.contiguous() for x in (q, k, v, block_mask))
    out = torch.empty_like(q)
    B, H, T, D = q.shape
    grid = (triton.cdiv(T, block_size), B * H)
    _kernel[grid](q, k, v, scale, block_mask, out,
                  *q.stride(), *k.stride(), *v.stride(), *block_mask.stride(), *out.stride(),
                  H, T, BM=block_size, BN=block_size, D=D, num_warps=4, num_stages=2)
    return out
