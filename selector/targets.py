"""Load a teacher .pt and turn it into training tensors for the selector.

Produces, per document:
  q_feat : [L, G, qb, d]   pooled query features, averaged over the query heads in
                           each kv_head_group  (selector INPUT)
  k_feat : [L, G, kb, d]   pooled key features per kv_head_group                (INPUT)
  target : [L, G, qb, kb]  teacher block-marginal p^t, row-normalized over the
                           causal keys of each query block                     (TARGET)
  cmask  : [qb, kb] bool   True where key-block k is causally visible to query-block q

G = num_kv_heads (the selected_blocks kv_head_group axis). Qwen2.5-Coder-1.5B: H=12, G=2.
The teacher `pooled` mass is per query head (H repeated); we average the H/G query heads
inside each group so target, q_feat and k_feat all share the G axis.
"""
import torch


def causal_block_mask(qb, kb, device=None):
    """Block-level causal mask: query-block q may attend key-block k iff k <= q.
    Key-block q itself is the (partially visible) diagonal and is included."""
    q = torch.arange(qb, device=device)[:, None]
    k = torch.arange(kb, device=device)[None, :]
    return k <= q                                   # [qb, kb] bool


def load_teacher(path, device="cpu", eps=1e-9):
    d = torch.load(path, map_location=device)
    if "pooledQ" not in d or "pooledK" not in d:
        raise KeyError(
            f"{path} has no 'pooledQ'/'pooledK' — it predates the selector-input dump. "
            f"Re-run scripts/extract_teacher_targets.py to regenerate it.")

    pooled = d["pooled"].float()      # [1, L, H, qb, kb]  teacher mass
    q_pool = d["pooledQ"].float()     # [1, L, H, qb, dd]
    k_pool = d["pooledK"].float()     # [1, L, G, kb, dd]
    pooled, q_pool, k_pool = pooled[0], q_pool[0], k_pool[0]   # drop batch

    L, H, qb, kb = pooled.shape
    G = k_pool.shape[1]
    assert H % G == 0, f"H={H} not divisible by G={G}"
    rep = H // G

    # Average the query heads inside each kv group -> shared G axis.
    mass = pooled.view(L, G, rep, qb, kb).mean(dim=2)          # [L, G, qb, kb]
    q_feat = q_pool.view(L, G, rep, qb, -1).mean(dim=2)        # [L, G, qb, dd]
    k_feat = k_pool                                            # [L, G, kb, dd]

    # Row-normalize over causal keys -> teacher marginal p^t. Future keys zeroed first
    # (chunked_pool already left them ~0; make it exact so normalization is clean).
    cmask = causal_block_mask(qb, kb, device=mass.device)     # [qb, kb]
    mass = mass * cmask[None, None]                            # kill any future leakage
    row = mass.sum(dim=-1, keepdim=True)                      # [L, G, qb, 1]
    target = mass / row.clamp_min(eps)

    meta = {
        "seq_len": int(d["seq_len"]), "block_size": int(d["block_size"]),
        "needle_block": int(d.get("needle_block", -1)),
        "num_layers": L, "num_groups": G, "qb": qb, "kb": kb,
        "head_dim": q_feat.shape[-1],
    }
    return {"q_feat": q_feat, "k_feat": k_feat, "target": target,
            "cmask": cmask, "row_mass": row.squeeze(-1), "meta": meta}
