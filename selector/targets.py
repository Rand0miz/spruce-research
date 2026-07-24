"""Load a teacher .pt and turn it into training tensors for the selector.

Produces, per document:
  q_feat : [L, G, qb, P, d]  pooled query prototype features, already averaged over
                             the query heads in each kv_head_group during extraction
                             (selector INPUT; no head averaging happens here)
  k_feat : [L, G, kb, P, d]  pooled key prototype features per kv_head_group  (INPUT)
  target : [L, G, qb, kb]    teacher block-marginal p^t, row-normalized over the
                             causal keys of each query block                 (TARGET)
  cmask  : [qb, kb] bool     True where key-block k is causally visible to query-block q

G = num_kv_heads (the selected_blocks kv_head_group axis). Qwen2.5-Coder-1.5B: H=12, G=2.
P = number of prototypes per block (proto 0 is the block mean, protos 1..P-1 are the
tokens farthest from it). The teacher `pooled` mass is still per query head (H repeated);
we average the H/G query heads inside each group so target shares the G axis with
q_feat/k_feat.
"""
import torch


def load_selector_features(path, device="cpu"):
    """Load only pooled Q/K features needed by live selector traversal.

    The checkpoint is mapped to CPU first so dense teacher-mass tensors are
    never copied to the selector GPU. Only q_feat/k_feat move to ``device``.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("proto") is None or d["pooledK"].dim() != 6:
        raise KeyError(
            f"{path} lacks P-prototype selector features ('proto' key / 6-D pooledK). "
            "Re-extract with scripts/extract_teacher_targets.py to regenerate it.")

    q_feat = d["pooledQ"].float()[0].to(device)
    k_feat = d["pooledK"].float()[0].to(device)
    L, G, qb, _, head_dim = q_feat.shape
    kb = k_feat.shape[2]
    meta = {
        "seq_len": int(d["seq_len"]), "block_size": int(d["block_size"]),
        "needle_block": int(d.get("needle_block", -1)),
        "num_layers": L, "num_groups": G, "qb": qb, "kb": kb,
        "head_dim": head_dim, "proto": int(d["proto"]),
        "rope": d.get("rope"),
        "features_only": bool(d.get("features_only", False)),
    }
    return {"q_feat": q_feat, "k_feat": k_feat, "meta": meta}


def causal_block_mask(qb, kb, device=None):
    """Block-level causal mask: query-block q may attend key-block k iff k <= q.
    Key-block q itself is the (partially visible) diagonal and is included."""
    q = torch.arange(qb, device=device)[:, None]
    k = torch.arange(kb, device=device)[None, :]
    return k <= q                                   # [qb, kb] bool


def load_teacher(path, device="cpu", eps=1e-9):
    d = torch.load(path, map_location=device)
    if d.get("proto") is None or d["pooledK"].dim() != 6:
        raise KeyError(
            f"{path} lacks P-prototype selector features ('proto' key / 6-D pooledK). "
            f"It predates multi-prototype extraction. Re-extract with "
            f"scripts/extract_teacher_targets.py to regenerate it.")
    if d.get("pooled") is None:
        raise ValueError(
            f"{path} is a selector-feature-only artifact and has no dense "
            "teacher mass; it can be used for traversal/128K replay but not training")

    pooled = d["pooled"].float()      # [1, L, H, qb, kb]  teacher mass
    q_feat = d["pooledQ"].float()[0]  # [L, G, qb, P, dd]  selector input (already grouped)
    k_feat = d["pooledK"].float()[0]  # [L, G, kb, P, dd]
    pooled = pooled[0]                # [L, H, qb, kb]

    L, H, qb, kb = pooled.shape
    G = k_feat.shape[1]
    assert H % G == 0, f"H={H} not divisible by G={G}"
    rep = H // G

    # Average the teacher mass over the query heads inside each kv group -> shared G axis.
    mass = pooled.view(L, G, rep, qb, kb).mean(dim=2)         # [L, G, qb, kb]

    # Row-normalize over causal keys -> teacher marginal p^t.
    cmask = causal_block_mask(qb, kb, device=mass.device)    # [qb, kb]
    mass = mass * cmask[None, None]                          # kill any future leakage
    row = mass.sum(dim=-1, keepdim=True)                     # [L, G, qb, 1]
    target = mass / row.clamp_min(eps)

    meta = {
        "seq_len": int(d["seq_len"]), "block_size": int(d["block_size"]),
        "needle_block": int(d.get("needle_block", -1)),
        "num_layers": L, "num_groups": G, "qb": qb, "kb": kb,
        "head_dim": q_feat.shape[-1], "proto": int(d["proto"]),
    }
    return {"q_feat": q_feat, "k_feat": k_feat, "target": target,
            "cmask": cmask, "row_mass": row.squeeze(-1), "meta": meta}
