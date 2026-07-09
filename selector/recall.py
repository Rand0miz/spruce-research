"""Recall metrics — the real KS1 signal, separate from loss.

Loss falling does not prove the ranking is right. What KS1 checks: does the selector's
top-r contain the teacher's top-r? And, on needle prompts, does the reader row's top-k
keep the needle block? Mirrors the checks in scripts/ks1_lite.py, now on the trained gate.
"""
import torch


@torch.no_grad()
def recall_metrics(scores, target, cmask, budgets=(1, 2, 4, 8, 16), needle_block=None):
    """scores/target [L,G,qb,kb], cmask [qb,kb]. Returns dict:
       recall@r : mean over rows of |student_topr ∩ teacher_topr| / r
       needle_hit@k (reader row) : fraction of (layer,group) whose reader-row top-k
                                   contains needle_block  (only if needle_block given)."""
    L, G, qb, kb = scores.shape
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~cmask[None, None], neg_inf)

    valid = target.sum(dim=-1) > 0.5                        # [L,G,qb] real teacher rows
    q_idx = torch.arange(qb, device=scores.device)          # causal keys available = q+1
    out = {}

    for r in budgets:
        rr = min(r, kb)
        s_top = masked.topk(rr, dim=-1).indices            # [L,G,qb,rr]
        t_top = target.topk(rr, dim=-1).indices
        member = torch.zeros_like(target, dtype=torch.bool)
        member.scatter_(-1, s_top, True)                   # student's chosen set
        overlap = member.gather(-1, t_top).sum(dim=-1).float()   # [L,G,qb] in [0,rr]
        enough = (q_idx + 1 >= rr)[None, None].expand(L, G, qb)  # row has >= rr keys
        m = valid & enough
        out[f"recall@{r}"] = float((overlap / rr * m).sum() / m.sum().clamp_min(1))

        # Mass coverage: fraction of the teacher's attention mass that lands inside
        # the student's top-r blocks. target rows sum to 1, so (target*member) is the
        # captured mass. This is what actually matters for sparse attention — a
        # rank-swap among near-tied blocks barely dents coverage but tanks set-recall.
        coverage = (target * member).sum(dim=-1)                 # [L,G,qb] in [0,1]
        out[f"coverage@{r}"] = float((coverage * m).sum() / m.sum().clamp_min(1))

        # Oracle ceiling: coverage of the teacher's OWN top-r blocks. This is the max
        # mass any top-r selection can capture. student coverage / oracle = how much of
        # the achievable mass the gate got. If oracle itself is low, attention is
        # diffuse and no small budget can cover it — a data property, not a gate fault.
        omember = torch.zeros_like(target, dtype=torch.bool)
        omember.scatter_(-1, t_top, True)
        ocov = (target * omember).sum(dim=-1)                    # [L,G,qb] in [0,1]
        out[f"oracle_cov@{r}"] = float((ocov * m).sum() / m.sum().clamp_min(1))

    if needle_block is not None and 0 <= needle_block < kb:
        reader = masked[:, :, -1, :]                        # [L,G,kb] last query block
        for k in budgets:
            kk = min(k, kb)
            top = reader.topk(kk, dim=-1).indices           # [L,G,kk]
            hit = (top == needle_block).any(dim=-1).float() # [L,G]
            out[f"needle_hit@{k}"] = float(hit.mean())
    return out
