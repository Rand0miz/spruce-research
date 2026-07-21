"""Recall metrics — the real KS1 signal, separate from loss.

Loss falling does not prove the ranking is right. What KS1 checks: does the selector's
top-r contain the teacher's top-r? And, on needle prompts, does the reader row's top-k
keep the needle block? Mirrors the checks in scripts/ks1_lite.py, now on the trained gate.
"""
import torch


def _ratio(student, teacher, eps=1e-9):
    """student/teacher, clamped to [0,1]. Returns nan when the teacher itself is 0 --
    there is nothing to preserve, so the gate can be neither right nor wrong."""
    if teacher <= eps:
        return float("nan")
    return float(min(student / teacher, 1.0))


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

        # Fraction of the ACHIEVABLE mass the gate got. This, not recall@r, is the
        # honest "how close to the teacher" number: recall@r docks a full point for
        # swapping two near-tied blocks carrying no mass; cov_ratio does not.
        out[f"cov_ratio@{r}"] = _ratio(out[f"coverage@{r}"], out[f"oracle_cov@{r}"])

    if needle_block is not None and 0 <= needle_block < kb:
        reader = masked[:, :, -1, :]                        # [L,G,kb] last query block
        # Teacher oracle: does the teacher's OWN reader-row top-k keep the needle?
        # If this is low, the mass target itself ranks the needle out -> the gate is
        # being TAUGHT to drop it (mass-KL vs retrieval tension, not a recipe fault).
        # If this is high but needle_hit is low, it's a gate/recipe problem.
        treader = target.masked_fill(~cmask[None, None], 0.0)[:, :, -1, :]  # [L,G,kb]
        for k in budgets:
            kk = min(k, kb)
            top = reader.topk(kk, dim=-1).indices           # [L,G,kk]
            hit = (top == needle_block).any(dim=-1)         # [L,G] bool
            out[f"needle_hit@{k}"] = float(hit.float().mean())
            ttop = treader.topk(kk, dim=-1).indices          # [L,G,kk]
            thit = (ttop == needle_block).any(dim=-1)        # [L,G] bool
            out[f"teacher_needle@{k}"] = float(thit.float().mean())

            # The per-head mean above is pessimistic. Most heads are local/sink heads
            # that were never going to read the needle -- they drag the mean down for
            # teacher and gate alike. selected_blocks is per kv_head_group, so a layer
            # reads the UNION of its groups' choices, and evidence only has to survive
            # in ONE group per layer to reach the next layer. That is the quantity the
            # retrieval-preservation claim rests on.
            out[f"needle_union@{k}"] = float(hit.any(dim=1).float().mean())
            out[f"teacher_needle_union@{k}"] = float(thit.any(dim=1).float().mean())

            # Preservation THROUGH the conversion: the teacher is the ceiling, so score
            # against it, not against 1.0. teacher_needle@k < 1 is a property of the
            # backbone, not a gate failure.
            out[f"ndl_ratio@{k}"] = _ratio(
                out[f"needle_hit@{k}"], out[f"teacher_needle@{k}"])
            out[f"ndl_union_ratio@{k}"] = _ratio(
                out[f"needle_union@{k}"], out[f"teacher_needle_union@{k}"])
    return out
