"""Distillation loss — forward KL with the teacher as reference.

Verified against the anchor papers (not memory):
  SeerAttention (2410.13276):  loss = D_KL(gt || score)
  SpotAttention (2606.22874):  L = sum_t p^t log(p^t / p^s),  p^s = softmax_t(scores)
Both are forward KL, teacher normalized, student = softmax over the candidate keys.

Here the candidate set per query-block is its causal key-blocks (k <= q). This is the
flat / leaf-level loss; the tree sums the same KL across coarser levels (step 2).
"""
import torch
import torch.nn.functional as F


def kl_loss(scores, target, cmask, eps=1e-9, row_valid_thresh=0.5):
    """scores [L,G,qb,kb] raw, target [L,G,qb,kb] teacher marginal (causal, row-sum 1),
    cmask [qb,kb] bool. Returns (mean forward-KL over valid rows, num_valid_rows)."""
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~cmask[None, None], neg_inf)
    log_s = F.log_softmax(masked, dim=-1)                    # [L,G,qb,kb]

    log_t = target.clamp_min(eps).log()
    contrib = torch.where(target > 0, target * (log_t - log_s),
                          torch.zeros_like(target))
    kl_row = contrib.sum(dim=-1)                             # [L,G,qb]

    # A row is valid only if the teacher gave it a real distribution (row-sum ~ 1).
    valid = target.sum(dim=-1) > row_valid_thresh           # [L,G,qb]
    n = valid.sum().clamp_min(1)
    loss = (kl_row * valid).sum() / n
    return loss, int(valid.sum())


def topk_membership_loss(scores, target, cmask, k=8, row_valid_thresh=0.5):
    """Balanced BCE that forces the gate to rank the teacher's top-k blocks high.

    KL matches attention *mass*, so a small-mass but retrieval-critical block (the
    needle) can be dropped for near-zero KL cost. This term is mass-blind: it turns
    the teacher's top-k blocks into POSITIVES (label 1) and the remaining causal
    blocks into NEGATIVES (label 0), then pushes student scores up on positives and
    down on negatives via logsigmoid. Positives and negatives are averaged separately
    so the k positives are not drowned out by the many negatives (class balance).

    scores/target [L,G,qb,kb], cmask [qb,kb]. Returns (loss, num_positive_blocks).
    """
    L, G, qb, kb = scores.shape
    kk = min(k, kb)
    t_top = target.topk(kk, dim=-1).indices                  # [L,G,qb,kk]
    member = torch.zeros_like(target, dtype=torch.bool)
    member.scatter_(-1, t_top, True)

    # Restrict positives to blocks the teacher actually attends (target>0). On short
    # query rows topk() would otherwise pad the set with zero-mass future blocks.
    valid = (target.sum(dim=-1) > row_valid_thresh)[..., None]   # [L,G,qb,1]
    pos = member & (target > 0) & valid                      # teacher top-k, causal
    neg = cmask[None, None] & (~pos) & valid                 # other causal candidates

    logp = F.logsigmoid(scores)                              # score high  -> loss low
    logn = F.logsigmoid(-scores)                             # score low   -> loss low
    p = pos.to(logp.dtype)
    ng = neg.to(logn.dtype)
    lp = -(logp * p).sum() / p.sum().clamp_min(1)            # avg over positives
    ln = -(logn * ng).sum() / ng.sum().clamp_min(1)          # avg over negatives
    return 0.5 * (lp + ln), int(pos.sum())


def combined_loss(scores, target, cmask, lambda_topk=0.5, k=8):
    """KL (mass matching) + lambda * top-k membership (needle retention).
    Returns (total, {kl, bce, n_valid})."""
    kl, nv = kl_loss(scores, target, cmask)
    bce, _ = topk_membership_loss(scores, target, cmask, k=k)
    total = kl + lambda_topk * bce
    return total, {"kl": float(kl), "bce": float(bce), "n": nv}
