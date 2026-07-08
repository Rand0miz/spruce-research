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
