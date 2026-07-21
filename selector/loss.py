"""Distillation loss -- forward KL with the teacher as reference.

Verified against the anchor papers (not memory):
  SeerAttention (2410.13276):  loss = D_KL(gt || score)
  SpotAttention (2606.22874):  L = sum_t p^t log(p^t / p^s),  p^s = softmax_t(scores)
Both are forward KL, teacher normalized, student = softmax over the candidate keys.

Here the candidate set per query-block is its causal key-blocks (k <= q). This is the
flat / leaf-level loss; the tree sums the same KL across coarser levels.
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
    valid = target.sum(dim=-1) > row_valid_thresh            # [L,G,qb]
    n = valid.sum().clamp_min(1)
    loss = (kl_row * valid).sum() / n
    return loss, int(valid.sum())


def topk_membership_loss(scores, target, cmask, k=8, row_valid_thresh=0.5):
    """Balanced BCE that forces the gate to rank the teacher's top-k blocks high.

    KL matches attention mass, so a small-mass but retrieval-critical block can be
    dropped for near-zero KL cost. This term is mass-blind: it turns the teacher's
    top-k blocks into positives and the remaining causal blocks into negatives,
    then pushes student scores up on positives and down on negatives via logsigmoid.

    scores/target [L,G,qb,kb], cmask [qb,kb]. Returns (loss, num_positive_blocks).
    """
    L, G, qb, kb = scores.shape
    kk = min(k, kb)
    t_top = target.topk(kk, dim=-1).indices                  # [L,G,qb,kk]
    member = torch.zeros_like(target, dtype=torch.bool)
    member.scatter_(-1, t_top, True)

    # Restrict positives to blocks the teacher actually attends. On short query rows
    # topk() would otherwise pad the set with zero-mass future blocks.
    valid = (target.sum(dim=-1) > row_valid_thresh)[..., None]   # [L,G,qb,1]
    pos = member & (target > 0) & valid                      # teacher top-k, causal
    neg = cmask[None, None] & (~pos) & valid                 # other causal candidates

    logp = F.logsigmoid(scores)                              # score high -> loss low
    logn = F.logsigmoid(-scores)                             # score low -> loss low
    p = pos.to(logp.dtype)
    ng = neg.to(logn.dtype)
    lp = -(logp * p).sum() / p.sum().clamp_min(1)            # avg over positives
    ln = -(logn * ng).sum() / ng.sum().clamp_min(1)          # avg over negatives
    return 0.5 * (lp + ln), int(pos.sum())


def needle_topk_loss(scores, target, cmask, needle_block, k=8,
                     margin=0.0, row_valid_thresh=0.5):
    """Encourage the reader row to retain a known needle in its top-k blocks.

    Only the final query block (the question/reader row) is supervised. A group
    participates only when the teacher itself puts ``needle_block`` in its top-k.
    The softplus term compares the needle score with the k-th best other causal
    block, a differentiable surrogate for student top-k membership.

    Returns ``(loss, eligible_layer_groups)``. Invalid needle metadata or a
    teacher that never selects the needle produces a zero, gradient-safe loss.
    """
    _, _, qb, kb = scores.shape
    if needle_block is None or not 0 <= int(needle_block) < kb:
        return scores.sum() * 0.0, 0

    needle_block = int(needle_block)
    visible = cmask[-1]
    if not bool(visible[needle_block]):
        return scores.sum() * 0.0, 0

    # With fewer than k competing blocks, the needle is automatically top-k.
    competitors = int(visible.sum()) - 1
    kk = min(int(k), competitors)
    if kk <= 0:
        return scores.sum() * 0.0, 0

    reader_scores = scores[:, :, qb - 1, :]
    reader_target = target[:, :, qb - 1, :]
    teacher_top = reader_target.topk(min(int(k), kb), dim=-1).indices
    teacher_keeps_needle = (teacher_top == needle_block).any(dim=-1)
    valid = reader_target.sum(dim=-1) > row_valid_thresh
    eligible = teacher_keeps_needle & valid
    n = int(eligible.sum())
    if n == 0:
        return scores.sum() * 0.0, 0

    neg_inf = torch.finfo(scores.dtype).min
    other_scores = reader_scores.masked_fill(~visible[None, None], neg_inf)
    other_scores[:, :, needle_block] = neg_inf
    threshold = other_scores.topk(kk, dim=-1).values[..., -1]
    needle_scores = reader_scores[:, :, needle_block]
    loss = F.softplus(threshold - needle_scores + margin)
    return (loss * eligible).sum() / eligible.sum(), n
def topk_set_loss(scores, target, cmask, topk=8, margin=0.0, row_valid_thresh=0.5):
    """Compatibility wrapper for older training/test code.

    The bad merge path used a margin-ranking helper named topk_set_loss. The current
    recipe uses the balanced top-k membership BCE above. Keep the old name callable
    so stale scripts fail less often, but reject non-zero margins because they are
    not part of this loss.
    """
    if margin:
        raise ValueError("topk_set_loss no longer supports margin; use margin=0.0")
    return topk_membership_loss(
        scores, target, cmask, k=topk, row_valid_thresh=row_valid_thresh)


def combined_loss(scores, target, cmask, lambda_topk=0.5, k=8):
    """KL (mass matching) + lambda * top-k membership (needle retention).
    Returns (total, {kl, bce, n_valid})."""
    kl, nv = kl_loss(scores, target, cmask)
    bce, _ = topk_membership_loss(scores, target, cmask, k=k)
    total = kl + lambda_topk * bce
    return total, {"kl": float(kl), "bce": float(bce), "n": nv}

