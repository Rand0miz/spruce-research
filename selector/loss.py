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


def topk_boundary_loss(scores, target, cmask, k=8, margin=0.0,
                       row_valid_thresh=0.5):
    """Rank every teacher top-k block above the hardest causal non-top-k block.

    Balanced BCE is useful for broad set membership, but averaging all negatives
    dilutes the few near-boundary mistakes that determine exact top-k recall.
    This listwise boundary term focuses each eligible row on its weakest teacher
    top-k score and strongest competing score.

    Rows need more than ``k`` visible blocks and ``k`` positive-mass teacher
    blocks. Returns ``(loss, eligible_rows)``.
    """
    _, _, _, kb = scores.shape
    kk = min(int(k), kb)
    if kk <= 0:
        return scores.sum() * 0.0, 0

    teacher_top = target.topk(kk, dim=-1).indices
    teacher_values = target.gather(-1, teacher_top)
    teacher_member = torch.zeros_like(target, dtype=torch.bool)
    teacher_member.scatter_(-1, teacher_top, True)

    visible = cmask[None, None]
    valid = target.sum(dim=-1) > row_valid_thresh
    has_choice = cmask.sum(dim=-1)[None, None] > kk
    has_full_positive_set = (teacher_values > 0).all(dim=-1)
    eligible = valid & has_choice & has_full_positive_set
    n = int(eligible.sum())
    if n == 0:
        return scores.sum() * 0.0, 0

    weakest_positive = scores.gather(-1, teacher_top).amin(dim=-1)
    neg_inf = torch.finfo(scores.dtype).min
    hardest_negative = scores.masked_fill(
        ~(visible & ~teacher_member), neg_inf).amax(dim=-1)
    loss = F.softplus(
        hardest_negative - weakest_positive + float(margin))
    return (loss * eligible).sum() / eligible.sum(), n


def needle_topk_loss(scores, target, cmask, needle_block, k=8,
                     margin=0.0, row_valid_thresh=0.5,
                     require_teacher_topk=True):
    """Encourage the reader row to retain a known needle in its top-k blocks.

    Only the final query block (the question/reader row) is supervised. With
    ``require_teacher_topk`` (legacy default) a group participates only when
    the teacher itself puts ``needle_block`` in its top-k; block-pooled teacher
    mass provably misses the evidence on natural cases (LOG 2026-07-26), so
    ``require_teacher_topk=False`` supervises every valid reader row instead —
    the needle block is ground truth from prompt construction, not from the
    teacher. The softplus term compares the needle score with the k-th best
    other causal block, a differentiable surrogate for student top-k
    membership.

    Returns ``(loss, eligible_layer_groups)``. Invalid needle metadata (or, in
    teacher mode, a teacher that never selects the needle) produces a zero,
    gradient-safe loss.
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
    valid = reader_target.sum(dim=-1) > row_valid_thresh
    if require_teacher_topk:
        teacher_top = reader_target.topk(min(int(k), kb), dim=-1).indices
        teacher_keeps_needle = (teacher_top == needle_block).any(dim=-1)
        eligible = teacher_keeps_needle & valid
    else:
        eligible = valid
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


def needle_union_topk_loss(scores, target, cmask, needle_block, k=8,
                           margin=0.0, row_valid_thresh=0.5,
                           require_teacher_topk=True):
    """Keep evidence in at least one KV group per teacher-eligible layer.

    ``selected_blocks`` routes independently per KV group, while retrieval only
    requires one group in a layer to carry the evidence forward. The existing
    per-group needle loss averages every eligible group. This union objective
    instead optimizes the best eligible group in each layer, matching the
    per-layer any-group preservation metric used by the natural retrieval eval.

    Returns ``(loss, eligible_layers)``.
    """
    L, _, qb, kb = scores.shape
    if needle_block is None or not 0 <= int(needle_block) < kb:
        return scores.sum() * 0.0, 0

    needle_block = int(needle_block)
    visible = cmask[-1]
    if not bool(visible[needle_block]):
        return scores.sum() * 0.0, 0

    competitors = int(visible.sum()) - 1
    kk = min(int(k), competitors)
    if kk <= 0:
        return scores.sum() * 0.0, 0

    reader_scores = scores[:, :, qb - 1, :]
    reader_target = target[:, :, qb - 1, :]
    valid = reader_target.sum(dim=-1) > row_valid_thresh
    if require_teacher_topk:
        teacher_top = reader_target.topk(min(int(k), kb), dim=-1).indices
        teacher_keeps_needle = (teacher_top == needle_block).any(dim=-1)
        eligible_groups = teacher_keeps_needle & valid
    else:
        # Evidence location is ground truth from prompt construction; do not
        # let a teacher blind spot (LOG 2026-07-26) silence the loss.
        eligible_groups = valid
    eligible_layers = eligible_groups.any(dim=1)
    n = int(eligible_layers.sum())
    if n == 0:
        return scores.sum() * 0.0, 0

    neg_inf = torch.finfo(scores.dtype).min
    other_scores = reader_scores.masked_fill(
        ~visible[None, None], neg_inf)
    other_scores[:, :, needle_block] = neg_inf
    threshold = other_scores.topk(kk, dim=-1).values[..., -1]
    gaps = threshold - reader_scores[:, :, needle_block]

    # Ineligible groups cannot satisfy the union. The minimum selects the
    # easiest teacher-supported route in each layer and sends gradient there.
    positive_inf = torch.finfo(scores.dtype).max
    best_gap = gaps.masked_fill(~eligible_groups, positive_inf).amin(dim=1)
    loss = F.softplus(best_gap + float(margin))
    return (loss * eligible_layers).sum() / eligible_layers.sum(), n


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

