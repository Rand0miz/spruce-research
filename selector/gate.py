"""Flat selector gate — SeerAttention form, the leaf level of the tree.

score(q, k) = max over (query-proto i, key-proto j) of (Wq . q_i) . (Wk . k_j) / sqrt(d_proj)

Wq, Wk are the ONLY trainable parameters. One (Wq, Wk) pair per layer (SeerAttention
trains a per-layer gate); shared across the G kv-groups within a layer, since q_feat /
k_feat already carry the per-group signal. Raw scores are returned; causal masking +
softmax live in the loss / recall so this module stays a pure scorer.
"""
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class FlatGate(nn.Module):
    def __init__(self, num_layers, head_dim, proj_dim=None):
        super().__init__()
        d = head_dim
        p = proj_dim or head_dim
        self.proj_dim = p
        # [L, d, p] per-layer projections. Init like nn.Linear(bias=False).
        self.Wq = nn.Parameter(torch.empty(num_layers, d, p))
        self.Wk = nn.Parameter(torch.empty(num_layers, d, p))
        for W in (self.Wq, self.Wk):
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    @staticmethod
    def _layer_scores(q_feat_l, k_feat_l, Wq_l, Wk_l):
        """Single layer: q_feat_l [G,qb,P,d], k_feat_l [G,kb,P,d], Wq_l/Wk_l [d,p]
        -> unscaled MaxSim scores [G,qb,kb].

        Never materializes [G,qb,kb,P,P] or [G,qb,kb,p]-for-all-layers-at-once.
        Step 1 (no_grad): find the winning (query-proto i, key-proto j) index pair
        per (q,k) entry via the same running-max loop as a plain forward -- this is
        just index bookkeeping, nothing here is saved for backward. Step 2: gather
        the winning projected query/key vectors and take ONE differentiable dot
        product. The gradient of a max is the subgradient routed to the argmax
        element, so this reproduces the exact gradient of the full pairwise max
        while only ever retaining per-(q,k) *single* p-vectors, not P*P of them.

        Called once per layer through torch.utils.checkpoint (see forward()), so
        the [G,qb,kb,p] gather tensors for one layer are freed before the next
        layer starts -- only the (small) inputs of each layer are retained across
        the whole stack, not every layer's intermediate simultaneously.
        """
        qp = torch.einsum("gqid,dp->gqip", q_feat_l, Wq_l)   # [G,qb,P,p]
        kp = torch.einsum("gkjd,dp->gkjp", k_feat_l, Wk_l)   # [G,kb,P,p]
        G, qb, P, p = qp.shape
        kb = kp.shape[1]

        with torch.no_grad():
            qp_d, kp_d = qp.detach(), kp.detach()
            best_val = best_i = best_j = None
            for j in range(P):                                # loop key-protos
                kpj = kp_d[:, :, j, :]                         # [G,kb,p]
                s = torch.einsum("gqip,gkp->gqik", qp_d, kpj)  # [G,qb,P,kb]
                val, idx_i = s.max(dim=2)                      # max over query-protos -> [G,qb,kb]
                if best_val is None:
                    best_val, best_i = val, idx_i
                    best_j = torch.full_like(idx_i, j)
                else:
                    better = val > best_val
                    best_val = torch.where(better, val, best_val)
                    best_i = torch.where(better, idx_i, best_i)
                    best_j = torch.where(better, torch.full_like(idx_i, j), best_j)

        # differentiable gather: winning query-proto vector per (q,k), via a view
        # (no allocation) expanded over kb, then gather along the P axis only.
        qp_exp = qp.unsqueeze(2).expand(-1, -1, kb, -1, -1)          # [G,qb,kb,P,p] view
        qi_idx = best_i[:, :, :, None, None].expand(-1, -1, -1, 1, p)  # [G,qb,kb,1,p]
        gq = torch.gather(qp_exp, 3, qi_idx).squeeze(3)               # [G,qb,kb,p]

        kp_exp = kp.unsqueeze(1).expand(-1, qb, -1, -1, -1)          # [G,qb,kb,P,p] view
        kj_idx = best_j[:, :, :, None, None].expand(-1, -1, -1, 1, p)  # [G,qb,kb,1,p]
        gk = torch.gather(kp_exp, 3, kj_idx).squeeze(3)               # [G,qb,kb,p]

        return (gq * gk).sum(dim=-1)                                  # [G,qb,kb]

    def forward(self, q_feat, k_feat):
        """q_feat [L,G,qb,P,d], k_feat [L,G,kb,P,d] -> raw MaxSim scores [L,G,qb,kb].
        score(q,k) = max over (query-proto i, key-proto j) of (Wq.q_i).(Wk.k_j).
        Chunked over the layer axis via torch.utils.checkpoint so the per-layer
        gather tensors ([G,qb,kb,p], see _layer_scores) never coexist for all L
        layers at once in the autograd graph -- see _layer_scores docstring."""
        L = q_feat.shape[0]
        outs = []
        for l in range(L):
            out_l = checkpoint.checkpoint(
                self._layer_scores, q_feat[l], k_feat[l], self.Wq[l], self.Wk[l],
                use_reentrant=False,
            )
            outs.append(out_l)
        scores = torch.stack(outs, dim=0)                     # [L,G,qb,kb]
        return scores / math.sqrt(self.proj_dim)
