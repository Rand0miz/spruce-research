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

    def forward(self, q_feat, k_feat):
        """q_feat [L,G,qb,P,d], k_feat [L,G,kb,P,d] -> raw MaxSim scores [L,G,qb,kb].
        score(q,k) = max over (query-proto i, key-proto j) of (Wq.q_i).(Wk.k_j).
        Running max over key-prototypes so [L,G,qb,kb,P,P] is never materialized."""
        qp = torch.einsum("lgqid,ldp->lgqip", q_feat, self.Wq)   # [L,G,qb,P,p]
        kp = torch.einsum("lgkjd,ldp->lgkjp", k_feat, self.Wk)   # [L,G,kb,P,p]
        best = None
        for j in range(kp.shape[3]):                             # loop key-protos
            kpj = kp[:, :, :, j, :]                              # [L,G,kb,p]
            s = torch.einsum("lgqip,lgkp->lgqik", qp, kpj)       # [L,G,qb,P,kb]
            s = s.amax(dim=3)                                    # max over query-protos
            best = s if best is None else torch.maximum(best, s)
        return best / math.sqrt(self.proj_dim)                   # [L,G,qb,kb]
