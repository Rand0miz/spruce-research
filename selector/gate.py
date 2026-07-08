"""Flat selector gate — SeerAttention form, the leaf level of the tree.

score(q, k) = (Wq . pooledQ[q]) . (Wk . pooledK[k])^T / sqrt(d_proj)

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
        """q_feat [L,G,qb,d], k_feat [L,G,kb,d] -> raw scores [L,G,qb,kb]."""
        qp = torch.einsum("lgqd,ldp->lgqp", q_feat, self.Wq)
        kp = torch.einsum("lgkd,ldp->lgkp", k_feat, self.Wk)
        return torch.einsum("lgqp,lgkp->lgqk", qp, kp) / math.sqrt(self.proj_dim)
