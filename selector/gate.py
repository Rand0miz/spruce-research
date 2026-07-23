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
            if torch.is_grad_enabled():
                out_l = checkpoint.checkpoint(
                    self._layer_scores, q_feat[l], k_feat[l], self.Wq[l], self.Wk[l],
                    use_reentrant=False,
                )
            else:
                # Checkpointing only saves training activation memory. During
                # inference it adds Python/autograd overhead without a benefit.
                out_l = self._layer_scores(
                    q_feat[l], k_feat[l], self.Wq[l], self.Wk[l])
            outs.append(out_l)
        scores = torch.stack(outs, dim=0)                     # [L,G,qb,kb]
        return scores / math.sqrt(self.proj_dim)

    @staticmethod
    def _layer_candidate_scores(q_feat_l, candidate_k_l, Wq_l, Wk_l):
        """Score only paired candidate keys for inference-time traversal.

        q_feat_l is [G,qb,P,d] and candidate_k_l is [G,qb,C,P,d].
        Unlike the flat scorer, key candidate C is specific to each query row.
        """
        qp = torch.einsum("gqid,dp->gqip", q_feat_l, Wq_l)
        kp = torch.einsum("gqcjd,dp->gqcjp", candidate_k_l, Wk_l)
        best = None
        for j in range(kp.shape[3]):
            scores = torch.einsum("gqip,gqcp->gqic", qp, kp[:, :, :, j])
            values = scores.amax(dim=2)
            best = values if best is None else torch.maximum(best, values)
        return best

    @torch.no_grad()
    def project_queries(self, q_feat):
        """Project [L,G,qb,P,d] queries once for every tree level."""
        return torch.einsum("lgqid,ldp->lgqip", q_feat, self.Wq)

    @torch.no_grad()
    def project_keys(self, k_feat):
        """Project [L,G,nodes,P,d] keys once before per-query gathering."""
        return torch.einsum("lgkjd,ldp->lgkjp", k_feat, self.Wk)

    @torch.no_grad()
    def score_projected_candidates(
            self, projected_q, projected_k, candidate_ids, layer_chunk=1):
        """Score already-projected per-query candidate node IDs.

        projected_q: [L,G,qb,P,p], projected_k: [L,G,nodes,P,p],
        candidate_ids: [L,G,qb,C]. Returns [L,G,qb,C]. Layer-wise
        gathering keeps the temporary candidate feature tensor bounded.
        """
        if (projected_q.dim() != 5 or projected_k.dim() != 5
                or candidate_ids.dim() != 4):
            raise ValueError("candidate scoring expects q/k 5-D and IDs 4-D")
        L, G, qb = projected_q.shape[:3]
        if (projected_k.shape[:2] != (L, G)
                or candidate_ids.shape[:3] != (L, G, qb)):
            raise ValueError("candidate scoring layer/group/query dimensions mismatch")

        if layer_chunk < 1:
            raise ValueError("layer_chunk must be >= 1")
        outputs = []
        nodes, P, projection_dim = projected_k.shape[2:]
        C = candidate_ids.shape[-1]
        for layer_start in range(0, L, layer_chunk):
            layer_end = min(layer_start + layer_chunk, L)
            chunk_layers = layer_end - layer_start
            ids = candidate_ids[layer_start:layer_end].clamp(
                min=0, max=nodes - 1)
            source = projected_k[layer_start:layer_end, :, None].expand(
                chunk_layers, G, qb, nodes, P, projection_dim)
            gather_ids = ids[..., None, None].expand(
                chunk_layers, G, qb, C, P, projection_dim)
            candidates = torch.gather(source, 3, gather_ids)
            # [Lc,G,qb,1,P,p] @ [Lc,G,qb,C,p,P]
            # -> [Lc,G,qb,C,P,P]. A small layer chunk amortizes Python/kernel
            # launch overhead while bounding the gathered candidate storage.
            pairwise = torch.matmul(
                projected_q[layer_start:layer_end, :, :, None],
                candidates.transpose(-1, -2),
            )
            outputs.append(pairwise.amax(dim=(-1, -2)))
        return torch.cat(outputs, dim=0) / math.sqrt(self.proj_dim)

    @torch.no_grad()
    def score_candidates(self, q_feat, k_feat, candidate_ids):
        """Convenience path that projects Q/K once before candidate scoring."""
        return self.score_projected_candidates(
            self.project_queries(q_feat), self.project_keys(k_feat), candidate_ids)
