"""Tree utilities for the learned selector.

The deployed selector descends a key-axis tree for each leaf query block. Training
can still use full supervision: score every key node at every level, then apply the
same forward-KL loss used by the flat leaf gate.

This module deliberately keeps query blocks at leaf resolution. The tree is over
key blocks only, matching the eventual selected_blocks[..., query_block, :]
interface.
"""
from dataclasses import dataclass

import torch

from selector.loss import kl_loss


@dataclass(frozen=True)
class KeyTreeLevel:
    """Key-node features for one tree level.

    features: [L, G, num_nodes, P, d]
    starts/ends: [num_nodes] leaf-block half-open intervals [start, end)
    """
    level: int
    features: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor


@dataclass(frozen=True)
class TargetTreeLevel:
    """Teacher distribution and causal mask for one key-tree level.

    target: [L, G, qb, num_nodes], row-normalized over visible key nodes
    cmask: [qb, num_nodes], true when the node has at least one causal leaf block
    """
    level: int
    target: torch.Tensor
    cmask: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor


def _validate_radix(radix):
    if radix < 2:
        raise ValueError(f"radix must be >= 2, got {radix}")


def _merge_node_prototypes(nodes):
    """Merge child node prototypes and cap back to P.

    nodes is [L, G, child_count, P, d]. Proto 0 of the parent is the mean of child
    means; remaining slots are the child prototypes farthest from that parent mean.
    This is an approximation because extraction did not save raw tokens for parent
    nodes, but it preserves bounded-P MaxSim scoring through the tree.
    """
    L, G, _, P, d = nodes.shape
    parent_mean = nodes[..., 0, :].mean(dim=2)              # [L, G, d]
    candidates = nodes.reshape(L, G, -1, d)                 # [L, G, child_count*P, d]
    out = nodes.new_empty(L, G, P, d)
    out[:, :, 0, :] = parent_mean

    if P == 1:
        return out

    dist = (candidates - parent_mean[:, :, None, :]).pow(2).sum(dim=-1)
    take = min(P - 1, candidates.shape[2])
    idx = dist.topk(take, dim=-1).indices                   # [L, G, take]
    gather_idx = idx[..., None].expand(-1, -1, -1, d)
    picked = torch.gather(candidates, 2, gather_idx)         # [L, G, take, d]
    out[:, :, 1:1 + take, :] = picked

    if take < P - 1:
        out[:, :, 1 + take:, :] = parent_mean[:, :, None, :]
    return out


def build_key_tree(k_feat, radix=2):
    """Build key-node feature levels from leaf key features.

    k_feat is [L, G, kb, P, d]. Returns levels leaf-first. Odd tails are carried
    as smaller final parent nodes so every leaf block remains covered.
    """
    _validate_radix(radix)
    if k_feat.dim() != 5:
        raise ValueError(f"k_feat must be [L,G,kb,P,d], got {tuple(k_feat.shape)}")

    device = k_feat.device
    kb = k_feat.shape[2]
    starts = torch.arange(kb, device=device, dtype=torch.long)
    ends = starts + 1
    levels = [KeyTreeLevel(0, k_feat, starts, ends)]

    current = k_feat
    level = 0
    while current.shape[2] > 1:
        level += 1
        parent_feats = []
        parent_starts = []
        parent_ends = []
        n = current.shape[2]
        prev = levels[-1]
        for start in range(0, n, radix):
            end = min(start + radix, n)
            parent_feats.append(_merge_node_prototypes(current[:, :, start:end]))
            parent_starts.append(prev.starts[start])
            parent_ends.append(prev.ends[end - 1])
        current = torch.stack(parent_feats, dim=2)           # [L, G, parent_nodes, P, d]
        starts = torch.stack(parent_starts)
        ends = torch.stack(parent_ends)
        levels.append(KeyTreeLevel(level, current, starts, ends))

    return levels


def build_target_tree(target, row_mass=None, radix=2, eps=1e-9):
    """Build per-level teacher targets for the key-axis tree.

    target is [L, G, qb, kb], already causal and row-normalized at the leaf level.
    row_mass can be [L, G, qb] from load_teacher; when provided, raw mass is
    reconstructed before each level is normalized. With key-axis-only aggregation,
    this is equivalent to summing normalized rows, but it keeps the intended
    teacher-mass semantics explicit.
    """
    _validate_radix(radix)
    if target.dim() != 4:
        raise ValueError(f"target must be [L,G,qb,kb], got {tuple(target.shape)}")

    L, G, qb, kb = target.shape
    if row_mass is None:
        mass = target
    else:
        if row_mass.shape != (L, G, qb):
            raise ValueError(
                f"row_mass must be {(L, G, qb)}, got {tuple(row_mass.shape)}")
        mass = target * row_mass[..., None]

    device = target.device
    starts = torch.arange(kb, device=device, dtype=torch.long)
    ends = starts + 1
    levels = []
    level = 0

    while True:
        node_mass = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            node_mass.append(mass[..., start:end].sum(dim=-1))
        node_mass = torch.stack(node_mass, dim=-1)           # [L, G, qb, num_nodes]
        row = node_mass.sum(dim=-1, keepdim=True)
        target_l = node_mass / row.clamp_min(eps)

        q = torch.arange(qb, device=device)[:, None]
        cmask = starts[None, :] <= q                         # node has causal content
        levels.append(TargetTreeLevel(level, target_l, cmask, starts, ends))

        if starts.numel() == 1:
            break

        parent_starts = []
        parent_ends = []
        for i in range(0, starts.numel(), radix):
            j = min(i + radix, starts.numel())
            parent_starts.append(starts[i])
            parent_ends.append(ends[j - 1])
        starts = torch.stack(parent_starts)
        ends = torch.stack(parent_ends)
        level += 1

    return levels


def tree_kl_loss(gate, q_feat, key_levels, target_levels, level_weights=None):
    """Sum flat-gate KL over all key-tree levels.

    Returns (loss, stats). stats contains per-level KL and valid-row counts.
    """
    if len(key_levels) != len(target_levels):
        raise ValueError(
            f"key/target level count mismatch: {len(key_levels)} vs {len(target_levels)}")
    if level_weights is None:
        level_weights = [1.0] * len(key_levels)
    if len(level_weights) != len(key_levels):
        raise ValueError("level_weights must match number of key levels")

    total = None
    stats = []
    for weight, key_level, target_level in zip(level_weights, key_levels, target_levels):
        if key_level.level != target_level.level:
            raise ValueError(
                f"level mismatch: key {key_level.level}, target {target_level.level}")
        scores = gate(q_feat, key_level.features)
        loss, n_valid = kl_loss(scores, target_level.target, target_level.cmask)
        weighted = loss * float(weight)
        total = weighted if total is None else total + weighted
        stats.append({
            "level": key_level.level,
            "nodes": int(key_level.features.shape[2]),
            "kl": float(loss.detach()),
            "n_valid": int(n_valid),
            "weight": float(weight),
        })

    return total, stats
