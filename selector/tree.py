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


def _merge_level_prototypes(nodes, radix):
    """Vectorized parent construction for every node group in one level."""
    L, G, n, P, d = nodes.shape
    parents = (n + radix - 1) // radix
    padded_n = parents * radix
    if padded_n != n:
        nodes = torch.cat([
            nodes,
            nodes.new_zeros(L, G, padded_n - n, P, d),
        ], dim=2)

    grouped = nodes.view(L, G, parents, radix, P, d)
    valid_children = (
        torch.arange(padded_n, device=nodes.device).view(parents, radix) < n)
    valid_float = valid_children[None, None, :, :, None].to(nodes.dtype)
    counts = valid_children.sum(dim=1).clamp_min(1).to(nodes.dtype)
    parent_mean = (
        grouped[..., 0, :] * valid_float
    ).sum(dim=3) / counts[None, None, :, None]

    out = nodes.new_empty(L, G, parents, P, d)
    out[..., 0, :] = parent_mean
    if P == 1:
        return out

    candidates = grouped.reshape(L, G, parents, radix * P, d)
    candidate_valid = valid_children[:, :, None].expand(
        parents, radix, P).reshape(parents, radix * P)
    dist = (candidates - parent_mean[..., None, :]).pow(2).sum(dim=-1)
    dist = dist.masked_fill(
        ~candidate_valid[None, None], float("-inf"))
    idx = dist.topk(P - 1, dim=-1).indices
    gather_idx = idx[..., None].expand(-1, -1, -1, -1, d)
    out[..., 1:, :] = torch.gather(candidates, 3, gather_idx)
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
        n = current.shape[2]
        prev = levels[-1]
        parent_count = (n + radix - 1) // radix
        current = _merge_level_prototypes(current, radix)
        parent_ids = torch.arange(parent_count, device=device)
        starts = prev.starts[parent_ids * radix]
        ends = prev.ends[((parent_ids + 1) * radix - 1).clamp_max(n - 1)]
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


def tree_node_counts(num_leaf_nodes, radix=2, include_root=False):
    """Return leaf-first node counts for the fixed key tree."""
    _validate_radix(radix)
    if num_leaf_nodes < 1:
        raise ValueError("num_leaf_nodes must be >= 1")
    counts = []
    nodes = int(num_leaf_nodes)
    while nodes > 1:
        counts.append(nodes)
        nodes = (nodes + radix - 1) // radix
    if include_root:
        counts.append(1)
    return counts


def _parent_levels(key_level, target_level, radix=2, eps=1e-9):
    """Construct one parent key/target level from the current level."""
    _validate_radix(radix)
    if key_level.level != target_level.level:
        raise ValueError(
            f"level mismatch: key {key_level.level} vs target {target_level.level}")
    nodes = int(key_level.features.shape[2])
    if nodes != int(target_level.target.shape[-1]):
        raise ValueError("key and target node counts differ")
    if nodes <= 1:
        raise ValueError("the root has no parent")

    parent_count = (nodes + radix - 1) // radix
    parent_features = _merge_level_prototypes(key_level.features, radix)
    parent_ids = torch.arange(
        parent_count, device=key_level.starts.device)
    starts = key_level.starts[parent_ids * radix]
    ends = key_level.ends[
        ((parent_ids + 1) * radix - 1).clamp_max(nodes - 1)]

    padded_nodes = parent_count * radix
    target = target_level.target
    if padded_nodes != nodes:
        target = torch.cat([
            target,
            target.new_zeros(*target.shape[:-1], padded_nodes - nodes),
        ], dim=-1)
    parent_target = target.view(
        *target.shape[:-1], parent_count, radix).sum(dim=-1)
    parent_target = parent_target / parent_target.sum(
        dim=-1, keepdim=True).clamp_min(eps)
    qb = parent_target.shape[2]
    query = torch.arange(qb, device=starts.device)[:, None]
    cmask = starts[None, :] <= query
    level = key_level.level + 1
    return (
        KeyTreeLevel(level, parent_features, starts, ends),
        TargetTreeLevel(level, parent_target, cmask, starts, ends),
    )


def iter_tree_levels(k_feat, target, cmask, radix=2):
    """Yield aligned key/teacher levels without retaining the whole tree.

    Training can backpropagate one level at a time and keep peak activation
    memory near the flat leaf-level run. The materialized builders remain the
    evaluation path.
    """
    _validate_radix(radix)
    if k_feat.dim() != 5:
        raise ValueError(
            f"k_feat must be [L,G,kb,P,d], got {tuple(k_feat.shape)}")
    if target.dim() != 4:
        raise ValueError(
            f"target must be [L,G,qb,kb], got {tuple(target.shape)}")
    kb = int(k_feat.shape[2])
    if int(target.shape[-1]) != kb:
        raise ValueError("key features and target have different leaf counts")
    if cmask.shape != target.shape[-2:]:
        raise ValueError(
            f"cmask must be {tuple(target.shape[-2:])}, got {tuple(cmask.shape)}")

    starts = torch.arange(kb, device=k_feat.device, dtype=torch.long)
    ends = starts + 1
    key_level = KeyTreeLevel(0, k_feat, starts, ends)
    target_level = TargetTreeLevel(0, target, cmask, starts, ends)
    while True:
        yield key_level, target_level
        if key_level.features.shape[2] == 1:
            break
        key_level, target_level = _parent_levels(
            key_level, target_level, radix=radix)


def ancestor_node_id(leaf_block, starts, ends):
    """Return the unique node whose half-open range contains ``leaf_block``."""
    leaf_block = int(leaf_block)
    if leaf_block < 0:
        return -1
    matches = (starts <= leaf_block) & (leaf_block < ends)
    indices = matches.nonzero(as_tuple=False).flatten()
    if indices.numel() != 1:
        return -1
    return int(indices.item())


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
