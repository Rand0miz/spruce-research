"""Export top-down traversal selections to the frozen selected_blocks interface.

Input: teacher target .pt files containing pooledQ/pooledK features.
Output: one .pt file per input with:
  selected_blocks: int32 [batch=1, layer, kv_head_group, query_block, K]
  meta: source path, beam/radix/local-window settings, model shape

This is Stage 3 glue: it turns the learned tree traversal into the exact tensor
shape consumed by sparse attention code.
"""
import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from interfaces.validator import PAD_VALUE, validate_selected_blocks
from scripts.eval_tree_traversal import load_gate, traverse_to_leaf_ids
from selector.targets import load_selector_features
from selector.tree import build_key_tree


def expand_paths(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    return paths


def safe_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


@torch.no_grad()
def selected_mask_to_blocks(selected_mask, k_selected, local_window=1):
    """Convert [L,G,qb,kb] bool mask to selected_blocks [1,L,G,qb,K].

    Local blocks q-local_window..q are forced first, then traversal-selected blocks
    are added until the fixed row width is full. Final rows are sorted increasing
    and padded with PAD_VALUE.
    """
    if selected_mask.dim() != 4:
        raise ValueError(f"selected_mask must be [L,G,qb,kb], got {tuple(selected_mask.shape)}")
    if k_selected < 1:
        raise ValueError(f"k_selected must be >= 1, got {k_selected}")
    if local_window < 0:
        raise ValueError(f"local_window must be >= 0, got {local_window}")
    if k_selected < local_window + 1:
        raise ValueError(
            f"k_selected={k_selected} cannot fit local window size {local_window + 1}")

    device = selected_mask.device
    _, _, qb, kb = selected_mask.shape
    key_ids = torch.arange(kb, device=device)
    query_ids = torch.arange(qb, device=device)[:, None]
    causal = key_ids[None, :] <= query_ids
    local = causal & (key_ids[None, :] >= query_ids - local_window)
    local = local[None, None]

    # Preserve the previous policy exactly: forced local IDs have priority;
    # if the union exceeds K, retain the smallest traversal-selected nonlocal
    # IDs that fit. All work remains vectorized on the source device.
    nonlocal_selected = selected_mask.bool() & causal[None, None] & ~local
    local_count = local.sum(dim=-1)
    remaining = k_selected - local_count
    nonlocal_rank = nonlocal_selected.cumsum(dim=-1)
    keep_nonlocal = nonlocal_selected & (nonlocal_rank <= remaining[..., None])
    keep = local | keep_nonlocal

    sentinel = torch.full_like(key_ids, kb)
    packed = torch.where(keep, key_ids, sentinel).sort(dim=-1).values[..., :k_selected]
    packed = packed.masked_fill(packed == kb, PAD_VALUE).to(torch.int32)
    return packed.unsqueeze(0)


@torch.no_grad()
def selected_ids_to_blocks(
        selected_ids, k_selected, local_window=1, key_blocks=None):
    """Pack compact traversal IDs directly into ``selected_blocks``.

    This preserves ``selected_mask_to_blocks`` policy exactly: forced local
    blocks take priority, then the smallest causal nonlocal traversal IDs fill
    the remaining slots. Work scales with the beam width rather than all key
    blocks.
    """
    if selected_ids.dim() != 4:
        raise ValueError(
            f"selected_ids must be [L,G,qb,C], got {tuple(selected_ids.shape)}")
    if k_selected < 1:
        raise ValueError(f"k_selected must be >= 1, got {k_selected}")
    if local_window < 0:
        raise ValueError(f"local_window must be >= 0, got {local_window}")
    if k_selected < local_window + 1:
        raise ValueError(
            f"k_selected={k_selected} cannot fit local window size {local_window + 1}")

    device = selected_ids.device
    qb = selected_ids.shape[2]
    kb = qb if key_blocks is None else int(key_blocks)
    if kb < 1:
        raise ValueError(f"key_blocks must be >= 1, got {kb}")

    ids = selected_ids.to(torch.long)
    query = torch.arange(qb, device=device, dtype=torch.long)
    q = query[None, None, :, None]
    sentinel = max(qb, kb)
    nonlocal_valid = (
        (ids >= 0) & (ids < kb) & (ids <= q)
        & (ids < q - local_window)
    )
    nonlocal_ids = torch.where(
        nonlocal_valid, ids, torch.full_like(ids, sentinel))
    nonlocal_ids = nonlocal_ids.sort(dim=-1).values
    unique = nonlocal_ids != sentinel
    if nonlocal_ids.shape[-1] > 1:
        unique[..., 1:] &= (
            nonlocal_ids[..., 1:] != nonlocal_ids[..., :-1])

    local_count = (query + 1).clamp(max=local_window + 1)
    remaining = k_selected - local_count
    unique_rank = unique.cumsum(dim=-1)
    keep_nonlocal = unique & (
        unique_rank <= remaining[None, None, :, None])
    nonlocal_ids = torch.where(
        keep_nonlocal, nonlocal_ids,
        torch.full_like(nonlocal_ids, sentinel))

    # Construct local IDs in increasing order, padding nonexistent negative
    # IDs with the same sortable sentinel.
    local_offsets = torch.arange(
        local_window, -1, -1, device=device, dtype=torch.long)
    local_ids = query[:, None] - local_offsets[None, :]
    local_ids = torch.where(
        local_ids >= 0, local_ids,
        torch.full_like(local_ids, sentinel))
    local_ids = local_ids[None, None].expand(
        selected_ids.shape[0], selected_ids.shape[1], -1, -1)

    combined = torch.cat([nonlocal_ids, local_ids], dim=-1)
    packed = combined.sort(dim=-1).values[..., :k_selected]
    packed = packed.masked_fill(packed == sentinel, PAD_VALUE).to(torch.int32)
    return packed.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--targets", nargs="+", required=True,
                    help="teacher .pt paths or globs with pooledQ/pooledK")
    ap.add_argument("--out-dir", default="selected_blocks")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--radix", type=int, default=2)
    ap.add_argument("--k-selected", type=int, default=None,
                    help="fixed selected_blocks width; default beam + local_window + 1")
    ap.add_argument("--local-window", type=int, default=1,
                    help="force blocks q-local_window..q into every causal row")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = expand_paths(args.targets)
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    k_selected = args.k_selected
    if k_selected is None:
        k_selected = args.beam + args.local_window + 1
    if k_selected < args.local_window + 1:
        raise SystemExit(
            f"--k-selected {k_selected} cannot fit local window size {args.local_window + 1}")

    os.makedirs(args.out_dir, exist_ok=True)
    gate, cfg = load_gate(args.gate, args.device)
    print(f"loaded gate {args.gate}  L={cfg['num_layers']} d={cfg['head_dim']} "
          f"proj={cfg['proj_dim']}  beam={args.beam} K={k_selected}")

    for path in paths:
        doc = load_selector_features(path, device=args.device)
        meta = doc["meta"]
        if meta["num_layers"] != cfg["num_layers"] or meta["head_dim"] != cfg["head_dim"]:
            raise SystemExit(f"{path} shape {meta} mismatches gate config {cfg}")

        key_levels = build_key_tree(doc["k_feat"], radix=args.radix)
        selected_ids = traverse_to_leaf_ids(
            gate, doc["q_feat"], key_levels, args.beam, radix=args.radix)
        selected_blocks = selected_ids_to_blocks(
            selected_ids, k_selected=k_selected,
            local_window=args.local_window,
            key_blocks=meta["kb"]).cpu()
        validate_selected_blocks(selected_blocks)

        out_path = os.path.join(args.out_dir, f"{safe_stem(path)}_beam{args.beam}_K{k_selected}.pt")
        torch.save({
            "selected_blocks": selected_blocks,
            "meta": {
                "source": path,
                "gate": args.gate,
                "beam": args.beam,
                "radix": args.radix,
                "k_selected": k_selected,
                "local_window": args.local_window,
                "seq_len": meta["seq_len"],
                "block_size": meta["block_size"],
                "num_layers": meta["num_layers"],
                "num_groups": meta["num_groups"],
                "query_blocks": meta["qb"],
                "key_blocks": meta["kb"],
                "rope": meta.get("rope"),
            },
        }, out_path)
        print(f"saved {out_path}  shape={tuple(selected_blocks.shape)}")


if __name__ == "__main__":
    main()
