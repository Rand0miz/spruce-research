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
from scripts.eval_tree_traversal import load_gate, traverse_to_leaf
from selector.targets import load_teacher
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

    device = selected_mask.device
    L, G, qb, kb = selected_mask.shape
    out = torch.full((1, L, G, qb, k_selected), PAD_VALUE,
                     dtype=torch.int32, device=device)

    for l in range(L):
        for g in range(G):
            for q in range(qb):
                ids = set()
                local_start = max(0, q - local_window)
                local_end = min(q, kb - 1)
                for block_id in range(local_start, local_end + 1):
                    ids.add(block_id)

                chosen = selected_mask[l, g, q].nonzero(as_tuple=False).flatten().tolist()
                for block_id in chosen:
                    block_id = int(block_id)
                    if block_id <= q:
                        ids.add(block_id)

                row = sorted(ids)
                if len(row) > k_selected:
                    local_ids = set(range(local_start, local_end + 1))
                    keep = sorted(local_ids)
                    for block_id in row:
                        if block_id not in local_ids:
                            keep.append(block_id)
                        if len(keep) == k_selected:
                            break
                    row = sorted(keep)

                out[0, l, g, q, :len(row)] = torch.tensor(
                    row, dtype=torch.int32, device=device)

    return out


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
        doc = load_teacher(path, device=args.device)
        meta = doc["meta"]
        if meta["num_layers"] != cfg["num_layers"] or meta["head_dim"] != cfg["head_dim"]:
            raise SystemExit(f"{path} shape {meta} mismatches gate config {cfg}")

        key_levels = build_key_tree(doc["k_feat"], radix=args.radix)
        mask = traverse_to_leaf(gate, doc["q_feat"], key_levels, args.beam)
        selected_blocks = selected_mask_to_blocks(
            mask, k_selected=k_selected, local_window=args.local_window).cpu()
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
            },
        }, out_path)
        print(f"saved {out_path}  shape={tuple(selected_blocks.shape)}")


if __name__ == "__main__":
    main()
