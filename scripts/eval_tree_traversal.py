"""Evaluate top-down tree traversal for a trained selector gate.

Unlike scripts/eval_tree.py, this simulates inference-time pruning: start at the
root, expand only selected parent nodes, keep a fixed-width beam at each level,
and compare the final selected leaf blocks against the dense teacher targets.

Usage:
  python -m scripts.eval_tree_traversal --gate selector_ckpt/flat_gate.pt \
      --targets teacher_targets/heldout_*.pt --beams 1 2 4 8 16
"""
import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from selector.gate import FlatGate
from selector.targets import load_teacher
from selector.tree import build_key_tree


def load_gate(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    gate = FlatGate(cfg["num_layers"], cfg["head_dim"], cfg["proj_dim"]).to(device)
    gate.load_state_dict(ckpt["state_dict"])
    gate.eval()
    return gate, cfg


def expand_paths(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    return paths


def child_parent_index(parent_level, child_level):
    """Map each child node to its parent node in the next coarser level."""
    parent_for_child = []
    for cs, ce in zip(child_level.starts.tolist(), child_level.ends.tolist()):
        matches = ((parent_level.starts <= cs) & (ce <= parent_level.ends)).nonzero()
        if matches.numel() != 1:
            raise ValueError(
                f"child range [{cs}, {ce}) matched {matches.numel()} parents")
        parent_for_child.append(int(matches[0]))
    return torch.tensor(parent_for_child, device=child_level.starts.device)


@torch.no_grad()
def traverse_to_leaf(gate, q_feat, key_levels, beam):
    """Return final selected leaf mask [L,G,qb,kb] from root-to-leaf traversal."""
    if beam < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")

    device = q_feat.device
    L, G, qb = q_feat.shape[:3]
    root_level = key_levels[-1]
    root_nodes = root_level.features.shape[2]

    selected = torch.ones((L, G, qb, root_nodes), dtype=torch.bool, device=device)
    q = torch.arange(qb, device=device)

    for child_level_idx in range(len(key_levels) - 2, -1, -1):
        child_level = key_levels[child_level_idx]
        parent_level = key_levels[child_level_idx + 1]
        parent_for_child = child_parent_index(parent_level, child_level)

        parent_selected = selected.index_select(-1, parent_for_child)
        causal = child_level.starts[None, :] <= q[:, None]
        candidates = parent_selected & causal[None, None]

        scores = gate(q_feat, child_level.features)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~candidates, neg_inf)

        k = min(beam, child_level.features.shape[2])
        top = masked.topk(k, dim=-1).indices
        selected = torch.zeros_like(candidates)
        selected.scatter_(-1, top, True)
        selected &= candidates

    return selected


@torch.no_grad()
def leaf_metrics(selected, target, cmask, budgets, selection_budget, needle_block=None):
    """Score selected leaf blocks against teacher leaf targets."""
    L, G, qb, kb = target.shape
    valid = target.sum(dim=-1) > 0.5
    q_idx = torch.arange(qb, device=target.device)
    selected = selected & cmask[None, None]
    out = {}

    selected_count = selected.sum(dim=-1).float()
    valid_rows = valid.sum().clamp_min(1)
    out["avg_selected"] = float((selected_count * valid).sum() / valid_rows)

    kk = min(selection_budget, kb)
    enough_sel = (q_idx + 1 >= kk)[None, None].expand(L, G, qb)
    m_sel = valid & enough_sel
    sel_denom = m_sel.sum().clamp_min(1)

    coverage = (target * selected).sum(dim=-1)
    out["coverage"] = float((coverage * m_sel).sum() / sel_denom)

    t_sel = target.topk(kk, dim=-1).indices
    oracle = torch.zeros_like(target, dtype=torch.bool)
    oracle.scatter_(-1, t_sel, True)
    oracle_cov = (target * oracle).sum(dim=-1)
    out["oracle_coverage"] = float((oracle_cov * m_sel).sum() / sel_denom)

    for r in budgets:
        rr = min(r, kb)
        t_top = target.topk(rr, dim=-1).indices
        overlap = selected.gather(-1, t_top).sum(dim=-1).float()
        enough = (q_idx + 1 >= rr)[None, None].expand(L, G, qb)
        m = valid & enough
        denom = m.sum().clamp_min(1)
        out[f"recall@{r}"] = float((overlap / rr * m).sum() / denom)

    if needle_block is not None and 0 <= needle_block < kb:
        reader = selected[:, :, -1, :]
        hit = reader[..., needle_block].float()
        out["needle_hit"] = float(hit.mean())

    return out


def format_budget_metrics(metrics, budgets, prefix):
    return "  ".join(f"{prefix}@{b}={metrics[f'{prefix}@{b}']:.3f}" for b in budgets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "selector_ckpt", "flat_gate.pt"))
    ap.add_argument("--targets", nargs="+", required=True,
                    help="HELD-OUT teacher .pt paths or globs")
    ap.add_argument("--radix", type=int, default=2)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="number of nodes kept after each tree level expansion")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="teacher top-k budgets to measure against final leaves")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="worst traversal recall@8 pass bar")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = expand_paths(args.targets)
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    gate, cfg = load_gate(args.gate, args.device)
    print(f"loaded gate {args.gate}  L={cfg['num_layers']} d={cfg['head_dim']} "
          f"proj={cfg['proj_dim']}  radix={args.radix}")
    print(f"beams={args.beams}  budgets={args.budgets}")

    sums = {beam: {} for beam in args.beams}
    counts = {beam: 0 for beam in args.beams}
    worst_r8 = {beam: 1.0 for beam in args.beams}

    with torch.no_grad():
        for p in paths:
            doc = load_teacher(p, device=args.device)
            m = doc["meta"]
            if m["num_layers"] != cfg["num_layers"] or m["head_dim"] != cfg["head_dim"]:
                raise SystemExit(f"{p} shape {m} mismatches gate config {cfg}")

            key_levels = build_key_tree(doc["k_feat"], radix=args.radix)
            print(f"\n{os.path.basename(p)}  seq={m['seq_len']} qb={m['qb']} "
                  f"kb={m['kb']} levels={len(key_levels)} needle_blk={m['needle_block']}")

            for beam in args.beams:
                selected = traverse_to_leaf(gate, doc["q_feat"], key_levels, beam)
                met = leaf_metrics(
                    selected, doc["target"], doc["cmask"],
                    tuple(args.budgets), beam, m["needle_block"])
                r8 = met.get("recall@8", float("nan"))
                worst_r8[beam] = min(worst_r8[beam], r8)

                counts[beam] += 1
                for name, val in met.items():
                    sums[beam][name] = sums[beam].get(name, 0.0) + float(val)

                rec = format_budget_metrics(met, args.budgets, "recall")
                ndl = (f"  ndl={met['needle_hit']:.2f}"
                       if "needle_hit" in met else "")
                print(f"  beam={beam:<3} selected={met['avg_selected']:.1f}  {rec}")
                print(f"           cov={met['coverage']:.3f}/{met['oracle_coverage']:.3f}{ndl}")

    if len(paths) > 1:
        print("\nmean over documents")
        for beam in args.beams:
            n = counts[beam]
            avg = {name: val / n for name, val in sums[beam].items()}
            rec = format_budget_metrics(avg, args.budgets, "recall")
            ndl = f"  ndl={avg['needle_hit']:.2f}" if "needle_hit" in avg else ""
            print(f"  beam={beam:<3} selected={avg['avg_selected']:.1f}  {rec}")
            print(f"           cov={avg['coverage']:.3f}/{avg['oracle_coverage']:.3f}{ndl}")

    print("\nworst traversal recall@8")
    ok_any = False
    for beam in args.beams:
        ok = worst_r8[beam] >= args.threshold
        ok_any = ok_any or ok
        print(f"  beam={beam:<3} {worst_r8[beam]:.3f}  "
              f"{'PASS' if ok else 'FAIL'} (bar {args.threshold})")
    sys.exit(0 if ok_any else 1)


if __name__ == "__main__":
    main()