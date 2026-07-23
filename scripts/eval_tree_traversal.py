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
from selector.plotting import save_tree_plot
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
    # Parent ranges are sorted, non-overlapping, and use exclusive ends.
    # searchsorted avoids per-node .tolist() calls and GPU/CPU synchronization.
    parent_for_child = torch.searchsorted(
        parent_level.ends, child_level.starts, right=True)
    if torch.any(parent_for_child >= parent_level.ends.numel()):
        raise ValueError("a child range falls outside all parent ranges")
    parent_starts = parent_level.starts.index_select(0, parent_for_child)
    parent_ends = parent_level.ends.index_select(0, parent_for_child)
    valid = ((parent_starts <= child_level.starts)
             & (child_level.ends <= parent_ends))
    if not bool(valid.all()):
        raise ValueError("a child range does not fit exactly one parent range")
    return parent_for_child


@torch.no_grad()
def traverse_to_leaf_ids(
        gate, q_feat, key_levels, beam, radix=2, layer_chunk=1):
    """Return compact selected leaf IDs ``[L,G,qb,C]``.

    Invalid tail slots are ``-1``. Keeping IDs compact avoids materializing the
    quadratic-in-block-count leaf mask on the deployed inference path.
    """
    if beam < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")
    if radix < 2:
        raise ValueError(f"radix must be >= 2, got {radix}")
    if layer_chunk < 1:
        raise ValueError(f"layer_chunk must be >= 1, got {layer_chunk}")

    device = q_feat.device
    L, G, qb = q_feat.shape[:3]
    root_level = key_levels[-1]
    if root_level.features.shape[2] != 1:
        raise ValueError("key tree must end in exactly one root node")

    # Fixed-width IDs avoid materializing a [L,G,qb,nodes] mask at each level.
    selected_ids = torch.zeros((L, G, qb, 1), dtype=torch.long, device=device)
    query_ids = torch.arange(qb, device=device)
    child_offsets = torch.arange(radix, device=device)
    projected_q = gate.project_queries(q_feat)

    for child_level_idx in range(len(key_levels) - 2, -1, -1):
        child_level = key_levels[child_level_idx]
        child_count = child_level.features.shape[2]
        candidates = (
            selected_ids[..., None] * radix + child_offsets
        ).flatten(start_dim=-2)
        valid_parent = (selected_ids[..., None] >= 0).expand(
            *selected_ids.shape, radix).flatten(start_dim=-2)
        valid = valid_parent & (candidates < child_count)
        safe_candidates = candidates.clamp(min=0, max=child_count - 1)
        candidate_starts = child_level.starts.index_select(
            0, safe_candidates.reshape(-1)).reshape_as(safe_candidates)
        valid = valid & (candidate_starts <= query_ids[None, None, :, None])

        projected_k = gate.project_keys(child_level.features)
        scores = gate.score_projected_candidates(
            projected_q, projected_k, safe_candidates,
            layer_chunk=layer_chunk)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~valid, neg_inf)

        k = min(beam, candidates.shape[-1], child_count)
        top = masked.topk(k, dim=-1).indices
        selected_ids = candidates.gather(-1, top)
        selected_valid = valid.gather(-1, top)
        selected_ids = selected_ids.masked_fill(~selected_valid, -1)

    return selected_ids


@torch.no_grad()
def leaf_ids_to_mask(selected_ids, leaf_count):
    """Convert compact ``[L,G,qb,C]`` IDs to an evaluation-only leaf mask."""
    if selected_ids.dim() != 4:
        raise ValueError(
            f"selected_ids must be [L,G,qb,C], got {tuple(selected_ids.shape)}")
    if leaf_count < 1:
        raise ValueError(f"leaf_count must be >= 1, got {leaf_count}")
    L, G, qb = selected_ids.shape[:3]
    device = selected_ids.device
    safe_ids = selected_ids.clamp(min=0, max=leaf_count)
    safe_ids = safe_ids.masked_fill(selected_ids < 0, leaf_count)
    with_pad = torch.zeros(
        (L, G, qb, leaf_count + 1), dtype=torch.bool, device=device)
    with_pad.scatter_(-1, safe_ids, True)
    return with_pad[..., :leaf_count]


@torch.no_grad()
def traverse_to_leaf(gate, q_feat, key_levels, beam, radix=2, layer_chunk=1):
    """Return evaluation mask ``[L,G,qb,kb]`` using compact traversal."""
    selected_ids = traverse_to_leaf_ids(
        gate, q_feat, key_levels, beam, radix=radix,
        layer_chunk=layer_chunk)
    return leaf_ids_to_mask(
        selected_ids, key_levels[0].features.shape[2])


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
    ap.add_argument("--plot", default=None,
                    help="PNG output path; default is beside --gate")
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
                selected = traverse_to_leaf(
                    gate, doc["q_feat"], key_levels, beam, radix=args.radix)
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

    averages = {
        beam: {name: value / counts[beam] for name, value in sums[beam].items()}
        for beam in args.beams
    }
    if len(paths) > 1:
        print("\nmean over documents")
        for beam in args.beams:
            n = counts[beam]
            avg = averages[beam]
            rec = format_budget_metrics(avg, args.budgets, "recall")
            ndl = f"  ndl={avg['needle_hit']:.2f}" if "needle_hit" in avg else ""
            print(f"  beam={beam:<3} selected={avg['avg_selected']:.1f}  {rec}")
            print(f"           cov={avg['coverage']:.3f}/{avg['oracle_coverage']:.3f}{ndl}")

    model_name = os.path.splitext(os.path.basename(args.gate))[0]
    plot_dir = os.path.join(os.path.dirname(args.gate) or ".", "eval_graphs", model_name)
    plot_path = args.plot or os.path.join(plot_dir, "traversal_eval.png")
    if save_tree_plot([averages[beam] for beam in args.beams], args.beams, args.budgets,
                      plot_path, "Top-down traversal evaluation", "Beam width"):
        print(f"evaluation graph -> {plot_path}")
    else:
        print("matplotlib not installed; skipped evaluation graph")

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
